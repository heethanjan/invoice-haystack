import os
import json
import time
import torch
import argparse
import open_clip
from PIL import Image
from transformers import AutoProcessor, AutoModel, Qwen3VLForConditionalGeneration, AutoImageProcessor, AutoTokenizer, PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer

# Qwen3-VL model setup - lazy loading
qwen_model_name = "Qwen/Qwen3-VL-8B-Instruct"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables for lazy loading
qwen_model = None
qwen_processor = None

def load_siglip_tokenizer(model_name):
    """Load a SigLIP-compatible tokenizer with robust fallbacks."""
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"AutoTokenizer failed for SigLIP, trying PreTrainedTokenizerFast: {e}")
        return PreTrainedTokenizerFast.from_pretrained(model_name)

def load_qwen_model():
    """Lazy load Qwen model only when needed"""
    global qwen_model, qwen_processor
    if qwen_model is None:
        print("Loading Qwen3-VL model for filtering...")
        try:
            # Recommended approach: use flash_attention_2 for better speed and memory
            qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                qwen_model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Flash attention not available, using default: {e}")
            # Fallback to default if flash_attention_2 is not available
            qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                qwen_model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
        qwen_processor = AutoProcessor.from_pretrained(qwen_model_name, trust_remote_code=True)
        qwen_model.eval()
    return qwen_model, qwen_processor

def qwen_single_image_inference(image_path, question):
    model, processor = load_qwen_model()
    image = Image.open(image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # Recommended approach: use apply_chat_template with tokenize=True
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,  # Reduced for faster yes/no answers
            do_sample=False
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

class HybridSolver:
    def __init__(self, siglip_model_name='google/siglip-so400m-patch14-384',
                 openclip_model_name='convnext_xxlarge',
                 text_model_name='BAAI/bge-large-en-v1.5',
                 image_root='dataset', image_dir="image/infor_100", 
                 text_dir="text/infor_100",
                 use_question_query=False, use_qwen_filter=True,
                 keep_embeddings_on_gpu=True):
        
        self.image_root = image_root
        self.image_dir = os.path.join(self.image_root, image_dir)
        self.text_dir = os.path.join(self.image_root, text_dir)
        self.keep_embeddings_on_gpu = keep_embeddings_on_gpu
        
        print(f"Loading models on {device}...")
        
        # Load multimodal models (SigLIP and OpenCLIP)
        self.siglip_model = AutoModel.from_pretrained(siglip_model_name).to(device)
        # Load SiGLIP components separately for compatibility with latest transformers
        self.siglip_image_processor = AutoImageProcessor.from_pretrained(siglip_model_name)
        # SiGLIP tokenizer with robust fallbacks
        self.siglip_tokenizer = load_siglip_tokenizer(siglip_model_name)
        
        self.openclip_model, _, self.openclip_preprocess = open_clip.create_model_and_transforms(
            openclip_model_name, pretrained='laion2b_s34b_b82k_augreg_soup'
        )
        self.openclip_model.eval().to(device)
        self.openclip_tokenizer = open_clip.get_tokenizer(openclip_model_name)
        
        # Load text model
        print(f"Loading text embedding model...")
        self.text_model = SentenceTransformer(text_model_name, device=device)

        self.use_question_query = use_question_query
        self.use_qwen_filter = use_qwen_filter
        self.device = device
        
        # Cache for embeddings
        self.document_basenames = []  # Aligned base names (without extensions)
        self.document_image_paths = []  # Aligned image paths
        self.document_text_paths = []   # Aligned text paths
        self.image_embeds_siglip = None
        self.image_embeds_openclip = None
        self.text_embeds = None
        
        # Precompute embeddings
        self._precompute_embeddings()
        
    def filter_with_qwen(self, top_k_images, question):
        start_time = time.time()
        filtered_images = []
        for img_path in top_k_images:
            qwen_prompt = question + " Can this image provide the answer for this question? Answer only yes or no."
            answer = qwen_single_image_inference(img_path, qwen_prompt)
            answer = answer.replace("only answer", "")
            if "yes" in answer.lower():
                filtered_images.append(img_path)
        
        if len(filtered_images) == 0:
            print("No images passed the Qwen filter. Keeping the top-10 images.")
            filtered_images = top_k_images[:10]

        elapsed_time = time.time() - start_time
        return filtered_images, elapsed_time

    def _precompute_embeddings(self, batch_size=100):
        """Precompute embeddings for all images and texts once."""
        print("Precomputing image and text embeddings...")
        start_time = time.time()
        
        # Get image files
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png', ".jpeg"))]
        image_dict = {os.path.splitext(f)[0]: os.path.join(self.image_dir, f) for f in image_files}
        
        # Get text files
        text_files = [f for f in os.listdir(self.text_dir) if f.endswith('.txt')]
        text_dict = {os.path.splitext(f)[0]: os.path.join(self.text_dir, f) for f in text_files}
        
        # Find common base names (documents that exist in both modalities)
        common_basenames = sorted(set(image_dict.keys()) & set(text_dict.keys()))
        
        # Build aligned paths
        self.document_basenames = common_basenames
        self.document_image_paths = [image_dict[bn] for bn in common_basenames]
        self.document_text_paths = [text_dict[bn] for bn in common_basenames]
        
        print(f"Found {len(image_files)} images and {len(text_files)} text files")
        print(f"Aligned {len(common_basenames)} documents that exist in both modalities")
        
        # Precompute image embeddings
        all_siglip_embeds = []
        all_openclip_embeds = []
        
        for i in range(0, len(self.document_image_paths), batch_size):
            batch_paths = self.document_image_paths[i:i + batch_size]
            
            images_siglip = [Image.open(img).convert("RGB") for img in batch_paths]
            image_inputs_siglip = self.siglip_image_processor(images=images_siglip, return_tensors="pt").to(self.device)
            
            images_openclip = [self.openclip_preprocess(Image.open(img).convert("RGB")).to(self.device) for img in batch_paths]
            image_inputs_openclip = torch.stack(images_openclip)
            
            with torch.no_grad():
                image_embeds_siglip = self.siglip_model.get_image_features(**image_inputs_siglip)
                # Handle model output object - extract the actual tensor
                if hasattr(image_embeds_siglip, 'pooler_output'):
                    image_embeds_siglip = image_embeds_siglip.pooler_output
                elif not isinstance(image_embeds_siglip, torch.Tensor):
                    image_embeds_siglip = image_embeds_siglip[0]
                image_embeds_siglip /= image_embeds_siglip.norm(dim=-1, keepdim=True)
                all_siglip_embeds.append(image_embeds_siglip.cpu())
                
                image_embeds_openclip = self.openclip_model.encode_image(image_inputs_openclip)
                image_embeds_openclip /= image_embeds_openclip.norm(dim=-1, keepdim=True)
                all_openclip_embeds.append(image_embeds_openclip.cpu())
            
            del images_siglip, images_openclip, image_inputs_siglip, image_inputs_openclip
            torch.cuda.empty_cache()
            
            print(f"Processed image batch {i//batch_size + 1}/{(len(self.document_image_paths) + batch_size - 1)//batch_size}")
        
        self.image_embeds_siglip = torch.cat(all_siglip_embeds, dim=0)
        self.image_embeds_openclip = torch.cat(all_openclip_embeds, dim=0)
        
        # Precompute text embeddings
        all_text_embeds = []
        for i in range(0, len(self.document_text_paths), batch_size):
            batch_paths = self.document_text_paths[i:i + batch_size]
            texts = [open(txt, 'r', encoding='utf-8', errors='ignore').read() for txt in batch_paths]
            batch_embeddings = self.text_model.encode(texts, convert_to_tensor=True, show_progress_bar=False, device=self.device)
            all_text_embeds.append(batch_embeddings.cpu())
            
            print(f"Processed text batch {i//batch_size + 1}/{(len(self.document_text_paths) + batch_size - 1)//batch_size}")
        
        self.text_embeds = torch.cat(all_text_embeds, dim=0)
        
        # Keep embeddings on GPU if requested for faster retrieval
        if self.keep_embeddings_on_gpu:
            print("Moving embeddings to GPU for faster retrieval...")
            self.image_embeds_siglip = self.image_embeds_siglip.to(self.device)
            self.image_embeds_openclip = self.image_embeds_openclip.to(self.device)
            self.text_embeds = self.text_embeds.to(self.device)
        
        elapsed_time = time.time() - start_time
        print(f"Embedding precomputation completed in {elapsed_time:.2f} seconds")
        print(f"Cached embeddings for {len(self.document_image_paths)} aligned documents")

    def get_combined_top_k_documents(self, needle_word, pos_image, k=10):
        """Get top-k documents using 3 models (2 multimodal + 1 text)."""
        start_time = time.time()
        
        # Prepare text inputs for all models
        text_inputs_siglip = self.siglip_tokenizer([needle_word], return_tensors="pt", padding=True).to(self.device)
        text_inputs_openclip = self.openclip_tokenizer([needle_word]).to(self.device)

        with torch.no_grad():
            # Compute text embeddings for multimodal models
            text_embeds_siglip = self.siglip_model.get_text_features(**text_inputs_siglip)
            # Handle model output object - extract the actual tensor
            if hasattr(text_embeds_siglip, 'pooler_output'):
                text_embeds_siglip = text_embeds_siglip.pooler_output
            elif not isinstance(text_embeds_siglip, torch.Tensor):
                text_embeds_siglip = text_embeds_siglip[0]
            text_embeds_siglip = text_embeds_siglip.squeeze(0)
            text_embeds_siglip /= text_embeds_siglip.norm()

            text_embeds_openclip = self.openclip_model.encode_text(text_inputs_openclip)
            text_embeds_openclip /= text_embeds_openclip.norm(dim=-1, keepdim=True)
            
            # Compute text embedding for text model
            text_query_embed = self.text_model.encode(needle_word, convert_to_tensor=True, show_progress_bar=False, device=self.device)
            text_query_embed = text_query_embed.unsqueeze(0)
            text_query_embed = torch.nn.functional.normalize(text_query_embed, p=2, dim=1)
            
            # Move cached embeddings to GPU if not already there
            if self.keep_embeddings_on_gpu:
                image_embeds_siglip_gpu = self.image_embeds_siglip
                image_embeds_openclip_gpu = self.image_embeds_openclip
                text_embeds_gpu = self.text_embeds
            else:
                image_embeds_siglip_gpu = self.image_embeds_siglip.to(self.device)
                image_embeds_openclip_gpu = self.image_embeds_openclip.to(self.device)
                text_embeds_gpu = self.text_embeds.to(self.device)

            # Calculate image similarities
            similarities_siglip = (image_embeds_siglip_gpu @ text_embeds_siglip.unsqueeze(1)).squeeze(1)
            similarities_openclip = (image_embeds_openclip_gpu @ text_embeds_openclip.T).squeeze(1)
            
            # Calculate text similarities
            similarities_text = torch.matmul(text_embeds_gpu, text_query_embed.T).squeeze(1)
            
            # Combine scores - equal weight for all 3 models
            combined_scores = (
                similarities_siglip + 
                similarities_openclip.squeeze() + 
                similarities_text
            ) / 3.0
            
            # Get top-k indices
            top_k_indices = torch.topk(combined_scores, k=k).indices.cpu().numpy()
            
        # Get image paths (we return images as the primary reference)
        top_k_images = [self.document_image_paths[idx] for idx in top_k_indices]
        
        # Check if positive image is in top-k (ignoring extensions)
        pos_image_base = os.path.splitext(pos_image)[0]  # Remove extension
        top_k_includes_pos = []
        
        for i in range(k):
            # Use document_basenames for comparison
            top_k_base_names = [self.document_basenames[top_k_indices[j]] for j in range(i+1)]
            top_k_includes_pos.append(int(pos_image_base in top_k_base_names))

        elapsed_time = time.time() - start_time
        return top_k_images, top_k_includes_pos, elapsed_time

    def process_dataset(self, dataset_file, output_dir, top_k_filter=20):
        with open(dataset_file, 'r') as f:
            data = json.load(f)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        total_entries = len(data)
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        top10_correct = 0

        total_combined_time = 0
        total_qwen_filtering_time = 0

        for idx, entry in enumerate(data):
            question = entry.get("conversations", [{}])[0].get("value", "")
            id = entry.get("id", "unknown_id")

            if self.use_question_query:
                needle_word = question
            else:
                needle_word = entry.get("needle", "")

            pos_image = entry.get("pos_image", [None])[0]
            if not pos_image:
                print(f"Warning: Missing 'pos_image' for entry {id}")
                continue

            pos_image_full_path = os.path.join(self.image_dir, pos_image)

            print(f"Processing entry {idx + 1}/{total_entries}")

            # Get top-k using all 4 models
            top_k_images, top_k_includes_pos, combined_time = self.get_combined_top_k_documents(
                needle_word, pos_image, k=top_k_filter
            )
            total_combined_time += combined_time

            # Apply Qwen filter if enabled
            if self.use_qwen_filter:
                top_k_images, qwen_time = self.filter_with_qwen(top_k_images[:top_k_filter], question)
                total_qwen_filtering_time += qwen_time
            else:
                top_k_images = top_k_images[:10]
                qwen_time = 0

            # Convert to base names for comparison (without extensions)
            filtered_images_basenames = [os.path.splitext(os.path.basename(img))[0] for img in top_k_images]
            pos_image_basename = os.path.splitext(pos_image)[0]

            # Calculate accuracy metrics - check if pos_image is in each top-k
            top_k_includes_pos_filtered = [int(pos_image_basename in filtered_images_basenames[:i + 1]) for i in range(len(filtered_images_basenames))]

            # Update counters
            if top_k_includes_pos_filtered[0]:
                top1_correct += 1
            if any(top_k_includes_pos_filtered[:3]):
                top3_correct += 1
            if any(top_k_includes_pos_filtered[:5]):
                top5_correct += 1
            if any(top_k_includes_pos_filtered[:10]):
                top10_correct += 1

            # Save results
            file_name = id + ".json"
            output_file = os.path.join(output_dir, file_name)

            top_k_images_basenames = [os.path.splitext(os.path.basename(img))[0] for img in top_k_images]

            with open(output_file, "w") as f_out:
                json.dump({
                    "question": question,
                    "needle_word": needle_word,
                    "top_10_images": top_k_images_basenames[:10],
                    "top_k_includes_pos": top_k_includes_pos_filtered,
                    "real_positive_image": pos_image_basename,
                    "combined_time": combined_time,
                    "qwen_filtering_time": qwen_time
                }, f_out, indent=4)

            print(f"Processed entry {idx + 1}/{total_entries}: Top-1 match: {top_k_includes_pos_filtered[0] if len(top_k_includes_pos_filtered) > 0 else 0}")

        top1_accuracy = top1_correct / total_entries
        top3_accuracy = top3_correct / total_entries
        top5_accuracy = top5_correct / total_entries
        top10_accuracy = top10_correct / total_entries

        avg_combined_time = total_combined_time / total_entries
        avg_qwen_filtering_time = total_qwen_filtering_time / total_entries

        log_file = os.path.join(output_dir, 'accuracy.log')

        with open(log_file, 'w') as log:
            log.write(f"Total Entries: {total_entries}\n")
            log.write(f"Top-1 Accuracy: {top1_accuracy:.2%}\n")
            log.write(f"Top-3 Accuracy: {top3_accuracy:.2%}\n")
            log.write(f"Top-5 Accuracy: {top5_accuracy:.2%}\n")
            log.write(f"Top-10 Accuracy: {top10_accuracy:.2%}\n")
            log.write(f"Average Combined Encoder (3 models) Inference Time: {avg_combined_time:.4f} seconds\n")
            log.write(f"Average Qwen2-VL Filtering Time: {avg_qwen_filtering_time:.4f} seconds\n")

        print(f"\n=== Final Results ===")
        print(f"Total Entries: {total_entries}")
        print(f"Top-1 Accuracy: {top1_accuracy:.2%}")
        print(f"Top-3 Accuracy: {top3_accuracy:.2%}")
        print(f"Top-5 Accuracy: {top5_accuracy:.2%}")
        print(f"Top-10 Accuracy: {top10_accuracy:.2%}")
        print(f"Average Combined Encoder (3 models) Inference Time: {avg_combined_time:.4f} seconds")
        print(f"Average Qwen2-VL Filtering Time: {avg_qwen_filtering_time:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid Retrieval with 2 multimodal + 1 text model.")
    parser.add_argument('--dataset_file', type=str, default="dataset/VHs_qa/VHs_full/single_needle/infor_100.json", help="Path to the dataset JSON file")
    parser.add_argument('--image_root', type=str, default="dataset", help="Root directory for images and texts")
    parser.add_argument('--image_dir', type=str, default="image/infor_100", help="Directory for images")
    parser.add_argument('--text_dir', type=str, default="text/infor_100", help="Directory for text files")
    parser.add_argument('--top_k_filter', type=int, default=10, help="Number of top results before Qwen filtering")
    parser.add_argument('--output_dir', type=str, default="output/hybrid_results", help="Output directory for results")
    parser.add_argument('--use_question_query', action='store_true', help="Use question as the query instead of needle word")
    parser.add_argument('--disable_qwen_filter', action='store_true', help="Disable Qwen filtering")
    args = parser.parse_args()

    solver = HybridSolver(
        siglip_model_name="google/siglip-so400m-patch14-384",
        openclip_model_name='convnext_xxlarge',
        text_model_name='BAAI/bge-large-en-v1.5',
        image_root=args.image_root, 
        image_dir=args.image_dir,
        text_dir=args.text_dir,
        use_question_query=args.use_question_query,
        use_qwen_filter=not args.disable_qwen_filter,
        keep_embeddings_on_gpu=True  # Keep embeddings on GPU for faster retrieval
    )
    solver.process_dataset(args.dataset_file, args.output_dir, top_k_filter=args.top_k_filter)
