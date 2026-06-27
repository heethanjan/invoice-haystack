import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoImageProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration, PreTrainedTokenizerFast
import os
import numpy as np
import json
import argparse
import time
import logging
import open_clip

import copy

# Qwen3-VL model setup - lazy loading
qwen_model_name = "Qwen/Qwen3-VL-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables for lazy loading
qwen_model = None
qwen_processor = None

def load_qwen_model():
    """Lazy load Qwen3 model only when needed"""
    global qwen_model, qwen_processor
    if qwen_model is None:
        print("Loading Qwen3-VL model...")
        try:
            # Try with flash_attention_2 and device_map
            qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                qwen_model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Loading with device_map failed: {e}")
            try:
                # Fallback: try without device_map
                print("Trying without device_map...")
                qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    qwen_model_name,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=True
                )
                qwen_model = qwen_model.to(device)
            except Exception as e2:
                print(f"Flash attention not available, using default: {e2}")
                # Final fallback: basic loading
                qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    qwen_model_name,
                    torch_dtype="auto",
                    trust_remote_code=True
                )
                qwen_model = qwen_model.to(device)
        
        qwen_processor = AutoProcessor.from_pretrained(qwen_model_name, trust_remote_code=True)
        qwen_model.eval()
    return qwen_model, qwen_processor

def load_siglip_tokenizer(model_name):
    """Load a SigLIP-compatible tokenizer with robust fallbacks."""
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"AutoTokenizer failed for SigLIP, trying PreTrainedTokenizerFast: {e}")
        return PreTrainedTokenizerFast.from_pretrained(model_name)

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
            max_new_tokens=128,
            do_sample=False
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

class CombinedSolver:
    def __init__(self, clip_model_name='openai/clip-vit-large-patch14', siglip_model_name='google/siglip-so400m-patch14-384',
                 openclip_model_name='convnext_xxlarge', image_root='dataset', image_dir="image/infor_100",use_question_query=False):
        
        self.image_root = image_root
        self.image_dir = os.path.join(self.image_root, image_dir)
        
        print(f"Loading CLIP model...")
        self.clip_model = AutoModel.from_pretrained(clip_model_name).to(device)
        self.clip_processor = AutoProcessor.from_pretrained(clip_model_name)
        
        print(f"Loading SigLIP model...")
        self.siglip_model = AutoModel.from_pretrained(siglip_model_name).to(device)
        # Use AutoImageProcessor and separate tokenizer for SigLIP
        self.siglip_image_processor = AutoImageProcessor.from_pretrained(siglip_model_name)
        self.siglip_tokenizer = load_siglip_tokenizer(siglip_model_name)
        
        print(f"Loading OpenCLIP model...")
        self.openclip_model, _, self.openclip_preprocess = open_clip.create_model_and_transforms(openclip_model_name, pretrained='laion2b_s34b_b82k_augreg_soup')
        self.openclip_model.eval().to(device)
        self.openclip_tokenizer = open_clip.get_tokenizer(openclip_model_name)

        self.use_question_query = use_question_query
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Cache for embeddings
        self.image_paths = []
        self.image_embeds_clip = None
        self.image_embeds_siglip = None
        self.image_embeds_openclip = None
        
        # Precompute embeddings
        self._precompute_embeddings()
        
    def _precompute_embeddings(self, batch_size=100):
        """Precompute embeddings for all images once at startup."""
        print("Precomputing image embeddings...")
        start_time = time.time()
        
        # Get all image files
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png', ".jpeg"))]
        self.image_paths = [os.path.join(self.image_dir, img) for img in image_files]
        
        print(f"Found {len(self.image_paths)} images. Computing embeddings in batches...")
        
        all_clip_embeds = []
        all_siglip_embeds = []
        all_openclip_embeds = []
        
        for i in range(0, len(self.image_paths), batch_size):
            batch_paths = self.image_paths[i:i + batch_size]
            
            # Load images for CLIP
            images_clip = [Image.open(img).convert("RGB") for img in batch_paths]
            image_inputs_clip = self.clip_processor(images=images_clip, return_tensors="pt").to(self.device)
            
            # Load images for SigLIP
            images_siglip = [Image.open(img).convert("RGB") for img in batch_paths]
            image_inputs_siglip = self.siglip_image_processor(images=images_siglip, return_tensors="pt").to(self.device)
            
            # Load images for OpenCLIP
            images_openclip = [self.openclip_preprocess(Image.open(img).convert("RGB")).to(self.device) for img in batch_paths]
            image_inputs_openclip = torch.stack(images_openclip)
            
            with torch.no_grad():
                # CLIP embeddings
                image_output_clip = self.clip_model.get_image_features(**image_inputs_clip)
                if hasattr(image_output_clip, 'pooler_output'):
                    image_embeds_clip = image_output_clip.pooler_output
                elif isinstance(image_output_clip, torch.Tensor):
                    image_embeds_clip = image_output_clip
                else:
                    image_embeds_clip = image_output_clip[0]
                image_embeds_clip /= image_embeds_clip.norm(dim=1, keepdim=True)
                all_clip_embeds.append(image_embeds_clip.cpu())
                
                # SigLIP embeddings
                image_output_siglip = self.siglip_model.get_image_features(**image_inputs_siglip)
                if hasattr(image_output_siglip, 'pooler_output'):
                    image_embeds_siglip = image_output_siglip.pooler_output
                elif isinstance(image_output_siglip, torch.Tensor):
                    image_embeds_siglip = image_output_siglip
                else:
                    image_embeds_siglip = image_output_siglip[0]
                image_embeds_siglip /= image_embeds_siglip.norm(dim=1, keepdim=True)
                all_siglip_embeds.append(image_embeds_siglip.cpu())
                
                # OpenCLIP embeddings
                image_embeds_openclip = self.openclip_model.encode_image(image_inputs_openclip)
                image_embeds_openclip /= image_embeds_openclip.norm(dim=-1, keepdim=True)
                all_openclip_embeds.append(image_embeds_openclip.cpu())
            
            del images_clip, images_siglip, images_openclip
            del image_inputs_clip, image_inputs_siglip, image_inputs_openclip
            torch.cuda.empty_cache()
            
            print(f"Processed batch {i//batch_size + 1}/{(len(self.image_paths) + batch_size - 1)//batch_size}")
        
        # Combine all embeddings
        self.image_embeds_clip = torch.cat(all_clip_embeds, dim=0).to(self.device)
        self.image_embeds_siglip = torch.cat(all_siglip_embeds, dim=0).to(self.device)
        self.image_embeds_openclip = torch.cat(all_openclip_embeds, dim=0).to(self.device)
        
        elapsed_time = time.time() - start_time
        print(f"Embedding precomputation completed in {elapsed_time:.2f} seconds")
        print(f"Cached embeddings for {len(self.image_paths)} images on GPU")
        
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
            print("No images passed the Qwen filter. Keeping the top-5 images.")
            filtered_images = top_k_images

        elapsed_time = time.time() - start_time
        return filtered_images, elapsed_time

    def get_combined_top_k_images(self, needle_word, pos_image, k=10, batch_size=100):
        start_time = time.time()

        text_inputs_clip = self.clip_processor(text=[needle_word], return_tensors="pt").to(self.device)
        text_inputs_siglip = self.siglip_tokenizer([needle_word], return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
        text_inputs_openclip = self.openclip_tokenizer([needle_word]).to(self.device)

        with torch.no_grad():
            # Extract text embeddings
            text_output_clip = self.clip_model.get_text_features(**text_inputs_clip)
            if hasattr(text_output_clip, 'pooler_output'):
                text_embeds_clip = text_output_clip.pooler_output.squeeze(0)
            elif isinstance(text_output_clip, torch.Tensor):
                text_embeds_clip = text_output_clip.squeeze(0)
            else:
                text_embeds_clip = text_output_clip[0].squeeze(0)
            text_embeds_clip /= text_embeds_clip.norm()

            text_output_siglip = self.siglip_model.get_text_features(**text_inputs_siglip)
            if hasattr(text_output_siglip, 'pooler_output'):
                text_embeds_siglip = text_output_siglip.pooler_output.squeeze(0)
            elif isinstance(text_output_siglip, torch.Tensor):
                text_embeds_siglip = text_output_siglip.squeeze(0)
            else:
                text_embeds_siglip = text_output_siglip[0].squeeze(0)
            text_embeds_siglip /= text_embeds_siglip.norm()

            text_embeds_openclip = self.openclip_model.encode_text(text_inputs_openclip)
            text_embeds_openclip /= text_embeds_openclip.norm(dim=-1, keepdim=True)

            # Use cached embeddings for similarity computation
            cosine_similarities_clip = torch.matmul(self.image_embeds_clip, text_embeds_clip).cpu().numpy().flatten()
            cosine_similarities_siglip = torch.matmul(self.image_embeds_siglip, text_embeds_siglip).cpu().numpy().flatten()
            cosine_similarities_openclip = (self.image_embeds_openclip @ text_embeds_openclip.T).squeeze().cpu().numpy().flatten()

            # Combine scores
            combined_scores = {}
            for idx, img_path in enumerate(self.image_paths):
                combined_score = (cosine_similarities_clip[idx] + cosine_similarities_siglip[idx] +
                                  cosine_similarities_openclip[idx])
                combined_scores[img_path] = combined_score

            top_k_images = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:k]
            top_k_includes_pos = [int(pos_image in top_k_images[:i + 1]) for i in range(k)]

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
        total_llava_filtering_time = 0

        for idx, entry in enumerate(data):
            question = entry["conversations"][0]["value"]

            id = entry["id"]


            if self.use_question_query:
                needle_word = question
            else:
                needle_word = entry["needle"]


            pos_image = os.path.join(self.image_dir, entry["pos_image"][0])

            top_k_images, top_k_includes_pos, combined_time = self.get_combined_top_k_images(needle_word, pos_image, k=10)
            total_combined_time += combined_time

            filtered_images, llava_filtering_time = self.filter_with_qwen(top_k_images, question)
            total_llava_filtering_time += llava_filtering_time


            filtered_images = [img.split("/")[-1] for img in filtered_images]

            pos_image = pos_image.split("/")[-1]

            top_k_includes_pos_filtered = [int(pos_image in filtered_images[:i + 1]) for i in range(len(filtered_images))]

            if top_k_includes_pos_filtered[0]:
                top1_correct += 1
            if any(top_k_includes_pos_filtered[:3]):
                top3_correct += 1
            if any(top_k_includes_pos_filtered[:5]):
                top5_correct += 1
            if any(top_k_includes_pos_filtered[:10]):
                top10_correct += 1


            file_name = id+".json"
            output_file = os.path.join(output_dir, f"{file_name}")

            filtered_images = [img.split("/")[-1] for img in filtered_images]

            pos_image = pos_image.split(".")[0]
            with open(output_file, "w") as f_out:
                json.dump({
                    "question": question,
                    "needle_word": needle_word,
                    "top_10_images": filtered_images[:10],
                    "top_k_includes_pos": top_k_includes_pos_filtered,
                    "real_positive_image": pos_image
                }, f_out, indent=4)

            print(f"Processed entry {idx + 1}/{total_entries}: Saved top 5 images to {output_file}")

        top1_accuracy = top1_correct / total_entries
        top3_accuracy = top3_correct / total_entries
        top5_accuracy = top5_correct / total_entries
        top10_accuracy = top10_correct / total_entries

        avg_combined_time = total_combined_time / total_entries
        avg_llava_filtering_time = total_llava_filtering_time / total_entries

        log_file = os.path.join(output_dir, 'accuracy.log')

    
        with open(log_file, 'w') as log:
            log.write(f"Total Entries: {total_entries}\n")
            log.write(f"Top-1 Accuracy: {top1_accuracy:.2%}\n")
            log.write(f"Top-3 Accuracy: {top3_accuracy:.2%}\n")
            log.write(f"Top-5 Accuracy: {top5_accuracy:.2%}\n")
            log.write(f"Top-10 Accuracy: {top10_accuracy:.2%}\n")
            log.write(f"Average Combined Encoder Inference Time: {avg_combined_time:.4f} seconds\n")
            log.write(f"Average LLaVA Filtering Time: {avg_llava_filtering_time:.4f} seconds\n")

        print(f"Total Entries: {total_entries}")
        print(f"Top-1 Accuracy: {top1_accuracy:.2%}")
        print(f"Top-3 Accuracy: {top3_accuracy:.2%}")
        print(f"Top-5 Accuracy: {top5_accuracy:.2%}")
        print(f"Top-10 Accuracy: {top10_accuracy:.2%}")
        print(f"Average Combined Encoder Inference Time: {avg_combined_time:.4f} seconds")
        print(f"Average Qwen3-VL Filtering Time: {avg_llava_filtering_time:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SiGLIP Solver with adjustable dataset and image directory.")
    parser.add_argument('--dataset_file', type=str, default="dataset/VHs_qa/VHs_full/single_needle/infor_100.json", help="Path to the dataset JSON file")
    parser.add_argument('--image_root', type=str, default="dataset", help="Root directory for images")
    parser.add_argument('--image_dir', type=str, default="image/infor_100", help="Directory for specific images")
    parser.add_argument('--top_k_filter', type=int, default=60, help="Output directory for results")
    parser.add_argument('--output_dir', type=str, default="output/siglip_results", help="Output directory for results")
    parser.add_argument('--use_question_query', action='store_true', help="Use question as the query instead of needle word")
    args = parser.parse_args()

    solver = CombinedSolver(clip_model_name='openai/clip-vit-large-patch14', siglip_model_name="google/siglip-so400m-patch14-384", image_root=args.image_root, image_dir=args.image_dir, use_question_query=args.use_question_query)
    solver.process_dataset(args.dataset_file, args.output_dir, top_k_filter=args.top_k_filter)







