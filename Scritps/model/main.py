import argparse
import os
import json
from tqdm import tqdm

def load_dataset(anns_path, v_RAG = False, retrival_path = ''):
    
    sources = json.load(open(anns_path, 'r'))
    preprocesses = []
    for source in sources:
        
        retrieval = json.load(open(os.path.join(retrival_path, f"{source['id']}.json"), 'r')) if v_RAG else None

        if v_RAG:
            assert '.'.join(source["pos_image"][0].split('.')[:-1]) == retrieval["real_positive_image"], f"paired image should be the same for the same question, but given {source['pos_image'][0]} and {retrieval['real_positive_image']}"
            preprocesses.append({
                "question": source["conversations"][0]["value"],
                "ground_truth": source["conversations"][1]["value"],
                "imageId": source["pos_image"][0],
                "retrieved_image": retrieval["top_10_images"],
                "id": source["id"]
            })
        else:       
            preprocesses.append({
                "question": source["conversations"][0]["value"],
                "ground_truth": source["conversations"][1]["value"],
                "imageId": source["pos_image"][0],
                "id": source["id"]
            })
            
    return preprocesses
    
def loop_whole_images(dataset, image_path):
    
    ann_path = os.path.join(os.path.dirname(image_path), "image_id_classify.json")
    with open(ann_path, 'r') as fp:
        anns = json.load(fp)

    images = [os.path.join(image_path, ann["image_id"]) for ann in anns if dataset in ann["folders"]]

    return images

def set_seed(seed):
    """Set the seed for reproducibility."""

    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main(args):
    
    # seed everything
    set_seed(args.seed)
    
    image_path = os.path.join(args.image_path, args.dataset.replace('-', '_'))

    # load dataset
    sources = load_dataset(args.anns_path, args.v_RAG, args.retrival_path)
    targets = []
    
    # load model with conditional imports
    if args.model_name == "gpt4o" or args.model_name == "gpt5":
        from gpt4o import EvalGPT4O
        model = EvalGPT4O(args.model_name, args.low_res, args.prompt)
    elif args.model_name == "qwen3_vl":
        from qwen3_vl import EvalQwen3VL
        model = EvalQwen3VL(args.model_name, args.pretrained, args.low_res, args.scale_factor, args.prompt, args.max_images_per_batch)
    elif args.model_name == "internvl2":
        from internvl2 import EvalInternVL2
        model = EvalInternVL2(args.model_name, args.pretrained, args.low_res, args.scale_factor, args.prompt, args.max_images_per_batch)
    elif args.model_name == "gemini":
        from gemini import EvalGemini
        model = EvalGemini(args.model_name, args.low_res, args.upload, args.prompt)
    else:
        raise NotImplementedError
    
    if args.v_RAG:
        
        # response per question
        for source in tqdm(sources):
        
            if args.debug and len(targets) >= 3:
                break

            # process image, every question retrieve different image
            # Support both formats: with and without file extensions
            imgs_path = []
            for img_path in source["retrieved_image"][:args.topk]:
                # First try the path as-is (e.g., 'image.png' or 'image')
                full_path = os.path.join(image_path, img_path)
                if os.path.exists(full_path):
                    imgs_path.append(full_path)
                else:
                    # If not found, try adding common extensions (handles names without extensions)
                    found = False
                    for ext in ['.png', '.jpg', '.jpeg']:
                        full_path_with_ext = os.path.join(image_path, img_path + ext)
                        if os.path.exists(full_path_with_ext):
                            imgs_path.append(full_path_with_ext)
                            found = True
                            break
                    if not found:
                        # If still not found, append original path (will fail with informative error)
                        imgs_path.append(full_path)
            images = model.batch_image(imgs_path)
            assert len(images) <= args.topk, f"images should be {args.topk}, but given {len(images)}"
            response = model.generate(source["question"], images)
            
            targets.append({
                "id": source["id"],
                "imageId": source["imageId"],
                "retrieved_image": [img_path for img_path in source["retrieved_image"][:args.topk]],
                "question": source["question"],
                "ground_truth": source["ground_truth"],
                "response": response
            })
        
        save_path = os.path.join(args.outpath, f"{args.dataset}_{args.model_name}_top_{args.topk}.json") if not args.debug else os.path.join(args.outpath, f"{args.dataset}_{args.model_name}_top_{args.topk}_debug.json")
        
    else:

        # Zero-shot: For each question, load only its specific target image
        # Batch mode: load all images from folder and send with all questions in one API call
        if args.batch and (args.model_name == "gemini" or args.model_name == "gpt4o" or args.model_name == "gpt5"):
            print(f"Processing in batch mode: loading all images from folder for {args.model_name}...")
            
            # First, load ALL images from the image_path folder
            all_folder_images = []
            if os.path.isdir(image_path):
                image_files = sorted([f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"Found {len(image_files)} images in folder: {image_path}")
                
                for img_file in image_files:
                    img_path = os.path.join(image_path, img_file)
                    
                    # Load and encode image based on model type
                    if args.model_name == "gpt4o" or args.model_name == "gpt5":
                        # For GPT: encode to base64
                        encoded = model.encode_image(img_path)
                        all_folder_images.append(encoded)
                    else:
                        # For Gemini: use PIL Image
                        from PIL import Image
                        image = Image.open(img_path)
                        if args.low_res:
                            image = image.resize((image.size[0]//4, image.size[1]//4))
                        all_folder_images.append(image)
            
            # Now pair each question with ALL images from the folder
            all_questions = []
            all_images = []
            image_id_map = []
            
            for source in sources:
                if args.debug and len(all_questions) >= 3:
                    break
                
                # Add the question
                all_questions.append(source["question"])
                # Add all images from the folder for this question
                all_images.extend(all_folder_images)
                image_id_map.append(source)
            
            # Send all images and questions in one API call
            print(f"Sending {len(all_questions)} questions with {len(all_folder_images)} images per question ({len(all_images)} total images) in a single API call...")
            batch_response = model.generate_batch(all_questions, all_images)
            
            # Parse responses - should be numbered as "Question 0:", "Question 1:", etc.
            response_lines = batch_response.split('\n')
            question_responses = {}
            current_question_idx = None
            current_response = []
            
            for line in response_lines:
                # Check if this line starts a new question response
                if line.startswith("Question "):
                    if current_question_idx is not None:
                        question_responses[current_question_idx] = '\n'.join(current_response).strip()
                    # Extract question number
                    try:
                        q_idx = int(line.split("Question ")[1].split(":")[0])
                        current_question_idx = q_idx
                        current_response = [line]
                    except:
                        current_response.append(line)
                else:
                    if current_question_idx is not None:
                        current_response.append(line)
            
            # Add the last response
            if current_question_idx is not None:
                question_responses[current_question_idx] = '\n'.join(current_response).strip()
            
            # Build targets from responses
            for idx, source in enumerate(image_id_map):
                response = question_responses.get(idx, batch_response)  # Fallback to full response if parsing fails
                targets.append({
                    "id": source["id"],
                    "imageId": source["imageId"],
                    "question": source["question"],
                    "ground_truth": source["ground_truth"],
                    "response": response
                })
        
        else:
            # For local GPU models (internvl2, qwen3_vl): load all images once, process each question with all images
            if args.model_name == "internvl2" or args.model_name == "qwen3_vl":
                print(f"Processing with {args.model_name}: loading all images from folder once, processing each question sequentially...")
                
                # Get all image file paths from the folder
                all_folder_image_paths = []
                if os.path.isdir(image_path):
                    image_files = sorted([f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    print(f"Found {len(image_files)} images in folder: {image_path}")
                    all_folder_image_paths = [os.path.join(image_path, img_file) for img_file in image_files]
                
                # Process each question one by one with all images
                for source in tqdm(sources):
                    if args.debug and len(targets) >= 3:
                        break
                    
                    # Load all images for this question
                    model.batch_image(all_folder_image_paths)
                    response = model.generate(source["question"], [])  # Empty list since images are loaded in batch_image
                    
                    targets.append({
                        "id": source["id"],
                        "imageId": source["imageId"],
                        "question": source["question"],
                        "ground_truth": source["ground_truth"],
                        "response": response
                    })
            
            else:
                # Standard mode: response per question (one API call per question, only specific image)
                for source in tqdm(sources):
                
                    if args.debug and len(targets) >= 3:
                        break
                    
                    # Load the specific target image for this question
                    img_file = source["imageId"]
                    img_path = os.path.join(image_path, img_file)
                    
                    # Handle case where extension might be missing
                    if not os.path.exists(img_path):
                        for ext in ['.png', '.jpg', '.jpeg']:
                            if os.path.exists(os.path.join(image_path, img_file.split('.')[0] + ext)):
                                img_path = os.path.join(image_path, img_file.split('.')[0] + ext)
                                break
                    
                    images = model.batch_image([img_path])
                    response = model.generate(source["question"], images)
                    
                    targets.append({
                        "id": source["id"],
                        "imageId": source["imageId"],
                        "question": source["question"],
                        "ground_truth": source["ground_truth"],
                        "response": response
                    })
    
        save_path = os.path.join(args.outpath, f"{args.dataset}_{args.model_name}_top_{args.topk}.json") if args.v_RAG else os.path.join(args.outpath, f"{args.dataset}_{args.model_name}.json")
        save_path = save_path if not args.debug else save_path.replace('.json', '_debug.json')
    
    os.makedirs(args.outpath, exist_ok = True)
    with open(save_path, 'w') as fp:
        json.dump(targets, fp, indent = 2)


if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type = str, default = "gpt4o")
    args.add_argument("--low_res", action = "store_true", help = "using low resolution image")
    args.add_argument("--scale_factor", type = int, default = 4, help = "scale factor for low resolution, working when low_res is True")
    args.add_argument("--no_patch", action = "store_true", help = "not using patches in llava onevision (to include more images)")
    args.add_argument("--upload", action = "store_true", help = "upload the images to the server for gemini when the input images is too much")
    args.add_argument("--batch", action = "store_true", help = "batch process all images and questions in a single API call (for gemini)")
    args.add_argument("--v_RAG", action = "store_true", help = "vision-centric retrieval augmented VQA")
    args.add_argument("--topk", type = int, default = 5, help = "answer with the top k images")
    args.add_argument("--prompt", type = str, default = None, help = "if you want to format the output, set it as: Answer the question using a single word or phrase.")
    args.add_argument("--debug", action = "store_true")
    args.add_argument("--seed", type = int, default = 42, help = "set seed for reproducibility")
    args.add_argument("--anns_path", type = str, default = "./data/test_docVQA.json")
    args.add_argument("--image_path", type = str, default = "./data/Test")
    args.add_argument("--retrival_path", type = str, default = None, help = "path to the retrival images")
    args.add_argument("--pretrained", type = str, default = None, help = "path to the pretrained model if needed")
    args.add_argument("--max_images_per_batch", type = int, default = 10, help = "maximum number of images to process at once to prevent OOM errors")
    args.add_argument("--outpath", type = str, default = "./output")
    args.add_argument(
        "--dataset", 
        choices = ["DocHaystack-100", "DocHaystack-200", "DocHaystack-1000", "InfoHaystack-100", "InfoHaystack-200", "InfoHaystack-1000", "invoiceVQA-500", "invoiceVQA-1000", "invoiceVQA-1500", "invoiceVQA-500-sani", "invoiceVQA-1000-sani", "invoiceVQA-1500-sani"],
        required = True,
        help = "choose benchmarks"
    )

    args = args.parse_args()
    main(args)