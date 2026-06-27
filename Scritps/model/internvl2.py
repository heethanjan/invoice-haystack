# code base: https://github.com/OpenGVLab/InternVL

from transformers import AutoTokenizer, AutoModel
import os
import torch
from PIL import Image
from base import BaseModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# Monkey-patch to fix InternVL compatibility with newer transformers
def _patch_internvl_model():
    from transformers.modeling_utils import PreTrainedModel
    original_finalize = PreTrainedModel._finalize_model_loading
    original_getattr = PreTrainedModel.__getattribute__
    
    def patched_finalize(cls, model, load_config, loading_info):
        try:
            # Try the original method
            return original_finalize(model, load_config, loading_info)
        except AttributeError as e:
            if "all_tied_weights_keys" in str(e):
                # Skip the problematic check for models that don't have this attribute
                return loading_info
            raise
    
    def patched_getattr(self, name):
        try:
            return original_getattr(self, name)
        except AttributeError as e:
            if name == "all_tied_weights_keys" and "InternVLChatModel" in str(type(self)):
                # Return empty dict for models that don't have this attribute
                return {}
            raise
    
    PreTrainedModel._finalize_model_loading = classmethod(patched_finalize)
    PreTrainedModel.__getattribute__ = patched_getattr

_patch_internvl_model()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=4):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class EvalInternVL2(BaseModel):

    def __init__(self, model_name = "internvl2", pretrained = "OpenGVLab/InternVL3-8B", low_res = False, scale_factor = 4, prompt = None, max_images_per_batch = 5):
        
        """
        Args:
            model_name: str, baseline model to use
            pretrained: str, path to the pretrained model
            low_res: bool, whether to use low resolution images
            scale_factor: int, scale factor to resize the image
            prompt: str, whether to constraint the model output space
            max_images_per_batch: int, maximum number of images to process at once (default: 4, safe for 80GB GPU)
        """

        self.model_name = model_name
        self.low_res = low_res
        self.scale_factor = scale_factor
        self.prompt = prompt
        self.max_images_per_batch = max_images_per_batch
        self.input_size = 448 if not low_res else 448 // scale_factor

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True, use_fast=False)
        
        # Patch to avoid meta tensor issue with InternVL
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        try:
            # Load with 4-bit quantization to reduce memory usage
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModel.from_pretrained(
                pretrained,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map='auto'
            )
            self.model = self.model.eval()
        except (RuntimeError, NotImplementedError) as e:
            if "meta tensors" in str(e) or "Cannot copy out of meta tensor" in str(e):
                # Fallback: load on CPU without quantization
                self.model = AutoModel.from_pretrained(
                    pretrained,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map='cpu'
                )
                self.model = self.model.eval()
            else:
                raise

        self.generation_config = dict(max_new_tokens=128, do_sample=False)

    def encode_image(self, image_path):
        
        pixel_values = load_image(image_path, input_size=self.input_size, max_num=12)
        return pixel_values

    def batch_image(self, image_path) -> list:
        
        assert type(image_path) == list or os.path.isdir(image_path), f"image_path should be a directory or a list of image paths, but given {image_path}"
        
        if type(image_path) == list:
            images = image_path
        else:
            images = [os.path.join(image_path, image) for image in os.listdir(image_path)]
        
        # Limit number of images to prevent OOM errors
        if len(images) > self.max_images_per_batch:
            print(f"Warning: Processing only {self.max_images_per_batch} out of {len(images)} images to prevent OOM errors.")
            images = images[:self.max_images_per_batch]

        # Encode all images and collect pixel values
        pixel_values_list = []
        for image in images:
            pixel_values = self.encode_image(image)
            pixel_values_list.append(pixel_values)

        # Concatenate all pixel values
        self.pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).cuda()
        
        return images

    @torch.no_grad()
    def generate(self, question, images):
        """
        Args:
            question: str, question to ask
            images: list, list of multiple image features
        """
        if self.prompt is not None:
            question = f"{question}\n{self.prompt}"

        # Create the conversation format expected by InternVL
        # InternVL uses <image> placeholders in the text
        num_patches = self.pixel_values.shape[0]
        image_tokens = '<image>' * num_patches
        
        question_with_images = f"{image_tokens}\n{question}"
        
        # Tokenize the question
        response = self.model.chat(
            self.tokenizer,
            self.pixel_values,
            question_with_images,
            self.generation_config
        )
        
        # Clear GPU cache to free memory
        torch.cuda.empty_cache()

        return response
    
    @torch.no_grad()
    def generate_batch(self, questions_list, image_paths):
        """
        Batch process multiple questions with the same set of images.
        
        Args:
            questions_list: list of str, list of all questions
            image_paths: list, list of image file paths
        """
        responses = []
        
        print(f"Processing {len(questions_list)} questions in batch with {len(image_paths)} images...")
        
        # Load all images and combine pixel values
        pixel_values_list = []
        for image_path in image_paths:
            if isinstance(image_path, str):
                pixel_values = self.encode_image(image_path)
            else:
                # Assume it's already encoded
                pixel_values = image_path
            pixel_values_list.append(pixel_values)
        
        # Concatenate all pixel values
        combined_pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).cuda()
        
        # Process questions in batches to avoid OOM
        batch_size = 4  # Process 4 questions at a time
        for batch_start in range(0, len(questions_list), batch_size):
            batch_end = min(batch_start + batch_size, len(questions_list))
            batch_questions = questions_list[batch_start:batch_end]
            
            # Process each question in this batch
            for question in batch_questions:
                if self.prompt is not None:
                    full_question = f"{question}\n{self.prompt}"
                else:
                    full_question = question
                
                # Create the conversation format expected by InternVL
                num_patches = combined_pixel_values.shape[0]
                image_tokens = '<image>' * num_patches
                
                question_with_images = f"{image_tokens}\n{full_question}"
                
                # Generate response
                response = self.model.chat(
                    self.tokenizer,
                    combined_pixel_values,
                    question_with_images,
                    self.generation_config
                )
                
                responses.append(response)
            
            # Clear GPU cache after each batch
            torch.cuda.empty_cache()
        
        return responses
    
if __name__ == "__main__":
    
    model = EvalInternVL3_5(low_res = True, scale_factor = 4)
    images = model.batch_image("./img")
    
    print("\n***************************Instance1*********************************\n")
    q = "Find the dog and tell me what breed is this dog in this set of images"
    resps = model.generate(q, images)
    print(f"Question: {q}\nAnswer: {resps}")

    print("\n***************************Instance2*********************************\n")
    q = "Where does the bear sit in?"
    resps = model.generate(q, images)
    print(f"Question: {q}\nAnswer: {resps}")

    print("\n***************************Instance3*********************************\n")
    q = "Is there a white horse in this set of image?"
    resps = model.generate(q, images)
    print(f"Question: {q}\nAnswer: {resps}")

    print("\n***************************Instance4*********************************\n")
    q = "Is there a black horse in this set of image?"
    resps = model.generate(q, images)
    print(f"Question: {q}\nAnswer: {resps}")
