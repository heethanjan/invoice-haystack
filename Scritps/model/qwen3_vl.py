# code base: https://github.com/QwenLM/Qwen3-VL

from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import os
import torch
from PIL import Image
from base import BaseModel

class EvalQwen3VL(BaseModel):

    def __init__(self, model_name = "qwen3_vl", pretrained = "Qwen/Qwen3-VL-8B-Instruct", low_res = False, scale_factor = 4, prompt = None, max_images_per_batch = 500):
        
        """
        Args:
            model_name: str, baseline model to use
            pretrained: str, path to the pretrained model
            low_res: bool, whether to use low resolution images
            scale_factor: int, scale factor to resize the image
            prompt: str, whether to constraint the model output space
            max_images_per_batch: int, maximum number of images to process at once (default: 10)
        """

        self.model_name = model_name
        self.low_res = low_res
        self.scale_factor = scale_factor
        self.prompt = prompt
        self.max_images_per_batch = max_images_per_batch

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype = torch.bfloat16,
            device_map = "auto",
        )
        self.model.eval()

        self.template = [
            {
                "role": "user",
                "content": [],
            }
        ]

    def encode_image(self, image_path):
        
        return Image.open(image_path).convert("RGB")

    def batch_image(self, image_path) -> list:
        
        # clear cached images
        self.template = [
            {
                "role": "user",
                "content": [],
            }
        ]
        
        assert type(image_path) == list or os.path.isdir(image_path), f"image_path should be a directory or a list of image paths, but given {image_path}"
        
        if type(image_path) == list:
            images = image_path
        else:
            images = [os.path.join(image_path, image) for image in os.listdir(image_path)]
        
        # Limit number of images to prevent OOM errors
        if len(images) > self.max_images_per_batch:
            print(f"Warning: Processing only {self.max_images_per_batch} out of {len(images)} images to prevent OOM errors.")
            images = images[:self.max_images_per_batch]

        for image in images:
            image = self.encode_image(image)
            if self.low_res:
                self.template[0]["content"].append(
                    {
                        "type": "image",
                        "image": image,
                        "resized_width": image.size[0] // self.scale_factor,
                        "resized_height": image.size[1] // self.scale_factor,
                    }
                )
            else:
                self.template[0]["content"].append(
                    {
                        "type": "image",
                        "image": image,
                    }
                )

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

        self.template[0]["content"].append(
            {
                "type": "text",
                "text": question
            }
        )
        text = self.processor.apply_chat_template(self.template, tokenize = False, add_generation_prompt = True)

        inputs = self.processor(
            text = [text],
            images = images,
            videos = None,
            padding = True,
            return_tensors = "pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens = 128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        text_outputs = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        self.template[0]["content"].pop()
        
        # Clear GPU cache to free memory
        torch.cuda.empty_cache()

        return text_outputs[0]
    
    @torch.no_grad()
    def generate_batch(self, questions_list, images):
        """
        Batch process multiple questions with the same set of images.
        
        Args:
            questions_list: list of str, list of all questions
            images: list, list of image paths or PIL images
        """
        responses = []
        
        print(f"Processing {len(questions_list)} questions in batch with {len(images)} images...")
        
        # Process questions in batches to avoid OOM
        batch_size = 4  # Process 4 questions at a time with all images
        for batch_start in range(0, len(questions_list), batch_size):
            batch_end = min(batch_start + batch_size, len(questions_list))
            batch_questions = questions_list[batch_start:batch_end]
            
            # Clear template and reload images for each batch
            self.template = [
                {
                    "role": "user",
                    "content": [],
                }
            ]
            
            # Add images to template
            for image_path in images:
                if isinstance(image_path, str):
                    image = self.encode_image(image_path)
                else:
                    image = image_path
                
                if self.low_res:
                    self.template[0]["content"].append(
                        {
                            "type": "image",
                            "image": image,
                            "resized_width": image.size[0] // self.scale_factor,
                            "resized_height": image.size[1] // self.scale_factor,
                        }
                    )
                else:
                    self.template[0]["content"].append(
                        {
                            "type": "image",
                            "image": image,
                        }
                    )
            
            # Process each question in the batch
            for question in batch_questions:
                if self.prompt is not None:
                    full_question = f"{question}\n{self.prompt}"
                else:
                    full_question = question
                
                # Create a copy of template for this question
                template_copy = [
                    {
                        "role": "user",
                        "content": self.template[0]["content"].copy() + [
                            {
                                "type": "text",
                                "text": full_question
                            }
                        ]
                    }
                ]
                
                text = self.processor.apply_chat_template(template_copy, tokenize = False, add_generation_prompt = True)
                
                # Get images from template
                template_images = [content["image"] for content in self.template[0]["content"] if content["type"] == "image"]
                
                inputs = self.processor(
                    text = [text],
                    images = template_images,
                    videos = None,
                    padding = True,
                    return_tensors = "pt",
                )
                inputs = inputs.to("cuda")
                
                generated_ids = self.model.generate(**inputs, max_new_tokens = 128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                text_outputs = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
                responses.append(text_outputs[0])
            
            # Clear GPU cache after each batch
            torch.cuda.empty_cache()
        
        return responses
    
if __name__ == "__main__":
    
    model = EvalQwen3VL(low_res = True, scale_factor = 4)
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
