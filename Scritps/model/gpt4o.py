# code base: https://platform.openai.com/docs/guides/vision

import os
import base64
from openai import OpenAI
from PIL import Image
from io import BytesIO
from base import BaseModel

class EvalGPT4O(BaseModel):
    
    def __init__(self, model_name = "gpt-5.2", low_res = False, prompt = None, api_key = "gpt-5", resize_scale = 0.5):
        
        """
        Args:
            model_name: str, baseline model to use
            low_res: bool, whether to use low resolution images
            prompt: str, whether to constraint the model output space
            api_key: str, api key of gpt to use
            resize_scale: float, scale factor for resizing images (0.5 = 50% of original size)
        """
        
        self.model_name = model_name
        self.low_res = low_res
        self.prompt = prompt
        self.api_key = api_key
        self.resize_scale = resize_scale
        self.model = OpenAI()

    def encode_image(self, image_path):
        
        # Open and resize image
        image = Image.open(image_path).convert("RGB")
        new_size = (int(image.width * self.resize_scale), int(image.height * self.resize_scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return image_base64

    def batch_image(self, image_path) -> list:
        
        assert type(image_path) == list or os.path.isdir(image_path), f"image_path should be a directory or a list of image paths, but given {image_path}"
        
        if type(image_path) == list:
            images = image_path
        else:
            images = [os.path.join(image_path, image) for image in os.listdir(image_path)]
        
        images = [self.encode_image(image) for image in images]

        return images

    def generate(self, question, images):
        """
        Args:
            question: str, question to ask
            images: list, list of multiple image features
        """
        if self.prompt is not None:
            question = f"{question}\n{self.prompt}"

        text_content = [{"type": "text", "text": question}]
        if self.low_res:
            image_contents = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpg;base64,{image}",
                        "detail": "low"
                    },
                }
                for image in images
            ]
        else:
            image_contents = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpg;base64,{image}"
                    },
                }
                for image in images
            ]
        
        completion = self.model.chat.completions.create(
            model = self.api_key,
            messages=[
                {
                    "role": "user",
                    "content": text_content + image_contents
                }
            ],
        )
        
        return completion.choices[0].message.content
    
    def generate_batch(self, questions, images):
        """
        Process multiple questions with all images from folder in a single API call.
        Args:
            questions: list of str, list of questions to ask
            images: list of base64 encoded images (all 500 images repeated for each question)
        Returns:
            str: formatted responses as "Question 0: response\nQuestion 1: response\n..."
        """
        # Build the prompt with all questions
        batch_prompt = ""
        for i, q in enumerate(questions):
            batch_prompt += f"Question {i}: {q}\n"
        
        if self.prompt is not None:
            batch_prompt += f"\n{self.prompt}"
        
        # Build image contents - for batch, we include all images once and reference them for all questions
        if self.low_res:
            image_contents = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpg;base64,{image}",
                        "detail": "low"
                    },
                }
                for image in images
            ]
        else:
            image_contents = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpg;base64,{image}"
                    },
                }
                for image in images
            ]
        
        text_content = [{"type": "text", "text": batch_prompt}]
        
        completion = self.model.chat.completions.create(
            model = self.api_key,
            messages=[
                {
                    "role": "user",
                    "content": text_content + image_contents
                }
            ],
        )
        
        return completion.choices[0].message.content
    
if __name__ == "__main__":
    
    # Answer the question using a single word or phrase.
    model = EvalGPT4O(low_res = True)
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