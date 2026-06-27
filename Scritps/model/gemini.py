# code base: https://ai.google.dev/gemini-api/docs/vision?hl=zh-cn&lang=python

import os
from google import genai
from base import BaseModel
from PIL import Image
import mimetypes
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure API key from environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it before running this script.")

# Create a global client with the API key
_client = genai.Client(api_key=api_key)

class EvalGemini(BaseModel):
    
    def __init__(self, model_name = "gemini", low_res = False, upload = False, prompt = None, api_key = "gemini-2.0-flash"):
        
        """
        Args:
            model_name: str, baseline model to use
            low_res: bool, whether to use low resolution images
            upload: bool, whether to upload the image to server, gemini can only process images from server when input images is too much
            prompt: str, whether to constraint the model output space
            api_key: str, api key of gemini to use
        """
        
        self.model_name = model_name
        self.low_res = low_res
        self.upload = upload
        self.prompt = prompt
        self.api_key = api_key
        self.client = _client

        self.exist_images = {}

    def encode_image(self, image_path):
        
        # Load image with PIL
        image = Image.open(image_path)
        
        if self.low_res:
            image = image.resize((image.size[0]//self.scale_factor, image.size[1]//self.scale_factor))
        
        return image

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

        inputs = [question] + images

        # Add delay to avoid rate limiting
        time.sleep(2)
        
        return self._generate_with_retry(inputs)
    
    def generate_batch(self, questions, images):
        """
        Process multiple questions with their corresponding images in a single API call.
        Args:
            questions: list of str, list of questions to ask
            images: list of PIL Images or image objects, one per question
        Returns:
            str: formatted responses as "Question 0: response\nQuestion 1: response\n..."
        """
        if self.prompt is not None:
            formatted_questions = [f"Question {i}: {q}\n{self.prompt}" for i, q in enumerate(questions)]
        else:
            formatted_questions = [f"Question {i}: {q}" for i, q in enumerate(questions)]
        
        # Build batch input: interleave questions and images
        batch_input = []
        for i, (question, image) in enumerate(zip(formatted_questions, images)):
            batch_input.append(question)
            batch_input.append(image)
        
        # Add delay to avoid rate limiting
        time.sleep(2)
        
        return self._generate_with_retry(batch_input)
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
    def _generate_with_retry(self, inputs):
        """Generate content with automatic retry on rate limit errors"""
        return self.client.models.generate_content(
            model=self.api_key,
            contents=inputs
        ).text
    
if __name__ == "__main__":
    
    model = EvalGemini()
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