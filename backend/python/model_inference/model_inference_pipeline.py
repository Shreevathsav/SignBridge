from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
from dotenv import load_dotenv
from backend.python.app import RequestDAO
import os

class ModelInference:

    def __init__(self):
        load_dotenv()  # Loads from .env file
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        login(token=hf_token)
        self.processor = AutoProcessor.from_pretrained("google/gemma-3n-E4B")
        self.model = AutoModelForImageTextToText.from_pretrained("google/gemma-3n-E4B")
    

    def inference(self, request_dao: RequestDAO):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ]
            },
            {
                "role": "user", "content": [
                    {"type": "image", "url": str(request_dao.input)},
                    {"type": "text", "text": request_dao.question},
                ]
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenizer=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True
        )

        generate_ids = self.model.generate(**inputs)
        print(self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
