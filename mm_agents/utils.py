from typing import List, Dict
import base64
import os
import json
import logging
import time
import re
from PIL import Image
from dotenv import load_dotenv
from io import BytesIO
import tempfile


from openai import OpenAI
import io

logger = logging.getLogger("utilities")

def encode_image(image_path, max_side=1600):
    img = Image.open(image_path)
    width, height = img.size
    if max(width, height) > max_side:
        img.thumbnail((max_side, max_side))
    else:
        logger.info("No resize needed")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
def encoded_img_to_pil_img(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    return image

def save_to_tmp_img_file(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    tmp_img_path = os.path.join(tempfile.mkdtemp(), "tmp_img.png")
    image.save(tmp_img_path)

    return tmp_img_path

def extract_json_from_code_block(response: str):
    logger.info("Extracting JSON from response...")
    content = response
    code_block = re.search(
                r'(?:```|~~~|、、、)(?:json\s*)?(.*?)(?:```|~~~|、、、)', 
                content, 
                re.DOTALL
            )
    json_str = code_block.group(1).strip()
    if json_str.endswith(','):
        json_str = json_str[:-1]
    return json_str


def extract_code_from_code_block(response: str):
    logger.info("Extracting code from response...")
    content = response
    code_block = re.search(
            r'(?:```|~~~|、、、)(?:python\s*)?(.*?)(?:```|~~~|、、、)', 
            content, 
            re.DOTALL
        )
    python_str = code_block.group(1).strip()
    return python_str

def call_llm(engine, messages: List[Dict], max_tokens=1000, top_p=0.9, temperature=0.5, num_responses=1, token_count = False):
    model = engine.strip()
    load_dotenv()
    
    if model.startswith("vllm"):
        VLLM_URL = os.getenv('VLLM_URL', 'http://localhost:8000/v1')
        VLLM_MODEL = os.getenv('VLLM_MODEL', 'Qwen/Qwen2.5-VL-32B-Instruct')
        logger.info(f"Generating content with vLLM deployed model: {VLLM_MODEL}")
        client = OpenAI(base_url=VLLM_URL, api_key="dummy-key")
        
        try:
            resp = client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                n=num_responses
            )
            
            if token_count:
                if num_responses > 1:
                    return [choice.message.content for choice in resp.choices], resp.usage.completion_tokens
                else:
                    return resp.choices[0].message.content, resp.usage.completion_tokens
            else:
                if num_responses > 1:
                    return [choice.message.content for choice in resp.choices]
                else:
                    return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"vLLM call failed: {e}")
            time.sleep(5)
            return "" if num_responses == 1 else [""]
    
    # OpenAI GPT
    elif model.startswith("gpt"):
        logger.info(f"Generating content with GPT model: {model}")
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            return "" if num_responses == 1 else [""]
        
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                n=num_responses
            )
            print(resp.usage.completion_tokens)

            if token_count:
                if num_responses > 1:
                    return [choice.message.content for choice in resp.choices], resp.usage.completion_tokens
                else:
                    return resp.choices[0].message.content, resp.usage.completion_tokens
            else:
                if num_responses > 1:
                    return [choice.message.content for choice in resp.choices]
                else:
                    return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            time.sleep(5)
            return "" if num_responses == 1 else [""] 
    else:
        raise ValueError(f"Unsupported model: {model}")