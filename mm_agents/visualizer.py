import os
import logging
import json
from .utils import call_llm, extract_json_from_code_block, extract_code_from_code_block, encode_image
from .prompts import VisualizerPrompt
from typing import Dict, List

logger = logging.getLogger(__name__)

class DataVisualizer:
    def __init__(self, 
                 model: str = "gpt-4o",
                 max_tokens: int = 1500,
                 top_p: float = 0.9,
                 num_directions: int = 3,
                 temperature: float = 0.5,
                 token_count: bool = False):
        self.engine = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        # Number of visualization directions to generate
        self.num_directions = num_directions
        self.temperature = temperature
        self.token_count = token_count
        self.direction_advisor_prompt = VisualizerPrompt.DIRECTION_ADVISOR
        self.direction_advisor_single_prompt = VisualizerPrompt.DIRECTION_ADVISOR_SINGLE
        self.code_generator_prompt = VisualizerPrompt.CODE_GENERATOR
        self.code_rectifier_prompt = VisualizerPrompt.CODE_RECTIFIER
        self.check_image_prompt = VisualizerPrompt.CHART_QUALITY_CHECKER
    
    def direction_advisor(self, metadata_report: Dict) -> List[Dict]:
        # Generate multiple(braching_factor) directions
        if self.num_directions == 1:
            prompt = self.direction_advisor_single_prompt
        else:
            prompt = self.direction_advisor_prompt.replace("{num_directions}", str(self.num_directions))
        metadata_info = json.dumps(metadata_report, indent=2)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the metadata and introductory information of the dataset:\n\n" + metadata_info
                    }
                ]
            }
        ]
        logger.info("Generating visualization directions...")
        
        if self.token_count:
            response, completion_tokens = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature, token_count=self.token_count)
        else:
            response = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature)
        try:
            # Load directions as JSON
            json_str = extract_json_from_code_block(response)
            logger.info(f"Visualization directions generated: {json_str}")
            directions = json.loads(json_str)
            if self.token_count:
                return directions, [completion_tokens]
            return directions
        except Exception as e:
            # Decoding failed, retry
            # Three attempts to decode the JSON
            logger.warning(f"JSON decode failed: {e}")

        raise ValueError(f"Failed to generate valid JSON after multiple attempts.")

    def code_rectify(self, code: str, error: str) -> str:
        logger.info("Rectifying the code...")
        # Rectification of the code based on error message
        prompt = self.code_rectifier_prompt
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }, 
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Original Code: {code}\n\nError: {error}"
                    },
                ]
            }
        ]
        response = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature)
        rectified_code = extract_code_from_code_block(response)
        logger.info(f"Rectified code: {rectified_code}")
        return rectified_code
    def execute_code_save_plot(self, code: str, code_path: str):
        flag, cnt = True, 0
        while flag and cnt < 3:
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)

            try:
                glb = {"__name__": "__main__"}
                exec(code, glb, glb)
                logger.info(f"Executed code successfully.")
                logger.info(f"Executed code and saved plot to {code_path}")
                flag = False
            except Exception as e:
                print(f"Invalid code: {e}")
                cnt += 1
                code = self.code_rectify(code, str(e))     
        if flag:
            raise RuntimeError(f"Failed to execute code after multiple attempts")
        
    def generate_plot_code_exec(self, metadata_info: Dict, direction: Dict, data_path: str, output_dir: str, idx: int, code_path: str=None) -> str:
        # Save the code execution result plot to plot_{idx}.png
        prompt = self.code_generator_prompt.format(
            data_path = data_path,
            output_path = os.path.join(output_dir, f"plot_{idx}.png"),
        )
        # Guided by metadata information and direction
        metadata_info = json.dumps(metadata_info, indent=2)
        payload = f"""
            Metadata Information:
            {metadata_info}
            Direction:
            {json.dumps(direction, indent=2)}
        """

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the dataset metadata information and the **visualization direction** to follow:\n\n" + payload
                    }
                ]
            }
        ]
        try:
            # Generate the code and execute it
            if self.token_count:
                response, completion_tokens = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature, token_count=self.token_count)
            else:
                response = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature)
            code = extract_code_from_code_block(response)
            self.execute_code_save_plot(code, code_path)
            logger.info(f"Generated plot code for direction {idx}: {code}")
            if self.token_count:
                return code, [completion_tokens]
            return code
        except Exception as e:
            # If error occurs during execution or parsing, retry the whole process
            logger.error(f"Error generating plot code for direction: {str(e)}")
        
        raise RuntimeError(f"Failed to generate valid plot code after multiple attempts.")
    def check_image_quality(self, img_files: List[str]) -> List[str]:
        # Check the quality of generated images, return the list of verified images
        verified_images = []
        tokens = []
        for img_path in img_files:
            base64_image = encode_image(img_path)
            messages = [
                {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.check_image_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please check the quality of the following chart image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
            ]
            logger.info(f"Checking quality for image {img_path}")
            
            if self.token_count:
                response, completion_tokens = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature, token_count=self.token_count)
            else:
                response = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature)
            try:
                # Parse out the quality judgment
                quality = extract_json_from_code_block(response)
                quality = quality.replace("True", "true").replace("False", "false")
                data = json.loads(quality)
                if "is_legible" in data and isinstance(data["is_legible"], bool):
                    is_legible = data["is_legible"]
                    if self.token_count:
                        tokens.append(completion_tokens)
                    break
                else:
                    logger.warning(f"Response missing is_legible key or wrong type: {response}")
            except Exception as e:
                # Parse error, retry the whole process
                logger.warning(f"Failed to parse quality response: {response}, error: {e}")
            
            
            if is_legible:
                verified_images.append(img_path)
            
            print(f"Image {img_path} legibility: {is_legible}")
        if self.token_count:
            return verified_images, tokens
        return verified_images
    def visualize_data(self, metadata_report: Dict, data_path: str, folder_path: str) -> str:
        output_dir = os.path.join(folder_path, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        directions = self.direction_advisor(metadata_report)

        logger.info(f"Directions for visualization: {json.dumps(directions, indent=2)}")
        logger.info(f"Now generating plots in {output_dir}")
        img_files = []

        for i, direction in enumerate(directions):
            plot_code = self.generate_plot_code_exec(
                metadata_report, 
                direction,
                data_path, 
                output_dir,
                idx=i+1,
                code_path=os.path.join(output_dir, f"plot_{i+1}.py")
            )
            logger.info(f"Generated plot code for direction {i+1} and now executing it.")

            logger.info(f"Finished executing plot code for direction {i+1}.")

            img_files.append(os.path.join(output_dir, f"plot_{i+1}.png"))

        verified_img_files = self.check_image_quality(img_files)
        
        return verified_img_files

