import logging
import json
from .utils import call_llm, encode_image, extract_json_from_code_block
from .prompts import InsightPrompt
from typing import Dict, List

logger = logging.getLogger(__name__)


class InsightValidator:
    def __init__(self, 
                model: str = "gpt-4o",
                max_tokens: int = 1500,
                top_p: float = 0.9,
                temperature: float = 0.5,
                majority_judger_num: int = 3,
                token_count: bool = False):

        self.engine = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.token_count = token_count
        # Evaluator Choice
        self.verify_insight_prompt = InsightPrompt.EVALUATE_INSIGHT_HARSH
        # Number of judges for majority voting
        self.majority_judger_num = majority_judger_num
    
    def validate_insight(self, insight_dict: Dict) -> Dict:
        # Initialize final insight dict, key is image path, value is list of evaluated insights
        tokens = []
        final_insight_dict = {img_path: [] for img_path in insight_dict.keys()}

        for img_path, insights in insight_dict.items():
            logger.info(f"Validating insights for image: {img_path} with {len(insights)} insights.")
            # Evaluate each insight for the given image
            for idx, insight in enumerate(insights):
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": self.verify_insight_prompt
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Here is the Insight: {insight}, the corresponding chart is as below:"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(img_path)}",
                                    "detail": "high"
                                }

                            }

                        ]
                    }
                ]
                logger.info(f"Validating the {idx}_th insight and calling LLM...")
                if self.token_count:
                    response, completion_tokens = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature, num_responses=self.majority_judger_num, token_count=self.token_count)
                else:
                    response = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature, num_responses=self.majority_judger_num)
                try:
                    # Parse each judge's response and aggregate scores
                    if self.majority_judger_num == 1:
                        response = [response]
                    recorder = {}
                    all_scores = []
                    recorder['insight'] = insight
                    recorder['detailed_scores'] = []
                    recorder['evidences'] = []
                    for res in response:
                        json_content = extract_json_from_code_block(res)
                        data = json.loads(json_content)
                        scores = data["scores"]
                        average = sum([score for score in scores.values()]) / len(scores.values())
                        scores["total_score"] = average
                        recorder['detailed_scores'].append(scores)
                        recorder['evidences'].append(data["evidence"])
                        all_scores.append(average)
                    recorder['scores'] = all_scores
                    recorder['avg_scores'] = sum(all_scores) / len(all_scores) if all_scores else 0
                    final_insight_dict[img_path].append(recorder)
                    if self.token_count:
                        tokens.append(completion_tokens)
                    break
                except Exception as e:
                    logger.error(f"Error decoding JSON: {str(e)}")
        if self.token_count:
            return final_insight_dict, tokens
        return final_insight_dict

class InsightGenerator:
    def __init__(self, 
                generator_model: str = "gpt-4o",
                branching_factor: int = 5,
                max_tokens: int = 1500,
                top_p: float = 0.9,
                temperature: float = 0.5,
                token_count: bool = False):

        self.generator_engine = generator_model
        # Generate branching factor insights per image
        self.branching_factor = branching_factor
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.token_count = token_count
        if branching_factor == 1:
            print("Using Single Version Insight Generation Prompt.")
            self.generate_insight_prompt = InsightPrompt.GEN_INSIGHT_SINGLE
        else:
            self.generate_insight_prompt = InsightPrompt.GEN_INSIGHT
    
    def generate_insight(self, verified_images: List[str]) -> Dict:
        insight_dict = {img_file: [] for img_file in verified_images}
        raw_insight_dict = {img_file: [] for img_file in verified_images}
        tokens = []
        for img_path in verified_images:
            base64_image = encode_image(img_path)
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.generate_insight_prompt.replace("{num_insights}", str(self.branching_factor))
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the chart as below. Please generate the insights accordingly:"
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
            logger.info(f"Generating insight for image: {img_path}")
            if self.token_count:
                response, completion_tokens = call_llm(self.generator_engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature, token_count=self.token_count)
            else:
                response = call_llm(self.generator_engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature)
            try:
                # Extract JSON content
                json_content = extract_json_from_code_block(response)
                logger.info(f"Generated insight response: {json_content}")
                data = json.loads(json_content)
                # insight_dict value is a list of insight descriptions
                insight_dict[img_path] = [desc['description'] for desc in data["insights"]]
                # raw_insight_dict value is a list of raw insight dictionaries(with description and evidence)
                raw_insight_dict[img_path] = data["insights"]
                if self.token_count:
                    tokens.append(completion_tokens)
                break
            except Exception as e:
                logger.error(f"Error parsing JSON: {str(e)}")
            if raw_insight_dict[img_path] == []:
                raise ValueError(f"No insights generated for image: {img_path}")
        if self.token_count:
            return insight_dict, raw_insight_dict, tokens
        return insight_dict, raw_insight_dict






        
