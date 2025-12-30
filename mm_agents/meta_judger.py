import logging
from .prompts import MetaJudgerPrompt
from .utils import call_llm, extract_json_from_code_block
from typing import Any, Dict, List
import ast, re
import json

logger = logging.getLogger(__name__)

class MetaJudger:
    def __init__(self, 
                 model="gpt-4.1-nano",
                 max_tokens=1500,
                 top_p=0.9,
                 temperature=1.0,
                 majority_judger_num: int = 1,
                 token_count: bool = False
                 ):
        self.engine = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.judge_metadata_prompt = MetaJudgerPrompt.METADATA_JUDGE
        self.judge_direction_prompt = MetaJudgerPrompt.DIRECTION_JUDGE
        self.judge_insight_prompt = MetaJudgerPrompt.INSIGHT_JUDGE
        self.token_count = token_count
        self.majority_judger_num = majority_judger_num
        
    def majority_vote(self, rankings: List[List[int]]) -> List[int]:
        if not rankings:
            return []
        
        num_items = len(rankings[0])
        score_board = [0] * num_items
        
        for ranking in rankings:
            for rank, item_index in enumerate(ranking):
                score_board[item_index - 1] += num_items - rank  # Higher rank gets more points
        
        final_ranking = sorted(range(1, num_items + 1), key=lambda x: score_board[x - 1], reverse=True)
        return final_ranking

    def judge_metadata(self, metadata: List[Dict[str, Any]]) -> List[int]:
        payload = "\n\n".join([f"{i}: {meta.get('introduction', str(meta))}" for i, meta in enumerate(metadata, 1)])
        completion_token = 0
        messages = [
            {
                "role": "system",
                "content" : [
                    {
                        "type": "text",
                        "text": self.judge_metadata_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content" : [
                    {
                        "type": "text",
                        "text": f"Here are **{len(metadata)}** data introductions to be ranked, please rank them from best to worst with all the indices from 1 to {len(metadata)}:\n\n" + payload
                    }
                ]
            }
        ]
        try:
            all_rankings = []
            logger.info("Judging metadata quality")
            
            if self.token_count:
                judgment, completion_token = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature, num_responses=self.majority_judger_num, token_count=self.token_count)
            else:
                judgment = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature, num_responses=self.majority_judger_num)
            # print("Judgment received:", judgment)
            if self.majority_judger_num == 1: 
                judgment = [judgment]
            for res in judgment:
                json_str = extract_json_from_code_block(res)
                json_content = json.loads(json_str)
                ranking = json_content.get("ranking", [])
                print("Extracted ranking:", ranking)
                all_rankings.append(ranking)

            logger.info(f"Metadata judgment: {judgment}")
            ranking = self.majority_vote(all_rankings)
            if ranking:
                if self.token_count:
                    return ranking, [completion_token]
                return ranking
        except Exception as e:
            logger.error(f"Error while ranking metadata: {e}")
            
        raise ValueError(f"Invalid ranking in response after multiple attempts while ranking metadata: {judgment}")

    def judge_direction(self, directions: List[Dict[str, Any]]) -> List[int]:
        payload = "\n\n".join([f"{i}: {json.dumps(direction)}" for i, direction in enumerate(directions, 1)])
        messages = [
            {
                "role": "system",
                "content" : [
                    {
                        "type": "text",
                        "text": self.judge_direction_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content" : [
                    {
                        "type": "text",
                        "text": f"Here are **{len(directions)}** visualization directions to be ranked, please rank them from best to worst with all the indices from 1 to {len(directions)}:\n\n" + payload
                    }
                ]
            }
        ]
        try:
            all_rankings = []
            logger.info("Judging directions quality")
            if self.token_count:
                judgment, completion_token = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature, num_responses=self.majority_judger_num, token_count=self.token_count)
            else:
                judgment = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature, num_responses=self.majority_judger_num)
            
            if self.majority_judger_num == 1: 
                judgment = [judgment]
            for res in judgment:
                json_str = extract_json_from_code_block(res)
                json_content = json.loads(json_str)
                ranking = json_content.get("ranking", [])
                print("Extracted ranking:", ranking)
                all_rankings.append(ranking)

            logger.info(f"Metadata judgment: {judgment}")
            ranking = self.majority_vote(all_rankings)
            if ranking:
                if self.token_count:
                    return ranking, [completion_token]
                return ranking
        except Exception as e:
            logger.error(f"Error while ranking metadata: {e}")
            

    def judge_insights(self, insights: List[str]) -> List[int]:
        payload = "\n\n".join([f"{i}: {insight}" for i, insight in enumerate(insights, 1)])
        messages = [
            {
                "role": "system",
                "content" : [
                    {
                        "type": "text",
                        "text": self.judge_insight_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content" : [
                    {
                        "type": "text",
                        "text": f"Here are **{len(insights)}** insights to be ranked, please rank them from best to worst with all the indices from 1 to {len(insights)}:\n\n" + payload
                    }
                ]
            }
        ]
        try:
            all_rankings = []
            logger.info("Judging insights quality")
            if self.token_count:
                judgment, completion_token = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature, num_responses=self.majority_judger_num, token_count=self.token_count)
            else:
                judgment = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature, num_responses=self.majority_judger_num)
            
            if self.majority_judger_num == 1: 
                judgment = [judgment]
            for res in judgment:
                json_str = extract_json_from_code_block(res)
                json_content = json.loads(json_str)
                ranking = json_content.get("ranking", [])
                print("Extracted ranking:", ranking)
                all_rankings.append(ranking)

            logger.info(f"Metadata judgment: {judgment}")
            ranking = self.majority_vote(all_rankings)
            if ranking:
                if self.token_count:
                    return ranking, [completion_token]
                return ranking
        except Exception as e:
            logger.error(f"Error while ranking metadata: {e}")
            
        raise ValueError(f"Invalid ranking in response after multiple attempts while ranking insights: {judgment}")