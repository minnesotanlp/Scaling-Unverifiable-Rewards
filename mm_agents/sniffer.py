# agents/collection/collector.py
import pandas as pd
import json
import os
import logging
from typing import Optional
from .prompts import SnifferPrompt
from .utils import call_llm
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

class DataSniffer:
    def __init__(self, 
                 model="gpt-4o",
                 num_metadata_report: int = 5,
                 max_tokens=1500,
                 top_p=0.9,
                 temperature=1.0,
                 token_count: bool = False
                 ):
        self.sample_size = 2
        self.engine = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        # Generate multiple metadata reports
        self.num_metadata_report = num_metadata_report
        self.token_count = token_count
        self.gen_explanation_prompt = SnifferPrompt.GEN_METADATA_SYS_PROMPT

    def load_data(self, data_path: str) -> pd.DataFrame:
        suffix = data_path.split('.')[-1].lower().strip()
        if suffix == 'csv':
            return pd.read_csv(data_path)
        elif suffix in ['json', 'jsonl']:
            with open(data_path) as f:
                data = json.load(f)
            return pd.json_normalize(data)
        elif suffix in ['xlsx', 'xls']:
            return pd.read_excel(data_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
    def generate_metadatareport(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Randomly sample the rows
        completion_token = 0
        sample_df = df.sample(n=self.sample_size).fillna("NULL") 
        sample_table = sample_df.to_csv(index=False)
        # Column info
        dtype_info = ", ".join([f"- `{col}`: {str(dtype)}" 
                          for col, dtype in df.dtypes.items()])
        # Aggregate to dict
        metadata = {
            "shape": f"{len(df)} rows * {len(df.columns)} columns",
            "dtypes": dtype_info,
        }
        messages = []
        messages.append({
            "role": "system",
            "content" : [
                {
                    "type": "text",
                    "text": self.gen_explanation_prompt
                }
            ]
        })
        # Metadata and sample data
        sniff_info = f"""
            ### Dataset Metadata Information
            {json.dumps(metadata, indent=2)}

            ### Sample Data (Random Sampled {self.sample_size} Rows)
            {sample_table}
        """
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Here is the sniffed information, including dataset metadata and sample data:\n\n{sniff_info}"
                }
            ]
        })
        metadata_lst = []
        if self.token_count:
            introduction, completion_token = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature, num_responses=self.num_metadata_report, token_count=self.token_count)
            
        else:
            introduction = call_llm(self.engine, messages, max_tokens=self.max_tokens, top_p=self.top_p, temperature=self.temperature, num_responses=self.num_metadata_report)
        if self.num_metadata_report > 1:
            for intro in introduction:
                logger.info(f"Generated dataset introduction: {intro}")
                # Aggregate to final metadata report
                metadata = {
                    "shape": f"{len(df)} rows * {len(df.columns)} columns",
                    "dtypes": dtype_info,
                    "introduction": intro,
                    "sample_data": sample_table
                }
                metadata_lst.append(metadata)
            if self.token_count:
                return metadata_lst, completion_token
            return metadata_lst
        else:
            metadata = {
                "shape": f"{len(df)} rows * {len(df.columns)} columns",
                "dtypes": dtype_info,
                "introduction": introduction,
                "sample_data": sample_table
            }
            if self.token_count:
                return [metadata], completion_token
            return [metadata]


    def normalize_and_save(self, data_path: str, folder_path: str) -> Tuple[pd.DataFrame, str, Dict]:  
        try:
            df = self.load_data(data_path)
            logger.info(f"Loaded data from {data_path} with shape {df.shape}")
        except Exception as e:
            logging.error(f"Error processing files in {data_path}: {e}")
        completion_token = 0
        if self.token_count:
            metadata_report, completion_token = self.generate_metadatareport(df)
        else:
            metadata_report = self.generate_metadatareport(df)
        logger.info(f"Generated metadata report: {json.dumps(metadata_report, indent=2)}")
        # Normalized data path
        new_data_path = os.path.join(folder_path, "data.csv")
        df.to_csv(new_data_path, index=False, encoding="utf-8")
        
        if self.token_count:
            return df, new_data_path, metadata_report, [completion_token]
        
        return df, new_data_path, metadata_report
    
