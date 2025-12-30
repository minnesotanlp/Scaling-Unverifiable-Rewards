from dataclasses import dataclass

@dataclass
class TTSConfig:
    generation_model: str = "vllm"
    judge_model: str = "gpt-4o"
    temperature: float = 1.0
    top_p: float = 0.9
    max_tokens: int = 1200
    pruning_ratio: float = 0.0
    branching_factor: int = 5
    majority_judger_num: int = 3
    workdir: str = "data/session"
    token_count: bool = True
    
