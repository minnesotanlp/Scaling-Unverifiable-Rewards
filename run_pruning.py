from tts_type import TTSConfig
from tts_pruning_parallel import run_full
import datetime
import argparse
import os


def parse_args():
    """Parse command line arguments for tree-of-thought search pruning."""
    parser = argparse.ArgumentParser(
        description="Run Selective TTS for data analysis tasks"
    )

    # TTSConfig parameters
    parser.add_argument(
        "--generation_model",
        type=str,
        required=True,
        default="gpt-4.1-nano",
        help="Model to use for generating thoughts/solutions (e.g., gpt-4.1-nano, gpt-4o)"
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        required=True,
        default="vllm",
        help="Model to use for judging/evaluating thoughts (e.g., vllm, gpt-4o)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        default=1.0,
        help="Sampling temperature for generation (higher = more random, lower = more deterministic)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        required=True,
        default=0.9,
        help="Nucleus sampling parameter (top-p sampling threshold)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        required=True,
        default=1500,
        help="Maximum number of tokens to generate per response"
    )
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        required=True,
        default=0.0,
        help="Ratio of thoughts to prune at each step (0.0 = no pruning, 1.0 = prune all)"
    )
    parser.add_argument(
        "--branching_factor",
        type=int,
        required=True,
        default=5,
        help="Number of alternative thoughts to generate at each step"
    )
    parser.add_argument(
        "--majority_judger_num",
        type=int,
        required=True,
        default=3,
        help="Number of judges to use for majority voting when evaluating thoughts"
    )
    parser.add_argument(
        "--token_count",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        required=True,
        default=True,
        help="Whether to count tokens during execution (True/False)"
    )
    parser.add_argument(
        "--workdir",
        type=str,
        required=True,
        default="results/VIS",
        help="Working directory to save results and logs"
    )

    # run_full parameters
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        default="dataset/VIS.csv",
        help="Path to the input dataset CSV file"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        required=True,
        default=8,
        help="Number of parallel runs to execute"
    )
    parser.add_argument(
        "--executor",
        type=str,
        required=True,
        default="process",
        choices=["process", "thread"],
        help="Type of executor to use for parallel execution (process or thread)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        required=True,
        default=max(20, os.cpu_count()),
        help="Maximum number of worker processes/threads for parallel execution"
    )

    return parser.parse_args()


def main():
    """Main function to run the pruning experiment."""
    args = parse_args()

    # Create TTSConfig from arguments
    cfg = TTSConfig(
        generation_model=args.generation_model,
        judge_model=args.judge_model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        pruning_ratio=args.pruning_ratio,
        branching_factor=args.branching_factor,
        majority_judger_num=args.majority_judger_num,
        token_count=args.token_count,
        workdir=args.workdir,
    )

    # Run the experiment
    start = datetime.datetime.now()
    summary = run_full(
        cfg,
        data_path=args.data_path,
        n_runs=args.n_runs,
        executor=args.executor,
        max_workers=args.max_workers,
    )
    end = datetime.datetime.now()

    # Print timing information
    print(f"Start time: {start.strftime('%Y%m%d_%H%M%S')}, "
          f"End time: {end.strftime('%Y%m%d_%H%M%S')}, "
          f"Duration: {end - start}")


if __name__ == "__main__":
    main()