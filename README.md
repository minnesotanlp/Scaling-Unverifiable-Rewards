# Scaling Unverifiable Rewards: A Case Study on Visual Insights

[![arXiv](https://img.shields.io/badge/arXiv-2512.22650-b31b1b.svg)](https://arxiv.org/abs/2512.22650)
[![Website](https://img.shields.io/badge/🌍-Project_Website-blue)](https://minnesotanlp.github.io/insight-scaling-webpage)

**Authors:** Shuyu Gan, James Mooney, Pan Hao, Renxiang Wang, Mingyi Hong, Qianwen Wang, Dongyeop Kang

**Affiliation:** University of Minnesota

---

## Abstract

Large Language Model (LLM) agents can increasingly automate complex reasoning through Test-Time Scaling (TTS), iterative refinement guided by reward signals. However, many real-world tasks involve multi-stage pipeline whose final outcomes lack verifiable rewards or sufficient data to train robust reward models, making judge-based refinement prone to accumulate error over stages.

We propose **Selective TTS**, a *process-based refinement* framework that scales inference across different stages in multi-agent pipeline, instead of repeated refinement over time by prior work. By distributing compute across stages and pruning low-quality branches early using process-specific judges, Selective TTS mitigates the judge drift and stabilizes refinement.

Grounded in the data science pipeline, we build an end-to-end multi-agent pipeline for generating visually insightful charts and report of given dataset, and design a reliable LLM-based judge model, aligned with human experts (Kendall's τ=0.55). Our proposed selective TTS then improves insight quality under a fixed compute budget, increasing mean scores from **61.64 to 65.86** while reducing variance.

We hope our findings serve as the first step toward to scaling complex, open-ended tasks with unverifiable rewards, such as scientific discovery and story generation.

---

## Installation

### 1. Clone the Repository

```bash
# Clone this repository
git clone https://github.com/minnesotanlp/Scaling-Unverifiable-Rewards.git

# Change directory into the cloned repository
cd Scaling-Unverifiable-Rewards
```

### 2. Create Conda Environment

We recommend using the latest version of Conda to manage the environment. Please ensure that the version of Python is >= 3.10.

```bash
# Create a Conda environment
conda create -n scaling-unverifiable-rewards python=3.10

# Activate the environment
conda activate scaling-unverifiable-rewards
```

### 3. Install Dependencies

```bash
# Install required dependencies
pip install -r requirements.txt
```

---

## Configuration

### Environment Variables

Create a `.env` file in the root directory to configure your API keys and model settings:

```bash
touch .env
```

#### Using OpenAI GPT Models

If you want to use GPT models (e.g., `gpt-4.1-nano`, `gpt-4o`) as the backbone, add your OpenAI API key to `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

#### Using vLLM for Local Model Deployment

If you want to use vLLM to deploy Hugging Face models locally, configure the following in `.env`:

```env
VLLM_URL="http://localhost:8000/v1"
VLLM_MODEL="Qwen/Qwen2.5-VL-32B-Instruct"
```

**Note:** You need to start the vLLM server before running experiments. See the [Experiment](#experiment) section for details.

---

## Experiment

### Running the Selective TTS Pipeline

#### 1. Configure Parameters

The main entry point is `run_pruning.py`. You can configure the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--generation_model` | str | `gpt-4.1-nano` | Model for generating thoughts/solutions (e.g., `gpt-4.1-nano`, `gpt-4o`, `vllm`) |
| `--judge_model` | str | `vllm` | Model for judging/evaluating thoughts (e.g., `vllm`, `gpt-4o`) |
| `--temperature` | float | `1.0` | Sampling temperature (higher = more random) |
| `--top_p` | float | `0.9` | Nucleus sampling threshold |
| `--max_tokens` | int | `1500` | Maximum tokens per generation |
| `--pruning_ratio` | float | `0.0` | Ratio of thoughts to prune (0.0 = no pruning, 1.0 = prune all) |
| `--branching_factor` | int | `5` | Number of alternative thoughts at each step |
| `--majority_judger_num` | int | `3` | Number of judges for majority voting |
| `--token_count` | bool | `True` | Whether to count tokens during execution |
| `--workdir` | str | `results` | Directory to save results |
| `--data_path` | str | `dataset/VIS.csv` | Path to input dataset |
| `--n_runs` | int | `8` | Number of parallel runs |
| `--executor` | str | `process` | Executor type (`process` or `thread`) |
| `--max_workers` | int | `20` | Maximum worker processes/threads |

#### 2. Start vLLM Server (if using vLLM)

If you set `--judge_model vllm` or `--generation_model vllm`, you need to deploy the model first using vLLM:

```bash
# Example: Deploy Qwen2.5-VL-32B-Instruct
vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-length 8192
```

Make sure the `VLLM_URL` and `VLLM_MODEL` in your `.env` file match the deployed server.

#### 3. Run the Experiment

We recommend running experiments in a `tmux` session for long-running tasks:

```bash
# Start a tmux session
tmux new -s scaling-experiment

# Run the experiment
python run_pruning.py \
    --generation_model gpt-4.1-nano \
    --judge_model vllm \
    --temperature 1.0 \
    --top_p 0.9 \
    --max_tokens 1500 \
    --pruning_ratio 0.5 \
    --branching_factor 5 \
    --majority_judger_num 3 \
    --token_count True \
    --workdir results \
    --data_path dataset/VIS.csv \
    --n_runs 8 \
    --executor process \
    --max_workers 20
```

---

## Output Structure

After running an experiment, results are saved in the working directory with the following structure:

```
results/VIS/
└── runs/
    ├── run_00_meta_0/
    │   ├── data/
    │   │   ├── metadata_report.json    # Data profiling results
    │   │   └── normalized_path.txt     # Path to processed dataset (internal use)
    │   └── viz/
    │       ├── code/
    │       │   ├── plot_1.py           # Visualization code for direction 1
    │       │   ├── plot_2.py
    │       │   └── ...
    │       ├── images/
    │       │   ├── plot_1.png          # Generated visualization for direction 1
    │       │   ├── plot_2.png
    │       │   └── ...
    │       ├── verified/
    │       │   ├── plot_1.png          # Charts that passed quality check
    │       │   ├── plot_2.png
    │       │   └── ...
    │       ├── verified_list.json      # List of verified visualizations
    │       ├── budget_used.txt         # Compute budget consumed
    │       ├── debug_budget.log        # Detailed budget tracking log
    │       ├── directions_raw.json     # Unpruned visualization directions
    │       ├── directions.json         # Pruned visualization directions
    │       ├── insights_raw.json       # Raw insights with reasoning process
    │       ├── insights_validated.json # Pruned insights with detailed scores
    │       ├── result_summary.json     # Summary (budget, charts, mean scores)
    │       └── scores.json             # Mean insight score for this run
    ├── run_00_meta_1/
    │   └── ...
    ├── run_01_meta_0/
    │   └── ...
    └── ...
```

### File Descriptions

| File | Description |
|------|-------------|
| `metadata_report.json` | Data profiling results. `meta_j` indicates this is the j-th metadata report (j-th child node from the root) in the i-th independent run |
| `normalized_path.txt` | Internal path to processed dataset |
| `plot_{i}.py` | Visualization code for the i-th direction |
| `plot_{i}.png` | Generated visualization for the i-th direction |
| `verified/` | Directory containing charts that passed quality verification |
| `directions_raw.json` | All generated visualization directions before pruning |
| `directions.json` | Visualization directions after pruning |
| `insights_raw.json` | Raw insights for each verified chart, including reasoning process |
| `insights_validated.json` | Pruned insights with detailed scoring for post-processing analysis |
| `result_summary.json` | Summary statistics: budget used, chart count, mean insight scores |
| `scores.json` | Mean insight score across all insights in `run_i_meta_j` |
| `budget_used.txt` / `debug_budget.log` | Compute budget tracking for each scaling process |

---

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{gan2025scalingunverifiablerewardscase,
      title={Scaling Unverifiable Rewards: A Case Study on Visual Insights},
      author={Shuyu Gan and James Mooney and Pan Hao and Renxiang Wang and Mingyi Hong and Qianwen Wang and Dongyeop Kang},
      year={2025},
      eprint={2512.22650},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.22650},
}
```

---

## Acknowledgments

This work was supported by the [Minnesota NLP](https://minnesotanlp.github.io/) group of University of Minnesota. We thank all contributors and collaborators who made this research possible.
