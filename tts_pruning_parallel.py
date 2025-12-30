import os
import json
import shutil
import traceback
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from mm_agents.sniffer import DataSniffer
from mm_agents.visualizer import DataVisualizer
from mm_agents.insight import InsightGenerator, InsightValidator
from mm_agents.meta_judger import MetaJudger

from tts_type import TTSConfig
import random

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _dump_json(obj: Any, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_text(text: str, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _copy_into(paths: List[str], dest_dir: str) -> List[str]:
    os.makedirs(dest_dir, exist_ok=True)
    out = []
    for p in paths:
        if os.path.isfile(p):
            dst = os.path.join(dest_dir, os.path.basename(p))
            shutil.copy2(p, dst)
            out.append(dst)
    return out

def prune_metadata_candidates(judger: MetaJudger, candidates: List[Dict[str, Any]], pruning_ratio: float = 0.0) -> List[Dict[str, Any]]:
    completion_token = []
    if pruning_ratio > 0.0:
        try:
            # Pruning the metadata reports
            ranking, completion_token = judger.judge_metadata(
                [candidate["metadata_report"] for candidate in candidates]
            )
            num_to_keep = max(1, int(len(ranking) * (1 - pruning_ratio)))
            keep_idx = set(ranking[:num_to_keep])
            candidates = [c for i, c in enumerate(candidates, 1) if i in keep_idx]
        except Exception as e:
            candidates = random.sample(candidates, max(1, int(len(candidates) * (1 - pruning_ratio))))
    return candidates, completion_token


def prune_directions(judger: MetaJudger, directions: List[Dict[str, Any]], pruning_ratio: float = 0.0) -> List[Dict[str, Any]]:
    completion_token = []
    if pruning_ratio > 0.0:
        try:
            # Pruning the visualization directions
            ranking, completion_token = judger.judge_direction(
                directions
            )
            num_to_keep = max(1, int(len(ranking) * (1 - pruning_ratio)))
            keep_idx = set(ranking[:num_to_keep])
            directions = [d for i, d in enumerate(directions, 1) if i in keep_idx]
        except Exception as e:
            directions = random.sample(directions, max(1, int(len(directions) * (1 - pruning_ratio))))
    return directions, completion_token


def prune_insights(judger: MetaJudger, insights: Dict[str, List[str]], pruning_ratio: float = 0.0) -> Dict[str, List[str]]:
    tokens = []
    if pruning_ratio > 0.0:
        for img_path, ins_list in insights.items():
            try:
                # Pruning the insights for each image
                ranking, completion_token = judger.judge_insights(
                    ins_list
                )
                num_to_keep = max(1, int(len(ranking) * (1 - pruning_ratio)))
                keep_idx = set(ranking[:num_to_keep])
                insights[img_path] = [ins for i, ins in enumerate(ins_list, 1) if i in keep_idx]
            except Exception as e:
                insights[img_path] = random.sample(ins_list, max(1, int(len(ins_list) * (1 - pruning_ratio))))
            tokens.extend(completion_token)
    return insights, tokens

class TTSPipeline:
    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
            
    def run_experiment(
        self,
        data_path: str,
        n_runs: int = 10,
        session_name: Optional[str] = None,
        executor: Literal["thread", "process"] = "thread",
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Make session folder
        session_dir = self._prepare_session_root(data_path, session_name, n_runs)
        # Record the results, including budgets used for generation and pruning
        manifest = {"session_dir": session_dir, "runs": [], "gen_budget": 0, "prune_budget": 0, "overall_budget": 0, "meta_report_gen": [], "meta_report_prune": []}
        # Pool for jobs (jobs starts with metadata candidates)
        all_metadata_jobs = []
       
        # Parallel execution
        Exec = ThreadPoolExecutor if executor == "thread" else ProcessPoolExecutor
        workers = max_workers or os.cpu_count() or 4
        
        

        # Budget tracking
        total_budget_gen = 0
        total_budget_prune = 0
        total_budget_so_far = 0
        tokens_run = {}
        # Step 1: Generate metadata candidates for all runs
        with Exec(max_workers=workers) as ex:
            futures = []
            for run_idx in range(1, n_runs + 1):
                run_id = f"run_{run_idx:02d}"
                futures.append(ex.submit(
                    _generate_metadata_candidates_worker,
                    run_id, session_dir, data_path, self.cfg
                ))
            for fut in as_completed(futures):
                # Record the budget
                jobs, budget_used_gen, budget_used_prune, tokens_used_gen, tokens_used_prune, run_id = fut.result()
                # Record the jobs for Step 2
                all_metadata_jobs.extend(jobs)
                total_budget_gen += budget_used_gen
                total_budget_prune += budget_used_prune
                total_budget_so_far += budget_used_gen + budget_used_prune
                tokens_metadata = {
                    "generation_tokens": tokens_used_gen,
                    "pruning_tokens": tokens_used_prune,
                    "all_tokens": tokens_used_gen + tokens_used_prune,
                    "budget_gen_meta": budget_used_gen,
                    "budget_prune_meta": budget_used_prune,
                    "total_budget_meta": budget_used_gen + budget_used_prune
                }
                tokens_run[run_id] = tokens_metadata
        _dump_json(tokens_run, os.path.join(session_dir, "metadata_tokens.json"))
        print(f"[TTS] Generated {len(all_metadata_jobs)} metadata candidates across {n_runs} runs, total budget so far: {total_budget_so_far}")
        # Recorder for the Step 2
        results = []
        # Step 2: Execute each metadata candidate through the full pipeline with branching
        with Exec(max_workers=workers) as ex:
            futures = {ex.submit(_execute_metadata_worker, job, self.cfg): job for job in all_metadata_jobs}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                if not res.get("error", False):
                    # If successful, accumulate budget
                    total_budget_so_far += res.get("budget_used", 0)
                    total_budget_gen += res.get("budget_gen", 0)
                    total_budget_prune += res.get("budget_prune", 0)
        # Finalize manifest
        manifest["runs"] = results
        # Record the budget for generation and pruning respectively
        manifest["gen_budget"] = total_budget_gen
        manifest["prune_budget"] = total_budget_prune
        manifest["overall_budget"] = total_budget_so_far

        _dump_json(manifest, os.path.join(session_dir, "manifest.json"))
        
        
        print(f"[TTS] Done. Total metadata jobs: {len(results)}, total generation budget used: {total_budget_gen}, total pruning budget used: {total_budget_prune}, overall total budget used: {total_budget_so_far}")
        return manifest
    
    def _prepare_session_root(self, data_path: str, session_name: Optional[str], n_runs: int) -> str:
        # Prepare the base dir
        base = self.cfg.workdir
        filename = os.path.basename(data_path)
        tag = session_name or f"{filename}"
        session_dir = _ensure_dir(os.path.join(base, tag))
        _dump_json(
            {
                "generation_model": self.cfg.generation_model,
                "judge_model": self.cfg.judge_model,
                "branching_factor": self.cfg.branching_factor,
                "workdir": self.cfg.workdir,
                "max_tokens": self.cfg.max_tokens,
                "top_p": self.cfg.top_p,
                "temperature": self.cfg.temperature,
                "data_path": os.path.abspath(data_path),
                "n_runs": n_runs,
                "pruning_ratio": self.cfg.pruning_ratio,
                "majority_judger_num": self.cfg.majority_judger_num,
                "token_count": self.cfg.token_count
            },
            os.path.join(session_dir, "config.json"),
        )
        return session_dir

def _generate_metadata_candidates_worker(run_id: str, session_dir: str, data_path: str, cfg: TTSConfig) -> Tuple[List[Dict[str, Any]], int, int]:
    """Generate multiple metadata candidates for a run, return jobs with run_i_meta_j dirs."""
    try:
        # Initialize budget counters
        generation_budget_used = 0
        pruning_budget_used = 0
        generation_tokens = []
        pruning_tokens = []
        # Initialize sniffer and stage-wise judger
        sniffer = DataSniffer(
            model=cfg.generation_model, num_metadata_report=cfg.branching_factor,
            max_tokens=cfg.max_tokens, top_p=cfg.top_p, temperature=cfg.temperature, token_count=cfg.token_count
        )
        judger = MetaJudger(
            model=cfg.judge_model, max_tokens=cfg.max_tokens, top_p=cfg.top_p, temperature=cfg.temperature,
            majority_judger_num=cfg.majority_judger_num, token_count=cfg.token_count
        )
        # Generate metadata candidates
        df, normalized_path, metadata_list, completion_token = sniffer.normalize_and_save(data_path, session_dir)
        
        # one sniffer call generating multiple metadata
        generation_budget_used += 1
        generation_tokens.extend(completion_token)
        
        candidates = [{"meta_id": j + 1, "metadata_report": m, "normalized_path": normalized_path}
                      for j, m in enumerate(metadata_list)]
        # Prune metadata candidates
        kept, completion_token = prune_metadata_candidates(judger, candidates, cfg.pruning_ratio)  # one sniffer call
        if cfg.pruning_ratio > 0.0:
            pruning_budget_used += 1
            pruning_tokens.extend(completion_token)

        jobs = []
        # Initialize jobs for each kept candidate
        for c in kept:
            meta_id = c["meta_id"]
            # run_i_meta_j(first branching phase)
            meta_dir = _ensure_dir(os.path.join(session_dir, "runs", f"{run_id}_meta_{meta_id}"))
            jobs.append({
                "run_id": run_id,
                "meta_id": meta_id,
                "meta_dir": meta_dir,
                "data_path": data_path,
                "metadata_report": c["metadata_report"],
                "normalized_path": c["normalized_path"],
            })
        return jobs, generation_budget_used, pruning_budget_used, generation_tokens, pruning_tokens, run_id
    except Exception as e:
        tb = traceback.format_exc()
        os.makedirs(os.path.join(session_dir, "runs", run_id), exist_ok=True)
        err_json = os.path.join(session_dir, "runs", run_id, "error.json")
        _dump_json({"error": str(e), "traceback": tb}, err_json)
        return [], 0, 0, [], [], run_id

def _count_insights(ins):
    if not ins:
        return 0
    if isinstance(ins, dict):
        # values are lists of insights for different images
        return sum(len(v) if isinstance(v, list) else 0 for v in ins.values())
    if isinstance(ins, list):
        return len(ins)
    return 0

def _execute_metadata_worker(job: Dict[str, Any], cfg: TTSConfig) -> Dict[str, Any]:
    """Process one metadata candidate through the pipeline."""
    # Parameters for this job, meta_id represents the metadata report candidate index
    run_id = job["run_id"]
    meta_id = job["meta_id"]
    meta_dir = job["meta_dir"]
    
    # Budget trackers
    budget = 0
    budget_gen = 0
    budget_prune = 0
    tokens = []
    tokens_gen = []
    tokens_prune = []
    debug_budget_log = f"[{run_id}_{meta_id}] Start budget from 0 (metadata accounted separately)\n"
    with open(os.path.join(meta_dir, "debug_budget.log"), "a") as f:
        f.write(debug_budget_log)  
    try:
        # Save metadata
        ddir = _ensure_dir(os.path.join(meta_dir, "data"))
        _dump_json(job["metadata_report"], os.path.join(ddir, "metadata_report.json"))
        _write_text(job["normalized_path"], os.path.join(ddir, "normalized_path.txt"))
        
        # Stage-wise judger
        judger = MetaJudger(
            model=cfg.judge_model, max_tokens=cfg.max_tokens, top_p=cfg.top_p, temperature=cfg.temperature, majority_judger_num=cfg.majority_judger_num, token_count=cfg.token_count
        )
        
        # Generate visualization directions
        vis = DataVisualizer(model=cfg.generation_model, max_tokens=cfg.max_tokens,
                             top_p=cfg.top_p, num_directions=cfg.branching_factor,
                             temperature=cfg.temperature, token_count=cfg.token_count)
        
        directions, completion_tokens = vis.direction_advisor(job["metadata_report"])
        # Record unpruned directions
        _dump_json(directions, os.path.join(meta_dir, "directions_raw.json"))
        
        # assume 1 LLM call for direction advisor
        budget += 1  
        budget_gen += 1
        tokens.extend(completion_tokens)
        tokens_gen.extend(completion_tokens)
        debug_budget_log += f"[{run_id}_{meta_id}] Added 1 for direction advisor, total budget: {budget}\n"
        debug_budget_log += f"{len(completion_tokens)} Tokens used for direction generation: {completion_tokens}\n\n"
        with open(os.path.join(meta_dir, "debug_budget.log"), "a") as f:
            f.write(debug_budget_log)
            
        # Prune directions
        directions, completion_tokens = prune_directions(judger, directions, cfg.pruning_ratio)
        _dump_json(directions, os.path.join(meta_dir, "directions.json"))
        if cfg.pruning_ratio > 0.0:
            # assume 1 LLM call for direction pruning
            budget += 1
            budget_prune += 1
            tokens.extend(completion_tokens)
            tokens_prune.extend(completion_tokens)
            debug_budget_log += f"[{run_id}_{meta_id}] Added 1 for direction pruning, total budget: {budget}\n"
            debug_budget_log += f"{len(completion_tokens)} Tokens used for direction pruning: {completion_tokens}\n\n"
            with open(os.path.join(meta_dir, "debug_budget.log"), "a") as f:
                f.write(debug_budget_log)
                
        # Generate and verify plots
        # Extract metadata info without introduction for plot generation
        metadata_info = {k:v for k,v in job["metadata_report"].items() if k != "introduction"}
        # metadata_info = job["metadata_report"]
        verified_paths, completion_tokens = _worker_generate_and_verify_plots(vis, meta_dir, metadata_info, directions, job["normalized_path"])
        if not verified_paths:
            raise ValueError("No verified plots generated.")
        
        # for each direction, assume 1 LLM call for code generation and 1 for plot verification
        budget += 2 * len(directions)  
        budget_gen += 2 * len(directions)
        tokens.extend(completion_tokens)
        tokens_gen.extend(completion_tokens)
        debug_budget_log += f"[{run_id}_{meta_id}] Added {2 * len(directions)} for {len(directions)} directions (code + verification), total budget: {budget}\n"
        debug_budget_log += f"{len(completion_tokens)} Tokens used for plot generation and verification: {completion_tokens}\n\n"
        with open(os.path.join(meta_dir, "debug_budget.log"), "a") as f:
            f.write(debug_budget_log)

        # Insight generation and validation
        gen = InsightGenerator(generator_model=cfg.generation_model, branching_factor=cfg.branching_factor, max_tokens=cfg.max_tokens,
                               top_p=cfg.top_p, temperature=cfg.temperature, token_count=cfg.token_count)
        val = InsightValidator(model=cfg.judge_model, max_tokens=cfg.max_tokens,
                               top_p=cfg.top_p, temperature=cfg.temperature, majority_judger_num=cfg.majority_judger_num,
                               token_count=cfg.token_count)
        
        # assume insight generation for each verified chart is 1 LLM call
        insight_dict, insights_raw, completion_tokens = gen.generate_insight(verified_paths)
        budget += len(verified_paths) 
        budget_gen += len(verified_paths)
        tokens.extend(completion_tokens)
        tokens_gen.extend(completion_tokens)
        debug_budget_log += f"[{run_id}_{meta_id}] Added {len(verified_paths)} for insight generation, total budget: {budget}\n"
        debug_budget_log += f"{len(completion_tokens)} Tokens used for insight generation: {completion_tokens}\n\n"
        with open(os.path.join(meta_dir, "debug_budget.log"), "a") as f:
            f.write(debug_budget_log)
        # Record unpruned insights
        _dump_json(insights_raw, os.path.join(meta_dir, "insights_raw.json"))
        
        # Prune insights
        insights_dict_pruned, completion_tokens = prune_insights(judger, insight_dict, cfg.pruning_ratio)
        if cfg.pruning_ratio > 0.0:
            # For each verified chart, assume 1 LLM call for insight pruning
            budget += len(verified_paths)
            budget_prune += len(verified_paths)
            tokens.extend(completion_tokens)
            tokens_prune.extend(completion_tokens)
            debug_budget_log += f"[{run_id}_{meta_id}] Added {len(verified_paths)} for insight pruning, total budget: {budget}\n"
            debug_budget_log += f"{len(completion_tokens)} Tokens used for insight pruning: {completion_tokens}\n\n"
            with open(os.path.join(meta_dir, "debug_budget.log"), "a") as f:
                f.write(debug_budget_log)
                
        # Validate insights
        final_insights_map, completion_tokens = val.validate_insight(insights_dict_pruned)
        total_insights_to_validate = _count_insights(insights_dict_pruned) 
        
        # assume each insight validation is 1 LLM call
        budget += total_insights_to_validate 
        budget_gen += total_insights_to_validate
        tokens.extend(completion_tokens)
        tokens_gen.extend(completion_tokens)
        debug_budget_log += f"[{run_id}_{meta_id}] Added {total_insights_to_validate} for insight validation, total budget: {budget}\n"
        debug_budget_log += f"{len(completion_tokens)} Tokens used for insight validation: {completion_tokens}\n\n"
        with open(os.path.join(meta_dir, "debug_budget.log"), "a") as f:
            f.write(debug_budget_log)
            
        # Score summarization and final dumps
        overall_avg, per_image_avg = _worker_summarize_scores(final_insights_map)
        _dump_json(final_insights_map, os.path.join(meta_dir, "insights_validated.json"))
        _dump_json({"overall_avg": overall_avg, "per_image_avg": per_image_avg}, os.path.join(meta_dir, "scores.json"))
        
        with open(os.path.join(meta_dir, "budget_used.txt"), "w", encoding="utf-8") as f:
            f.write(f"[Debug Log for {run_id}_{meta_id}]:\n\n{debug_budget_log}\n\nFinal total budget used: {budget}")
        
        result = {
            "run_id": run_id,
            "meta_id": meta_id,
            "meta_dir": meta_dir,
            "overall_avg": overall_avg,
            "per_img_avg": per_image_avg,
            "num_verified": len(verified_paths),
            "budget_used": budget,
            "budget_gen": budget_gen,
            "budget_prune": budget_prune,
            "debug_budget_log": debug_budget_log,
            "tokens_used": tokens,
            "tokens_gen": tokens_gen,
            "tokens_prune": tokens_prune,
            "len_tokens": len(tokens),
            "len_tokens_gen": len(tokens_gen),
            "len_tokens_prune": len(tokens_prune),
            "len(tokens) == budget": len(tokens) == budget,
            "error": False
        }
        # Record the result(budget, scores, etc.) for each run_i_meta_j
        _dump_json(result, os.path.join(meta_dir, "result_summary.json"))
        return result
    except Exception as e:
        # If any error occurs, log it and return error status, "error": True
        tb = traceback.format_exc()
        err_json = os.path.join(meta_dir, "error.json")
        _dump_json({"error": str(e), "traceback": tb}, err_json)
        return {"run_id": run_id, "meta_id": meta_id, "meta_dir": meta_dir, "error": True}

def _worker_generate_and_verify_plots(
    vis: DataVisualizer,
    meta_dir: str,
    metadata_info: Dict[str, Any],
    directions: List[Dict[str, Any]],
    normalized_path: str,
) -> List[str]:
    # Create dirs for codes, images, verified images
    viz_dir = _ensure_dir(os.path.join(meta_dir, "viz"))
    code_dir = _ensure_dir(os.path.join(viz_dir, "code"))
    img_dir = _ensure_dir(os.path.join(viz_dir, "images"))
    verified_dir = _ensure_dir(os.path.join(viz_dir, "verified"))

    generated_imgs: List[str] = []
    tokens = []
    # For each direction, generate plot code, execute and save plot
    for i, spec in enumerate(directions, start=1):
        code_path = os.path.join(code_dir, f"plot_{i}.py")
        code, completion_tokens = vis.generate_plot_code_exec(metadata_info, spec, normalized_path, img_dir, idx=i, code_path=code_path)
        _write_text(code, code_path)
        img_path = os.path.join(img_dir, f"plot_{i}.png")
        generated_imgs.append(img_path)
        tokens.extend(completion_tokens)
    # Verify generated images
    verified, completion_tokens = vis.check_image_quality(generated_imgs)
    tokens.extend(completion_tokens)
    verified_copies = _copy_into(verified, verified_dir)
    _dump_json(verified_copies, os.path.join(viz_dir, "verified_list.json"))
    return verified_copies, tokens


def _worker_summarize_scores(final_insights_map: Dict[str, List[Dict[str, Any]]]) -> Tuple[float, Dict[str, float]]:
    # Summarize per-image and overall average scores(Optional)
    per_image_avg: Dict[str, float] = {}
    all_scores: List[float] = []
    for image, insights in final_insights_map.items():
        scores = [float(ins.get("avg_scores", 0.0)) for ins in insights if ins.get("avg_scores") is not None]
        avg = sum(scores) / max(len(scores), 1)
        per_image_avg[image] = avg
        all_scores.extend(scores)
    overall = sum(all_scores) / max(len(all_scores), 1)
    return overall, per_image_avg

def run_full(
    cfg: TTSConfig,
    data_path: str,
    n_runs: int = 10,
    session_name: Optional[str] = None,
    executor: Literal["thread", "process"] = "thread",
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    # Run the full TTS pipeline with the given configuration and parameters
    pipeline = TTSPipeline(cfg)
    return pipeline.run_experiment(
        data_path=data_path,
        n_runs=n_runs,
        session_name=session_name,
        executor=executor,
        max_workers=max_workers,
    )