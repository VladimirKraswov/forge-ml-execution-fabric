from __future__ import annotations

import csv
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from reporter import Reporter
from schemas import JobConfig

logger = logging.getLogger(__name__)


def resolve_torch_dtype(dtype_value: str):
    value = str(dtype_value or "auto").lower().strip()
    if value == "auto":
        return "auto"
    if value in ("float16", "half", "fp16"):
        return torch.float16
    if value in ("bfloat16", "bf16"):
        return torch.bfloat16
    if value in ("float32", "float", "fp32"):
        return torch.float32
    return "auto"


def render_prompt_template(template: str, sample: Dict[str, Any]) -> str:
    tags = sample.get("hash_tags") or []
    tags_text = ", ".join(tags) if isinstance(tags, list) and tags else "none"

    rendered = template
    replacements = {
        "${question}": str(sample.get("question", "")),
        "${candidateAnswer}": str(sample.get("candidate_answer", "")),
        "${referenceScore}": str(sample.get("reference_score", "")),
        "${maxScore}": str(sample.get("max_score", 5)),
        "${tagsText}": tags_text,
    }

    for key, value in replacements.items():
        rendered = rendered.replace(key, value)

    return rendered


def extract_generated_text(tokenizer, prompt_text: str, output_ids, input_len: int) -> str:
    generated_ids = output_ids[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def parse_model_score(
    text: str,
    parsing_regex: Optional[str] = None,
    score_min: float = 0.0,
    score_max: float = 5.0,
) -> Dict[str, Any]:
    if not text or not str(text).strip():
        return {"score": None, "feedback": None, "parseError": True}

    clean_text = str(text).strip()

    try:
        json_match = re.search(r"\{[\s\S]*\}", clean_text)
        if json_match:
            data = json.loads(json_match.group(0))
            raw_score = data.get("score")
            if isinstance(raw_score, str):
                raw_score = float(raw_score)
            if isinstance(raw_score, (int, float)) and math.isfinite(raw_score):
                if score_min <= raw_score <= score_max:
                    return {
                        "score": float(raw_score),
                        "feedback": data.get("feedback") or data.get("reasoning"),
                        "parseError": False,
                    }
    except Exception:
        pass

    patterns = []
    if parsing_regex:
        patterns.append(parsing_regex)

    # Defaults
    patterns.extend([
        fr"score:\s*(\d+(?:\.\d+)?)\s*/\s*{int(score_max)}",
        fr"оценка:\s*(\d+(?:\.\d+)?)\s*/\s*{int(score_max)}",
        r"score:\s*(\d+(?:\.\d+)?)",
        r"оценка:\s*(\d+(?:\.\d+)?)",
        fr"(\d+(?:\.\d+)?)\s*/\s*{int(score_max)}",
        r"^(\d+(?:\.\d+)?)$",
    ])

    for pattern in patterns:
        try:
            match = re.search(pattern, clean_text, re.IGNORECASE | re.MULTILINE)
            if not match:
                continue
            value = float(match.group(1))
            if score_min <= value <= score_max:
                return {
                    "score": value,
                    "feedback": clean_text,
                    "parseError": False,
                }
        except Exception:
            continue

    return {
        "score": None,
        "feedback": clean_text,
        "parseError": True,
    }


def calculate_metrics(model_label: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = [
        row
        for row in rows
        if not row.get("parseError") and isinstance(row.get("predictedScore"), (int, float))
    ]
    total_samples = len(rows)

    if not valid_rows:
        return {
            "model": model_label,
            "samples": total_samples,
            "parseSuccessRate": 0.0,
            "mae": None,
            "rmse": None,
            "exactRate": 0.0,
            "within1Rate": 0.0,
            "within2Rate": 0.0,
            "meanSignedError": None,
            "avgPredictedScore": None,
            "parseErrors": sum(1 for row in rows if row.get("parseError")),
            "inferenceErrors": sum(1 for row in rows if row.get("inferenceError")),
            "emptyResponses": sum(1 for row in rows if not str(row.get("rawResponse") or "").strip()),
        }

    n = len(valid_rows)
    abs_errors = []
    sq_errors = []
    signed_errors = []
    exact = 0
    within1 = 0
    within2 = 0
    predicted_sum = 0.0

    for row in valid_rows:
        error = float(row["predictedScore"]) - float(row["referenceScore"])
        abs_error = abs(error)
        abs_errors.append(abs_error)
        sq_errors.append(error * error)
        signed_errors.append(error)
        predicted_sum += float(row["predictedScore"])

        if abs_error == 0:
            exact += 1
        if abs_error <= 1:
            within1 += 1
        if abs_error <= 2:
            within2 += 1

    return {
        "model": model_label,
        "samples": total_samples,
        "parseSuccessRate": n / total_samples if total_samples else 0.0,
        "mae": sum(abs_errors) / n,
        "rmse": math.sqrt(sum(sq_errors) / n),
        "exactRate": exact / n,
        "within1Rate": within1 / n,
        "within2Rate": within2 / n,
        "meanSignedError": sum(signed_errors) / n,
        "avgPredictedScore": predicted_sum / n,
        "parseErrors": sum(1 for row in rows if row.get("parseError")),
        "inferenceErrors": sum(1 for row in rows if row.get("inferenceError")),
        "emptyResponses": sum(1 for row in rows if not str(row.get("rawResponse") or "").strip()),
    }


def _load_eval_items(path: str, fmt: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    if fmt == "json":
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
            return payload["samples"]
        if isinstance(payload, list):
            return payload
        raise ValueError("evaluation dataset json must be a list or an object with 'samples'")

    items: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            items.append(json.loads(raw))
    return items


def _normalize_eval_items(cfg: JobConfig) -> List[Dict[str, Any]]:
    eval_cfg = cfg.evaluation
    ds_cfg = eval_cfg.dataset
    assert ds_cfg is not None

    if not ds_cfg.path:
        raise ValueError("evaluation.dataset.path is required before evaluation starts")

    raw_items = _load_eval_items(ds_cfg.path, ds_cfg.format)
    normalized: List[Dict[str, Any]] = []

    for idx, item in enumerate(raw_items):
        question = item.get(ds_cfg.question_field)
        answer = item.get(ds_cfg.answer_field)
        score = item.get(ds_cfg.score_field)
        max_score = item.get(ds_cfg.max_score_field) if ds_cfg.max_score_field else 5
        tags = item.get(ds_cfg.tags_field) if ds_cfg.tags_field else []

        if question is None or answer is None or score is None:
            continue

        try:
            reference_score = float(score)
        except Exception:
            continue

        if not isinstance(tags, list):
            tags = []

        normalized.append(
            {
                "id": item.get("id") or f"sample_{idx + 1}",
                "question": str(question),
                "candidate_answer": str(answer),
                "reference_score": reference_score,
                "max_score": max_score if isinstance(max_score, (int, float)) else 5,
                "hash_tags": [str(tag) for tag in tags],
            }
        )

    if cfg.evaluation.max_samples:
        normalized = normalized[: cfg.evaluation.max_samples]

    return normalized


def _build_prompt(tokenizer, template: str, sample: Dict[str, Any], system_prompt: Optional[str] = None) -> str:
    rendered = render_prompt_template(template, sample)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": rendered})

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return rendered


def _load_eval_model(cfg: JobConfig, training_result: Dict[str, Any]) -> tuple[Any, Any, str]:
    target = cfg.evaluation.target
    if target == "auto":
        target = "merged" if training_result.get("merged_dir") else "lora"

    torch_dtype = resolve_torch_dtype(cfg.model.dtype)

    if target == "merged":
        merged_dir = training_result.get("merged_dir")
        if not merged_dir or not Path(merged_dir).exists():
            raise ValueError("Merged model directory is missing, but evaluation.target='merged'")
        model_label = str(merged_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            merged_dir,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        return model, tokenizer, model_label

    base_model = cfg.model.local_path if cfg.model.source == "local" else cfg.model.repo_id
    if not base_model:
        raise ValueError("Cannot resolve base model for LoRA evaluation")

    lora_dir = training_result.get("lora_dir")
    if not lora_dir or not Path(lora_dir).exists():
        raise ValueError("LoRA adapter directory is missing, but evaluation.target='lora'")

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch_dtype,
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    if cfg.model.load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model = PeftModel.from_pretrained(model, lora_dir)

    return model, tokenizer, f"{base_model}+lora"


def run_evaluation(
    cfg: JobConfig,
    training_result: Dict[str, Any],
    reporter: Optional[Reporter] = None,
) -> Dict[str, Any]:
    if not cfg.evaluation.enabled:
        return {"enabled": False}

    if reporter:
        reporter.report_status(
            "running",
            stage="evaluation_prepare",
            progress=0,
            message="preparing evaluation",
        )

    samples = _normalize_eval_items(cfg)
    if not samples:
        raise ValueError("Evaluation dataset is empty after normalization")

    model, tokenizer, model_label = _load_eval_model(cfg, training_result)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = Path(cfg.outputs.eval_dir) / cfg.job_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    total = len(samples)

    if reporter:
        reporter.report_status(
            "running",
            stage="evaluation",
            progress=0,
            message="evaluation started",
            extra={"samples": total, "model": model_label},
        )

    model.eval()

    for index, sample in enumerate(samples, start=1):
        prompt = _build_prompt(
            tokenizer,
            cfg.evaluation.prompt_template,
            sample,
            system_prompt=cfg.evaluation.system_prompt,
        )

        row = {
            "sampleId": sample["id"],
            "question": sample["question"],
            "candidateAnswer": sample["candidate_answer"],
            "referenceScore": sample["reference_score"],
            "maxScore": sample["max_score"],
            "hashTags": sample["hash_tags"],
            "predictedScore": None,
            "predictedFeedback": None,
            "rawResponse": None,
            "parseError": True,
            "inferenceError": False,
            "absoluteError": None,
        }

        try:
            encoded = tokenizer(prompt, return_tensors="pt")
            model_inputs = {k: v.to(model.device) for k, v in encoded.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=cfg.evaluation.max_new_tokens,
                    do_sample=cfg.evaluation.do_sample,
                    temperature=cfg.evaluation.temperature if cfg.evaluation.do_sample else 1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            raw_response = extract_generated_text(
                tokenizer=tokenizer,
                prompt_text=prompt,
                output_ids=output_ids,
                input_len=int(model_inputs["input_ids"].shape[1]),
            )

            parsed = parse_model_score(
                raw_response,
                parsing_regex=cfg.evaluation.parsing_regex,
                score_min=cfg.evaluation.score_min,
                score_max=cfg.evaluation.score_max,
            )

            row["rawResponse"] = raw_response
            row["predictedScore"] = parsed["score"]
            row["predictedFeedback"] = parsed["feedback"]
            row["parseError"] = parsed["parseError"]

            if isinstance(parsed["score"], (int, float)):
                row["absoluteError"] = abs(float(parsed["score"]) - float(sample["reference_score"]))

        except Exception as exc:
            row["rawResponse"] = ""
            row["inferenceError"] = True
            row["parseError"] = True
            row["error"] = str(exc)

        rows.append(row)

        if reporter and (index == 1 or index == total or index % max(1, total // 10) == 0):
            reporter.report_progress(
                stage="evaluation",
                progress=round((index / total) * 100, 2),
                message=f"evaluated {index}/{total} samples",
                extra={
                    "processed": index,
                    "total": total,
                    "model": model_label,
                },
            )

    metrics = calculate_metrics(model_label, rows)

    summary_json_path = output_dir / "summary.json"
    result_json_path = output_dir / "result.json"
    summary_csv_path = output_dir / "summary.csv"
    detailed_csv_path = output_dir / "detailed.csv"

    with result_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_label,
                "metrics": metrics,
                "rows": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "samples",
                "parseSuccessRate",
                "mae",
                "rmse",
                "exactRate",
                "within1Rate",
                "within2Rate",
                "meanSignedError",
                "avgPredictedScore",
                "parseErrors",
                "inferenceErrors",
                "emptyResponses",
            ],
        )
        writer.writeheader()
        writer.writerow(metrics)

    with detailed_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sampleId",
            "question",
            "candidateAnswer",
            "referenceScore",
            "maxScore",
            "predictedScore",
            "absoluteError",
            "parseError",
            "inferenceError",
            "hashTags",
            "predictedFeedback",
            "rawResponse",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sampleId": row.get("sampleId"),
                    "question": row.get("question"),
                    "candidateAnswer": row.get("candidateAnswer"),
                    "referenceScore": row.get("referenceScore"),
                    "maxScore": row.get("maxScore"),
                    "predictedScore": row.get("predictedScore"),
                    "absoluteError": row.get("absoluteError"),
                    "parseError": row.get("parseError"),
                    "inferenceError": row.get("inferenceError"),
                    "hashTags": ", ".join(row.get("hashTags") or []),
                    "predictedFeedback": row.get("predictedFeedback"),
                    "rawResponse": row.get("rawResponse"),
                }
            )

    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass

    if reporter:
        reporter.report_status(
            "running",
            stage="evaluation_completed",
            progress=100,
            message="evaluation completed",
            extra=metrics,
        )

    return {
        "enabled": True,
        "target": cfg.evaluation.target,
        "model": model_label,
        "summary": metrics,
        "summary_json_path": str(summary_json_path),
        "result_json_path": str(result_json_path),
        "summary_csv_path": str(summary_csv_path),
        "detailed_csv_path": str(detailed_csv_path),
    }