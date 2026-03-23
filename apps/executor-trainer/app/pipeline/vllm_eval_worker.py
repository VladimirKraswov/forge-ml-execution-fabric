from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def resolve_vllm_dtype(dtype_value: str) -> str:
    value = str(dtype_value or "auto").lower().strip()
    if value in {"auto", "float16", "bfloat16", "float32"}:
        return value
    if value in {"half", "fp16"}:
        return "float16"
    if value in {"bf16"}:
        return "bfloat16"
    if value in {"float", "fp32"}:
        return "float32"
    return "auto"


def _get_by_path(payload: Dict[str, Any], path: Optional[str], default: Any = None) -> Any:
    if not path:
        return default

    current: Any = payload
    for part in str(path).split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def _normalize_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []

def _sanitize_tokenizer_config(model_dir: str) -> None:
    model_path = Path(model_dir)
    config_path = model_path / "tokenizer_config.json"

    if not config_path.exists():
        return

    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("failed to read tokenizer config %s: %s", config_path, exc)
        return

    changed = False

    extra_special_tokens = data.get("extra_special_tokens")
    if isinstance(extra_special_tokens, list):
        additional = data.get("additional_special_tokens")
        if not isinstance(additional, list):
            additional = []

        for token in extra_special_tokens:
            token_str = str(token).strip()
            if token_str and token_str not in additional:
                additional.append(token_str)

        data["additional_special_tokens"] = additional
        data.pop("extra_special_tokens", None)
        changed = True

    if changed:
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("sanitized tokenizer config: %s", config_path)


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

    patterns: List[str] = []
    if parsing_regex:
        patterns.append(parsing_regex)

    patterns.extend(
        [
            fr"score:\s*(\d+(?:\.\d+)?)\s*/\s*{int(score_max)}",
            fr"оценка:\s*(\d+(?:\.\d+)?)\s*/\s*{int(score_max)}",
            r"score:\s*(\d+(?:\.\d+)?)",
            r"оценка:\s*(\d+(?:\.\d+)?)",
            fr"(\d+(?:\.\d+)?)\s*/\s*{int(score_max)}",
            r"^(\d+(?:\.\d+)?)$",
        ]
    )

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


def _quadratic_score_error(predicted: float, reference: float) -> float:
    return (float(predicted) ** 2 - float(reference) ** 2) ** 2


def calculate_metrics(model_label: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = [
        row
        for row in rows
        if not row.get("parseError") and isinstance(row.get("predictedScore"), (int, float))
    ]
    total_samples = len(rows)

    if not valid_rows:
        parse_ok = 0.0
        return {
            "model": model_label,
            "samples": total_samples,
            "parseSuccessRate": parse_ok,
            "parseOkRate": parse_ok,
            "parseOk": parse_ok,
            "mae": None,
            "rmse": None,
            "exactRate": 0.0,
            "exact": 0.0,
            "within1Rate": 0.0,
            "plus1Rate": 0.0,
            "plus1": 0.0,
            "within2Rate": 0.0,
            "plus2Rate": 0.0,
            "plus2": 0.0,
            "meanSignedError": None,
            "bias": None,
            "avgPredictedScore": None,
            "meanQuadraticScoreError": None,
            "parseErrors": sum(1 for row in rows if row.get("parseError")),
            "inferenceErrors": sum(1 for row in rows if row.get("inferenceError")),
            "emptyResponses": sum(1 for row in rows if not str(row.get("rawResponse") or "").strip()),
        }

    n = len(valid_rows)
    abs_errors: List[float] = []
    sq_errors: List[float] = []
    signed_errors: List[float] = []
    quadratic_errors: List[float] = []
    exact = 0
    within1 = 0
    within2 = 0
    predicted_sum = 0.0

    for row in valid_rows:
        predicted = float(row["predictedScore"])
        reference = float(row["referenceScore"])
        error = predicted - reference
        abs_error = abs(error)

        abs_errors.append(abs_error)
        sq_errors.append(error * error)
        signed_errors.append(error)
        quadratic_errors.append(_quadratic_score_error(predicted, reference))
        predicted_sum += predicted

        if abs_error == 0:
            exact += 1
        if abs_error <= 1:
            within1 += 1
        if abs_error <= 2:
            within2 += 1

    parse_ok = n / total_samples if total_samples else 0.0
    bias = sum(signed_errors) / n
    exact_rate = exact / n
    plus1_rate = within1 / n
    plus2_rate = within2 / n

    return {
        "model": model_label,
        "samples": total_samples,
        "parseSuccessRate": parse_ok,
        "parseOkRate": parse_ok,
        "parseOk": parse_ok,
        "mae": sum(abs_errors) / n,
        "rmse": math.sqrt(sum(sq_errors) / n),
        "exactRate": exact_rate,
        "exact": exact_rate,
        "within1Rate": plus1_rate,
        "plus1Rate": plus1_rate,
        "plus1": plus1_rate,
        "within2Rate": plus2_rate,
        "plus2Rate": plus2_rate,
        "plus2": plus2_rate,
        "meanSignedError": bias,
        "bias": bias,
        "avgPredictedScore": predicted_sum / n,
        "meanQuadraticScoreError": sum(quadratic_errors) / n,
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


def _normalize_eval_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    eval_cfg = payload["evaluation"]
    ds_cfg = eval_cfg["dataset"]

    if not ds_cfg.get("path"):
        raise ValueError("evaluation.dataset.path is required before evaluation starts")

    raw_items = _load_eval_items(ds_cfg["path"], ds_cfg["format"])
    normalized: List[Dict[str, Any]] = []

    for idx, item in enumerate(raw_items):
        score_raw = _get_by_path(item, ds_cfg.get("score_field"))
        if score_raw is None:
            continue

        try:
            reference_score = float(score_raw)
        except Exception:
            continue

        max_score_field = ds_cfg.get("max_score_field")
        max_score_raw = _get_by_path(item, max_score_field, 5) if max_score_field else 5
        try:
            max_score = float(max_score_raw)
        except Exception:
            max_score = 5.0

        tags = _normalize_tags(_get_by_path(item, ds_cfg.get("tags_field")))

        sample_id = (
            item.get("id")
            or _get_by_path(item, "details.plain_ind")
            or _get_by_path(item, "details.question_ind")
            or f"sample_{idx + 1}"
        )

        if ds_cfg.get("task") == "score_prediction":
            prompt_value = _get_by_path(item, ds_cfg.get("prompt_field"))
            messages_value = _get_by_path(item, ds_cfg.get("messages_field"))

            if prompt_value is None and messages_value is None:
                continue

            normalized.append(
                {
                    "id": str(sample_id),
                    "prompt": str(prompt_value) if prompt_value is not None else None,
                    "messages": messages_value if isinstance(messages_value, list) else None,
                    "reference_score": reference_score,
                    "max_score": max_score,
                    "hash_tags": tags,
                    "raw_item": item,
                }
            )
            continue

        question = _get_by_path(item, ds_cfg.get("question_field"))
        answer = _get_by_path(item, ds_cfg.get("answer_field"))
        if question is None or answer is None:
            continue

        normalized.append(
            {
                "id": str(sample_id),
                "question": str(question),
                "candidate_answer": str(answer),
                "reference_score": reference_score,
                "max_score": max_score,
                "hash_tags": tags,
                "raw_item": item,
            }
        )

    max_samples = eval_cfg.get("max_samples")
    if max_samples:
        normalized = normalized[: int(max_samples)]

    return normalized


def _resolve_eval_target(payload: Dict[str, Any]) -> str:
    target = payload["evaluation"].get("target", "auto")
    if target == "auto":
        return "merged" if payload["training_result"].get("merged_dir") else "lora"
    return target


def _iter_batches(items: Sequence[Any], batch_size: int) -> Iterator[Tuple[int, Sequence[Any]]]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def _flatten_messages(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", ""))
        parts.append(f"{role}: {content}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


def _build_score_prediction_prompt(dataset_cfg: Dict[str, Any], sample: Dict[str, Any]) -> str:
    messages = sample.get("messages")
    if isinstance(messages, list) and messages:
        return _flatten_messages(messages)

    prompt_text = str(sample.get("prompt") or "")
    dataset_format = dataset_cfg.get("format", "instruction_output")

    if dataset_format == "instruction_output":
        return f"### Instruction:\n{prompt_text}\n\n### Response:\n"

    if dataset_format == "prompt_completion":
        return prompt_text

    if dataset_format == "messages":
        return _flatten_messages([{"role": "user", "content": prompt_text}])

    return prompt_text


def _build_judge_prompt(eval_cfg: Dict[str, Any], sample: Dict[str, Any]) -> str:
    rendered = render_prompt_template(
        eval_cfg.get("prompt_template", ""),
        {
            "question": sample["question"],
            "candidate_answer": sample["candidate_answer"],
            "reference_score": sample["reference_score"],
            "max_score": sample["max_score"],
            "hash_tags": sample["hash_tags"],
        },
    )

    if eval_cfg.get("system_prompt"):
        return f"{eval_cfg['system_prompt']}\n\n{rendered}"
    return rendered


def _extract_vllm_text(output: Any) -> str:
    try:
        outputs = getattr(output, "outputs", None) or []
        if not outputs:
            return ""
        text = getattr(outputs[0], "text", "") or ""
        return str(text).strip()
    except Exception:
        return ""


def _build_vllm_runtime(payload: Dict[str, Any]):
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    resolved_target = _resolve_eval_target(payload)
    model_cfg = payload["model"]
    eval_cfg = payload["evaluation"]
    lora_cfg = payload["lora"]
    training_result = payload["training_result"]

    dtype = resolve_vllm_dtype(model_cfg.get("dtype", "auto"))

    engine_args: Dict[str, Any] = {
        "tensor_parallel_size": max(1, int(eval_cfg.get("tensor_parallel_size", 1))),
        "dtype": dtype,
        "gpu_memory_utilization": float(eval_cfg.get("gpu_memory_utilization", 0.9)),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        "enforce_eager": bool(eval_cfg.get("enforce_eager", False)),
        "disable_log_stats": True,
    }

    if eval_cfg.get("max_num_seqs"):
        engine_args["max_num_seqs"] = int(eval_cfg["max_num_seqs"])

    if eval_cfg.get("max_num_batched_tokens"):
        engine_args["max_num_batched_tokens"] = int(eval_cfg["max_num_batched_tokens"])

    lora_request = None

    if resolved_target == "merged":
        merged_dir = training_result.get("merged_dir")
        if not merged_dir or not Path(merged_dir).exists():
            raise ValueError("Merged model directory is missing, but evaluation.target='merged'")

        _sanitize_tokenizer_config(str(merged_dir))

        model_label = str(merged_dir)
        llm = LLM(model=merged_dir, **engine_args)
        return llm, model_label, resolved_target, lora_request

    base_model = model_cfg.get("local_path") if model_cfg.get("source") == "local" else model_cfg.get("repo_id")
    if not base_model:
        raise ValueError("Cannot resolve base model for LoRA evaluation")

    lora_dir = training_result.get("lora_dir")
    if not lora_dir or not Path(lora_dir).exists():
        raise ValueError("LoRA adapter directory is missing, but evaluation.target='lora'")

    if Path(str(base_model)).exists():
        _sanitize_tokenizer_config(str(base_model))

    engine_args["enable_lora"] = True
    engine_args["max_loras"] = 1
    engine_args["max_lora_rank"] = max(16, int(lora_cfg.get("r", 16)))

    llm = LLM(model=base_model, **engine_args)
    lora_request = LoRARequest("eval_lora", 1, str(lora_dir))

    return llm, f"{base_model}+lora", resolved_target, lora_request


def _build_eval_prompts(payload: Dict[str, Any], samples: Sequence[Dict[str, Any]]) -> List[str]:
    ds_cfg = payload["evaluation"]["dataset"]
    prompts: List[str] = []
    for sample in samples:
        if ds_cfg.get("task") == "score_prediction":
            prompts.append(_build_score_prediction_prompt(payload["dataset"], sample))
        else:
            prompts.append(_build_judge_prompt(payload["evaluation"], sample))
    return prompts


def run_worker(request: Dict[str, Any]) -> Dict[str, Any]:
    eval_cfg = request["evaluation"]
    ds_cfg = eval_cfg["dataset"]
    job_name = request["job_name"]

    samples = _normalize_eval_items(request)
    if not samples:
        raise ValueError("Evaluation dataset is empty after normalization")

    llm, model_label, resolved_target, lora_request = _build_vllm_runtime(request)

    output_dir = Path(request["outputs"]["eval_dir"]) / job_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    total = len(samples)

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=int(eval_cfg.get("max_new_tokens", 32)),
        temperature=float(eval_cfg.get("temperature", 0.0)) if eval_cfg.get("do_sample") else 0.0,
    )

    batch_size = max(1, int(eval_cfg.get("batch_size", 16)))

    for start, batch in _iter_batches(samples, batch_size):
        prompts = _build_eval_prompts(request, batch)

        batch_rows: List[Dict[str, Any]] = []
        for sample, prompt in zip(batch, prompts):
            batch_rows.append(
                {
                    "sampleId": sample["id"],
                    "inputText": prompt,
                    "referenceScore": sample["reference_score"],
                    "maxScore": sample["max_score"],
                    "predictedScore": None,
                    "absoluteError": None,
                    "quadraticScoreError": None,
                    "parseError": True,
                    "inferenceError": False,
                    "predictedFeedback": None,
                    "rawResponse": None,
                    "hashTags": sample.get("hash_tags") or [],
                    "error": None,
                }
            )

        try:
            outputs = llm.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
                lora_request=lora_request,
            )
        except Exception as exc:
            error_text = str(exc)
            logger.exception("evaluation batch failed: start=%s size=%s", start, len(batch))
            for row in batch_rows:
                row["rawResponse"] = ""
                row["inferenceError"] = True
                row["parseError"] = True
                row["error"] = error_text
            rows.extend(batch_rows)
            continue

        for idx, row in enumerate(batch_rows):
            if idx >= len(outputs):
                row["rawResponse"] = ""
                row["inferenceError"] = True
                row["parseError"] = True
                row["error"] = "Missing output from vLLM"
                continue

            raw_response = _extract_vllm_text(outputs[idx])
            parsed = parse_model_score(
                raw_response,
                parsing_regex=eval_cfg.get("parsing_regex"),
                score_min=float(eval_cfg.get("score_min", 0.0)),
                score_max=float(eval_cfg.get("score_max", 5.0)),
            )

            row["rawResponse"] = raw_response
            row["predictedScore"] = parsed["score"]
            row["predictedFeedback"] = parsed["feedback"]
            row["parseError"] = parsed["parseError"]

            if isinstance(parsed["score"], (int, float)):
                predicted = float(parsed["score"])
                reference = float(row["referenceScore"])
                row["absoluteError"] = abs(predicted - reference)
                row["quadraticScoreError"] = _quadratic_score_error(predicted, reference)

        rows.extend(batch_rows)
        logger.info("==> evaluated %s/%s samples", min(start + len(batch), total), total)

    metrics = calculate_metrics(model_label, rows)

    summary_json_path = output_dir / "summary.json"
    result_json_path = output_dir / "result.json"
    summary_csv_path = output_dir / "summary.csv"
    detailed_csv_path = output_dir / "detailed.csv"

    result_payload = {
        "enabled": True,
        "engine": eval_cfg.get("engine", "vllm"),
        "target": resolved_target,
        "task": ds_cfg.get("task", "score_prediction"),
        "model": model_label,
        "summary": metrics,
        "summary_json_path": str(summary_json_path),
        "result_json_path": str(result_json_path),
        "summary_csv_path": str(summary_csv_path),
        "detailed_csv_path": str(detailed_csv_path),
    }

    with result_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_label,
                "engine": eval_cfg.get("engine", "vllm"),
                "target": resolved_target,
                "task": ds_cfg.get("task", "score_prediction"),
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
                "mae",
                "rmse",
                "exact",
                "plus1",
                "plus2",
                "bias",
                "parseOk",
                "avgPredictedScore",
                "meanQuadraticScoreError",
                "parseErrors",
                "inferenceErrors",
                "emptyResponses",
                "parseSuccessRate",
                "parseOkRate",
                "exactRate",
                "within1Rate",
                "within2Rate",
                "plus1Rate",
                "plus2Rate",
                "meanSignedError",
            ],
        )
        writer.writeheader()
        writer.writerow(metrics)

    with detailed_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sampleId",
            "inputText",
            "referenceScore",
            "maxScore",
            "predictedScore",
            "absoluteError",
            "quadraticScoreError",
            "parseError",
            "inferenceError",
            "hashTags",
            "predictedFeedback",
            "rawResponse",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sampleId": row.get("sampleId"),
                    "inputText": row.get("inputText"),
                    "referenceScore": row.get("referenceScore"),
                    "maxScore": row.get("maxScore"),
                    "predictedScore": row.get("predictedScore"),
                    "absoluteError": row.get("absoluteError"),
                    "quadraticScoreError": row.get("quadraticScoreError"),
                    "parseError": row.get("parseError"),
                    "inferenceError": row.get("inferenceError"),
                    "hashTags": ", ".join(row.get("hashTags") or []),
                    "predictedFeedback": row.get("predictedFeedback"),
                    "rawResponse": row.get("rawResponse"),
                    "error": row.get("error"),
                }
            )

    try:
        del llm
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return result_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()

    request_path = Path(args.request)
    response_path = Path(args.response)

    try:
        with request_path.open("r", encoding="utf-8") as f:
            request = json.load(f)

        result = run_worker(request)
        response_path.parent.mkdir(parents=True, exist_ok=True)
        with response_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        error_payload = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        response_path.parent.mkdir(parents=True, exist_ok=True)
        with response_path.open("w", encoding="utf-8") as f:
            json.dump(error_payload, f, indent=2, ensure_ascii=False)
        print(error_payload["traceback"], file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()