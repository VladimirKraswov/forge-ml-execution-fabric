from __future__ import annotations

import inspect
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import torch
import os
from datasets import load_dataset
from transformers import TrainerCallback, TrainingArguments
from trl import SFTTrainer

from ..adapters.reporter import Reporter
from ..bootstrap.schemas import JobConfig

logger = logging.getLogger(__name__)


def ensure_dirs(cfg: JobConfig) -> None:
    for path in [
        cfg.outputs.base_dir,
        cfg.outputs.logs_dir,
        cfg.outputs.lora_dir,
        cfg.outputs.checkpoints_dir,
        cfg.outputs.metrics_dir,
        cfg.outputs.merged_dir,
        cfg.outputs.quantized_dir,
        cfg.outputs.eval_dir,
        cfg.outputs.downloads_dir,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def resolve_model_args(cfg: JobConfig) -> tuple[str, bool, Optional[str]]:
    if cfg.model.source == "local":
        model_name = cfg.model.local_path
    else:
        model_name = cfg.model.repo_id

    if not model_name:
        raise ValueError("Model path/repo_id is empty")

    load_in_4bit = (
        bool(cfg.model.load_in_4bit)
        if cfg.model.load_in_4bit is not None
        else (cfg.training.method == "qlora")
    )

    logical_base_model_id = cfg.model.logical_base_model_id
    return model_name, load_in_4bit, logical_base_model_id


def safe_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except Exception:
        pass
    return None


def build_text_from_messages(messages: list[dict], tokenizer) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.upper()}: {content}")
        return "\n".join(parts)


def format_example(example: Dict[str, Any], cfg: JobConfig, tokenizer) -> Dict[str, str]:
    eos_token = tokenizer.eos_token or ""

    if cfg.dataset.format == "instruction_output":
        input_value = str(example.get(cfg.dataset.input_field, "") or "").strip()
        output_value = str(example.get(cfg.dataset.output_field, "") or "").strip()
        text = (
            f"### Instruction:\n{input_value}\n\n"
            f"### Response:\n{output_value}{eos_token}"
        )
        return {"text": text}

    if cfg.dataset.format == "prompt_completion":
        prompt_value = str(example.get(cfg.dataset.input_field, "") or "").strip()
        completion_value = str(example.get(cfg.dataset.output_field, "") or "").strip()
        text = f"{prompt_value}{completion_value}{eos_token}"
        return {"text": text}

    if cfg.dataset.format == "messages":
        messages = example.get(cfg.dataset.messages_field) or example.get("messages") or []
        if not isinstance(messages, list):
            return {"text": ""}
        text = build_text_from_messages(messages, tokenizer)
        if eos_token and not text.endswith(eos_token):
            text = f"{text}{eos_token}"
        return {"text": text}

    raise ValueError(f"Unsupported dataset format: {cfg.dataset.format}")


class TrainingProgressCallback(TrainerCallback):
    def __init__(self, reporter: Optional[Reporter]):
        self.reporter = reporter

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.reporter:
            return
        self.reporter.report_status(
            "running",
            stage="training",
            progress=0,
            message="training started",
            extra={"max_steps": int(state.max_steps or 0)},
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.reporter or not logs:
            return

        max_steps = int(state.max_steps or 0)
        progress = round((int(state.global_step) / max_steps) * 100, 2) if max_steps > 0 else None

        payload = {
            "step": int(state.global_step),
            "epoch": safe_float(state.epoch),
            **logs,
        }

        self.reporter.report_progress(
            stage="training",
            progress=progress,
            message="trainer log",
            extra=payload,
        )

    def on_save(self, args, state, control, **kwargs):
        if not self.reporter:
            return

        max_steps = int(state.max_steps or 0)
        progress = round((int(state.global_step) / max_steps) * 100, 2) if max_steps > 0 else None

        self.reporter.report_progress(
            stage="training",
            progress=progress,
            message="checkpoint saved",
            extra={"step": int(state.global_step)},
        )

    def on_train_end(self, args, state, control, **kwargs):
        if not self.reporter:
            return
        self.reporter.report_status(
            "running",
            stage="training",
            progress=100,
            message="training finished",
            extra={"step": int(state.global_step)},
        )


def _build_training_args(cfg: JobConfig, checkpoint_dir: str, has_validation: bool) -> TrainingArguments:
    common_kwargs = {
        "output_dir": checkpoint_dir,
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "warmup_ratio": cfg.training.warmup_ratio,
        "num_train_epochs": cfg.training.num_train_epochs,
        "learning_rate": cfg.training.learning_rate,
        "bf16": cfg.training.bf16,
        "fp16": not cfg.training.bf16,
        "logging_steps": cfg.training.logging_steps,
        "save_steps": cfg.training.save_steps,
        "save_strategy": "steps",
        "save_total_limit": cfg.training.save_total_limit,
        "optim": cfg.training.optim,
        "push_to_hub": False,
        "logging_dir": cfg.outputs.logs_dir,
    }

    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())

    filtered_kwargs = {k: v for k, v in common_kwargs.items() if k in supported}

    if "report_to" in supported:
        filtered_kwargs["report_to"] = "none"

    if has_validation:
        if "eval_steps" in supported:
            filtered_kwargs["eval_steps"] = cfg.training.eval_steps

        if "eval_strategy" in supported:
            filtered_kwargs["eval_strategy"] = "steps"
        elif "evaluation_strategy" in supported:
            filtered_kwargs["evaluation_strategy"] = "steps"
    else:
        if "eval_strategy" in supported:
            filtered_kwargs["eval_strategy"] = "no"
        elif "evaluation_strategy" in supported:
            filtered_kwargs["evaluation_strategy"] = "no"

    if "save_safetensors" in supported:
        filtered_kwargs["save_safetensors"] = True

    return TrainingArguments(**filtered_kwargs)


def _build_sft_trainer(
    cfg: JobConfig,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args: TrainingArguments,
):
    sig = inspect.signature(SFTTrainer.__init__)
    supported = set(sig.parameters.keys())

    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
    }

    if eval_dataset is not None and "eval_dataset" in supported:
        kwargs["eval_dataset"] = eval_dataset

    if "processing_class" in supported:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in supported:
        kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in supported:
        kwargs["dataset_text_field"] = "text"

    if "packing" in supported:
        kwargs["packing"] = cfg.training.packing

    if "max_seq_length" in supported:
        kwargs["max_seq_length"] = cfg.training.max_seq_length

    return SFTTrainer(**kwargs)


def run_training(cfg: JobConfig, reporter: Optional[Reporter] = None) -> dict:
    ensure_dirs(cfg)

    model_name, load_in_4bit, logical_base_model_id = resolve_model_args(cfg)

    logger.info("==> loading model: %s", model_name)
    if logical_base_model_id:
        logger.info("==> logical base model id: %s", logical_base_model_id)
    logger.info("==> method: %s", cfg.training.method)

    if reporter:
        reporter.report_status(
            "running",
            stage="load_model",
            progress=5,
            message="loading base model",
            extra={
                "model_name": model_name,
                "logical_base_model_id": logical_base_model_id,
                "load_in_4bit": load_in_4bit,
                "dtype": cfg.model.dtype,
            },
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=cfg.training.max_seq_length or cfg.model.max_seq_length,
        dtype=torch.bfloat16 if cfg.training.bf16 else None,
        load_in_4bit=load_in_4bit,
        local_files_only=(cfg.model.source == "local"),
        trust_remote_code=cfg.model.trust_remote_code,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", None) != "right":
        tokenizer.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        target_modules=cfg.lora.target_modules,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        use_gradient_checkpointing=cfg.lora.use_gradient_checkpointing,
        random_state=cfg.lora.random_state,
    )

    if not cfg.dataset.train_path:
        raise ValueError("dataset.train_path is required for training")

    if reporter:
        reporter.report_status(
            "running",
            stage="load_dataset",
            progress=15,
            message="loading dataset",
        )

    data_files = {"train": cfg.dataset.train_path}
    if cfg.dataset.val_path:
        data_files["validation"] = cfg.dataset.val_path

    dataset = load_dataset("json", data_files=data_files)

    dataset = dataset.map(
        lambda example: format_example(example, cfg, tokenizer),
        desc="Formatting dataset",
    )
    dataset = dataset.filter(
        lambda row: bool(str(row.get("text", "")).strip()),
        desc="Filtering empty rows",
    )

    if len(dataset["train"]) == 0:
        raise ValueError("Train dataset is empty after formatting/filtering")

    checkpoint_dir = str(Path(cfg.outputs.checkpoints_dir) / cfg.job_name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    training_args = _build_training_args(
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
        has_validation="validation" in dataset,
    )

    trainer = _build_sft_trainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        training_args=training_args,
    )

    trainer.add_callback(TrainingProgressCallback(reporter))

    logger.info("==> training started")
    train_result = trainer.train()
    logger.info("==> training finished")

    lora_dir = Path(cfg.outputs.lora_dir) / cfg.job_name
    merged_dir = Path(cfg.outputs.merged_dir) / cfg.job_name
    metrics_path = Path(cfg.outputs.metrics_dir) / f"{cfg.job_name}.train_metrics.json"
    history_path = Path(cfg.outputs.metrics_dir) / f"{cfg.job_name}.train_history.json"
    train_summary_path = Path(cfg.outputs.metrics_dir) / f"{cfg.job_name}.train_summary.json"

    lora_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    if reporter:
        reporter.report_status(
            "running",
            stage="save_lora",
            progress=92,
            message="saving lora adapters",
            extra={"path": str(lora_dir)},
        )

    logger.info("==> saving lora adapters to %s", lora_dir)
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    merged_saved = False
    if cfg.postprocess.merge_lora and cfg.postprocess.save_merged_16bit:
        if reporter:
            reporter.report_status(
                "running",
                stage="merge_lora",
                progress=96,
                message="saving merged model",
                extra={"path": str(merged_dir)},
            )

        logger.info("==> saving merged model to %s", merged_dir)
        logger.info("==> merged 16-bit save may take several minutes for 7B model")
        model.save_pretrained_merged(
            save_directory=str(merged_dir),
            tokenizer=tokenizer,
            save_method="merged_16bit",
        )
        logger.info("==> merged model saved")
        merged_saved = True

    metrics = dict(train_result.metrics)
    history = trainer.state.log_history

    final_loss = None
    for entry in reversed(history):
        if "loss" in entry:
            final_loss = entry["loss"]
            break

    train_summary = {
        "job_name": cfg.job_name,
        "train_rows": len(dataset["train"]),
        "validation_rows": len(dataset["validation"]) if "validation" in dataset else 0,
        "method": cfg.training.method,
        "base_model": model_name,
        "base_model_id": logical_base_model_id,
        "base_model_name_or_path": logical_base_model_id,
        "load_in_4bit": load_in_4bit,
        "bf16": cfg.training.bf16,
        "merged_saved": merged_saved,
        "train_runtime": metrics.get("train_runtime"),
        "train_samples_per_second": metrics.get("train_samples_per_second"),
        "train_steps_per_second": metrics.get("train_steps_per_second"),
        "train_loss": metrics.get("train_loss"),
        "final_loss": final_loss,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    with train_summary_path.open("w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2, ensure_ascii=False)

    logger.info("==> training artifacts saved")

    if reporter:
        reporter.report_status(
            "running",
            stage="train_completed",
            progress=100,
            message="training stage completed",
            extra=train_summary,
        )

    try:
        del trainer
    except Exception:
        pass

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "status": "success",
        "job_name": cfg.job_name,
        "base_model": model_name,
        "base_model_id": logical_base_model_id,
        "base_model_name_or_path": logical_base_model_id,
        "lora_dir": str(lora_dir),
        "merged_dir": str(merged_dir) if merged_saved else None,
        "checkpoint_dir": checkpoint_dir,
        "metrics_path": str(metrics_path),
        "history_path": str(history_path),
        "train_summary_path": str(train_summary_path),
        "summary": train_summary,
    }