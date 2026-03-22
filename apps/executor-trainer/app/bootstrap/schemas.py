from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_EVAL_PROMPT = """Ты — беспристрастный эксперт-оценщик. Тебе предоставлен вопрос и ответ кандидата.
Твоя задача — выставить оценку от 0 до 5, где:
5 — идеально правильный и полный ответ,
0 — совершенно неверный или отсутствующий ответ.

Теги вопроса: ${tagsText}
Эти теги являются метаданными. Не завышай и не занижай оценку только из-за тегов.

Вопрос: ${question}
Ответ кандидата: ${candidateAnswer}

Верни результат строго в формате JSON:
{
  "score": <число от 0 до 5>,
  "feedback": "<краткое пояснение>"
}

Если не можешь вернуть JSON, обязательно напиши в конце: "score: X/5"
"""


class AppBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ModelConfig(AppBaseModel):
    source: Literal["local", "huggingface"] = "local"
    local_path: Optional[str] = None
    repo_id: Optional[str] = None
    revision: str = "main"
    trust_remote_code: bool = False
    load_in_4bit: Optional[bool] = None
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    max_seq_length: int = 4096

    base_model: Optional[str] = None
    base_model_name_or_path: Optional[str] = None

    @staticmethod
    def _is_probably_local_path(value: str) -> bool:
        value = str(value or "").strip()
        if not value:
            return False
        if value.startswith("/"):
            return True
        if value.startswith("./") or value.startswith("../"):
            return True
        if value.startswith("\\") or "\\" in value:
            return True
        if len(value) > 1 and value[1] == ":":
            return True
        return False

    @property
    def logical_base_model_id(self) -> Optional[str]:
        for candidate in (
            self.repo_id,
            self.base_model_name_or_path,
            self.base_model,
        ):
            if not isinstance(candidate, str):
                continue
            value = candidate.strip()
            if not value:
                continue
            if self._is_probably_local_path(value):
                continue
            return value
        return None

    @model_validator(mode="after")
    def validate_source(self) -> "ModelConfig":
        if self.source == "local" and not self.local_path:
            raise ValueError("model.local_path is required when model.source='local'")
        if self.source == "huggingface" and not self.repo_id:
            raise ValueError("model.repo_id is required when model.source='huggingface'")
        return self


class DatasetConfig(AppBaseModel):
    source: Literal["local", "url"] = "local"
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    train_url: Optional[str] = None
    val_url: Optional[str] = None
    format: Literal["instruction_output", "messages", "prompt_completion"] = "instruction_output"
    input_field: str = "input"
    output_field: str = "output"
    messages_field: str = "messages"

    @model_validator(mode="after")
    def validate_dataset(self) -> "DatasetConfig":
        if self.source == "local" and not self.train_path:
            raise ValueError("dataset.train_path is required when dataset.source='local'")
        if self.source == "url" and not self.train_url:
            raise ValueError("dataset.train_url is required when dataset.source='url'")
        return self


class TrainingConfig(AppBaseModel):
    method: Literal["lora", "qlora"] = "qlora"
    max_seq_length: int = 4096
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 1
    save_steps: int = 50
    eval_steps: int = 50
    bf16: bool = True
    packing: bool = True
    save_total_limit: int = 2
    optim: str = "adamw_8bit"
    output_dir: Optional[str] = None


class LoraConfig(AppBaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    target_modules: List[str] = Field(default_factory=list)


class OutputsConfig(AppBaseModel):
    base_dir: str
    logs_dir: Optional[str] = None
    lora_dir: Optional[str] = None
    checkpoints_dir: Optional[str] = None
    metrics_dir: Optional[str] = None
    merged_dir: Optional[str] = None
    quantized_dir: Optional[str] = None
    eval_dir: Optional[str] = None
    downloads_dir: Optional[str] = None

    @model_validator(mode="after")
    def fill_defaults(self) -> "OutputsConfig":
        base = Path(self.base_dir)

        if not self.logs_dir:
            self.logs_dir = str(base / "logs")
        if not self.lora_dir:
            self.lora_dir = str(base / "lora")
        if not self.checkpoints_dir:
            self.checkpoints_dir = str(base / "checkpoints")
        if not self.metrics_dir:
            self.metrics_dir = str(base / "metrics")
        if not self.merged_dir:
            self.merged_dir = str(base / "merged")
        if not self.quantized_dir:
            self.quantized_dir = str(base / "quantized")
        if not self.eval_dir:
            self.eval_dir = str(base / "evaluation")
        if not self.downloads_dir:
            self.downloads_dir = str(base / "downloads")

        return self


class PostprocessConfig(AppBaseModel):
    merge_lora: bool = True
    save_merged_16bit: bool = True
    run_awq_quantization: bool = False


class AuthConfig(AppBaseModel):
    bearer_token: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)


class CallbackConfig(AppBaseModel):
    enabled: bool = False
    url: Optional[str] = None
    timeout_sec: int = 15
    auth: AuthConfig = Field(default_factory=AuthConfig)

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.url)


class ReportingConfig(AppBaseModel):
    status: CallbackConfig = Field(default_factory=CallbackConfig)
    progress: CallbackConfig = Field(default_factory=CallbackConfig)
    final: CallbackConfig = Field(default_factory=CallbackConfig)
    logs: CallbackConfig = Field(default_factory=CallbackConfig)


class EvaluationDatasetConfig(AppBaseModel):
    source: Literal["local", "url"] = "local"
    path: Optional[str] = None
    url: Optional[str] = None
    format: Literal["json", "jsonl"] = "jsonl"
    question_field: str = "question"
    answer_field: str = "candidate_answer"
    score_field: str = "reference_score"
    max_score_field: Optional[str] = "max_score"
    tags_field: Optional[str] = "hash_tags"

    @model_validator(mode="after")
    def validate_source(self) -> "EvaluationDatasetConfig":
        if self.source == "local" and not self.path:
            raise ValueError("evaluation.dataset.path is required when source='local'")
        if self.source == "url" and not self.url:
            raise ValueError("evaluation.dataset.url is required when source='url'")
        return self


class EvaluationConfig(AppBaseModel):
    enabled: bool = False
    target: Literal["auto", "lora", "merged"] = "auto"
    max_samples: Optional[int] = None
    max_new_tokens: int = 128
    temperature: float = 0.0
    do_sample: bool = False
    system_prompt: Optional[str] = None
    prompt_template: str = DEFAULT_EVAL_PROMPT
    parsing_regex: Optional[str] = None
    score_min: float = 0.0
    score_max: float = 5.0
    dataset: Optional[EvaluationDatasetConfig] = None

    @model_validator(mode="after")
    def validate_enabled(self) -> "EvaluationConfig":
        if self.enabled and self.dataset is None:
            raise ValueError("evaluation.dataset is required when evaluation.enabled=true")
        return self


class UrlArtifactTargets(AppBaseModel):
    logs_url: Optional[str] = None
    effective_config_url: Optional[str] = None
    summary_url: Optional[str] = None
    train_metrics_url: Optional[str] = None
    train_history_url: Optional[str] = None
    eval_summary_url: Optional[str] = None
    eval_details_url: Optional[str] = None
    lora_archive_url: Optional[str] = None
    merged_archive_url: Optional[str] = None
    full_archive_url: Optional[str] = None


class UploadConfig(AppBaseModel):
    enabled: bool = False
    target: Literal["local", "huggingface", "url"] = "local"
    upload_url: Optional[str] = None
    timeout_sec: int = 120
    auth: AuthConfig = Field(default_factory=AuthConfig)
    url_targets: UrlArtifactTargets = Field(default_factory=UrlArtifactTargets)

    repo_id_lora: Optional[str] = None
    repo_id_merged: Optional[str] = None
    repo_id_metadata: Optional[str] = None
    private: bool = True
    commit_message: str = "trainer-service upload"


class HuggingFacePublishConfig(AppBaseModel):
    enabled: bool = False
    push_lora: bool = False
    push_merged: bool = False
    repo_id_lora: Optional[str] = None
    repo_id_merged: Optional[str] = None
    repo_id_metadata: Optional[str] = None
    private: bool = True
    commit_message: str = "trainer-service upload"
    revision: Optional[str] = None


class PipelineStage(AppBaseModel):
    enabled: bool = True


class TrainingStage(PipelineStage, TrainingConfig):
    pass


class MergeStage(PipelineStage, PostprocessConfig):
    pass


class EvaluationStage(PipelineStage, EvaluationConfig):
    pass


class PublishStage(PipelineStage, HuggingFacePublishConfig):
    pass


class UploadStage(PipelineStage, UploadConfig):
    pass


class PipelineConfig(AppBaseModel):
    prepare_assets: PipelineStage = Field(default_factory=PipelineStage)
    training: TrainingStage = Field(default_factory=TrainingStage)
    merge: MergeStage = Field(default_factory=MergeStage)
    evaluation: EvaluationStage = Field(default_factory=EvaluationStage)
    publish: PublishStage = Field(default_factory=PublishStage)
    upload: UploadStage = Field(default_factory=UploadStage)


class JobConfig(AppBaseModel):
    job_id: Optional[str] = None
    job_name: str
    mode: Literal["local", "remote", "auto"] = "auto"

    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    lora: LoraConfig
    outputs: OutputsConfig

    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    upload: UploadConfig = Field(default_factory=UploadConfig)
    huggingface: HuggingFacePublishConfig = Field(default_factory=HuggingFacePublishConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    pipeline: Optional[PipelineConfig] = None

    report_url: Optional[str] = None
