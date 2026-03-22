import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

sys.modules["unsloth"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["datasets"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["trl"] = MagicMock()

import runner
from schemas import JobConfig


class TestRemoteFlow(unittest.TestCase):
    def setUp(self):
        self.config_dict = {
            "job_id": "job-test-001",
            "job_name": "test-job",
            "mode": "remote",
            "model": {
                "source": "local",
                "local_path": "/app",
                "repo_id": "Qwen/Qwen2.5-7B-Instruct",
                "base_model": "Qwen/Qwen2.5-7B-Instruct",
                "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
            },
            "dataset": {
                "source": "url",
                "train_url": "http://example.com/train.json",
                "val_url": "http://example.com/val.json",
            },
            "training": {
                "method": "lora",
            },
            "lora": {
                "target_modules": ["q_proj", "v_proj"],
            },
            "outputs": {
                "base_dir": "test_outputs",
                "logs_dir": "test_outputs/logs",
                "lora_dir": "test_outputs/lora",
                "checkpoints_dir": "test_outputs/checkpoints",
                "metrics_dir": "test_outputs/metrics",
                "merged_dir": "test_outputs/merged",
                "quantized_dir": "test_outputs/quantized",
                "eval_dir": "test_outputs/evaluation",
                "downloads_dir": "test_outputs/downloads",
            },
            "postprocess": {
                "merge_lora": True,
            },
            "upload": {
                "enabled": True,
                "target": "url",
                "upload_url": "http://example.com/upload",
                "url_targets": {
                    "summary_url": "http://example.com/upload/summary",
                    "logs_url": "http://example.com/upload/logs",
                    "merged_archive_url": "http://example.com/upload/merged",
                },
            },
            "report_url": "http://example.com/report",
        }
        self.cfg = JobConfig.model_validate(self.config_dict)

    @patch("runner.load_config")
    @patch("runner.Reporter")
    @patch("runner.AssetManager")
    @patch("runner.Archiver")
    @patch("runner.ArtifactUploader")
    @patch("runner.run_training")
    @patch("runner.try_hf_login")
    @patch("runner.setup_logging")
    @patch("pathlib.Path.open", mock_open())
    def test_full_pipeline_success(
        self,
        mock_path_open,
        mock_setup_logging,
        mock_hf_login,
        mock_run_training,
        mock_uploader_cls,
        mock_archiver_cls,
        mock_asset_manager_cls,
        mock_reporter_cls,
        mock_load_config,
    ):
        mock_load_config.return_value = self.cfg
        mock_setup_logging.return_value = Path("test_outputs/logs/trainer.log")
        mock_run_training.return_value = {
            "status": "success",
            "job_name": self.cfg.job_name,
            "base_model": "/app",
            "base_model_id": "Qwen/Qwen2.5-7B-Instruct",
            "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
            "lora_dir": "test_outputs/lora/test-job",
            "merged_dir": "test_outputs/merged/test-job",
            "checkpoint_dir": "test_outputs/checkpoints/test-job",
            "metrics_path": "test_outputs/metrics/test-job.train_metrics.json",
            "history_path": "test_outputs/metrics/test-job.train_history.json",
            "train_summary_path": "test_outputs/metrics/test-job.train_summary.json",
            "summary": {"train_loss": 1.23},
        }

        mock_uploader = mock_uploader_cls.return_value
        mock_uploader.ensure_hf_ready.return_value = {}
        mock_uploader.upload_non_summary_artifacts.return_value = (
            {"logs": {"url": "http://example.com/upload/logs", "path": "trainer.log"}},
            {},
        )
        mock_uploader.upload_to_huggingface.return_value = {}
        mock_uploader.upload_hf_metadata.return_value = {}
        mock_uploader.upload_summary.return_value = {
            "summary": {"url": "http://example.com/upload/summary", "path": "job-result.json"}
        }

        with patch("sys.argv", ["runner.py", "--config", "http://example.com/config.json"]):
            runner.main()

        reporter = mock_reporter_cls.return_value
        reporter.report_status.assert_any_call(
            "started",
            message="Training pipeline started",
            stage="bootstrap",
            progress=0,
        )
        reporter.report_status.assert_any_call(
            "finished",
            message="Training pipeline finished successfully",
            stage="finished",
            progress=100,
        )
        reporter.report_final.assert_called_once()

        mock_asset_manager_cls.return_value.prepare_dataset.assert_called_once_with(self.cfg)
        mock_asset_manager_cls.return_value.prepare_evaluation_dataset.assert_called_once_with(self.cfg)

        mock_archiver = mock_archiver_cls.return_value
        mock_archiver.make_archive.assert_called_once()
        mock_archiver.upload_archive.assert_called_once()

        mock_uploader.upload_non_summary_artifacts.assert_called_once()
        mock_uploader.upload_summary.assert_called_once()

    @patch("runner.load_config")
    @patch("runner.Reporter")
    @patch("runner.AssetManager")
    @patch("runner.ArtifactUploader")
    @patch("runner.Archiver")
    @patch("runner.run_training")
    @patch("runner.setup_logging")
    @patch("pathlib.Path.open", mock_open())
    def test_pipeline_failure(
        self,
        mock_path_open,
        mock_setup_logging,
        mock_run_training,
        mock_archiver_cls,
        mock_uploader_cls,
        mock_asset_manager_cls,
        mock_reporter_cls,
        mock_load_config,
    ):
        mock_load_config.return_value = self.cfg
        mock_setup_logging.return_value = Path("test_outputs/logs/trainer.log")
        mock_run_training.side_effect = Exception("Training failed")

        with patch("sys.argv", ["runner.py", "--config", "http://example.com/config.json"]):
            with self.assertRaises(SystemExit) as cm:
                runner.main()

        self.assertEqual(cm.exception.code, 1)
        mock_reporter_cls.return_value.report_error.assert_called_once_with(
            "Training failed",
            logs=unittest.mock.ANY,
        )

    @patch("runner.load_config")
    @patch("runner.Reporter")
    @patch("runner.AssetManager")
    @patch("runner.ArtifactUploader")
    @patch("runner.Archiver")
    @patch("runner.run_training")
    @patch("runner.try_hf_login")
    @patch("runner.setup_logging")
    @patch("pathlib.Path.open", mock_open())
    def test_upload_failure_is_not_fatal(
        self,
        mock_path_open,
        mock_setup_logging,
        mock_hf_login,
        mock_run_training,
        mock_archiver_cls,
        mock_uploader_cls,
        mock_asset_manager_cls,
        mock_reporter_cls,
        mock_load_config,
    ):
        mock_load_config.return_value = self.cfg
        mock_setup_logging.return_value = Path("test_outputs/logs/trainer.log")
        mock_run_training.return_value = {
            "status": "success",
            "job_name": self.cfg.job_name,
            "base_model": "/app",
            "base_model_id": "Qwen/Qwen2.5-7B-Instruct",
            "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
            "lora_dir": "test_outputs/lora/test-job",
            "merged_dir": "test_outputs/merged/test-job",
            "checkpoint_dir": "test_outputs/checkpoints/test-job",
            "metrics_path": "test_outputs/metrics/test-job.train_metrics.json",
            "history_path": "test_outputs/metrics/test-job.train_history.json",
            "train_summary_path": "test_outputs/metrics/test-job.train_summary.json",
            "summary": {"train_loss": 1.23},
        }

        mock_uploader = mock_uploader_cls.return_value
        mock_uploader.ensure_hf_ready.return_value = {}
        mock_uploader.upload_non_summary_artifacts.return_value = (
            {"logs": {"url": "http://example.com/upload/logs", "path": "trainer.log"}},
            {"merged_archive": "SSL EOF"},
        )
        mock_uploader.upload_to_huggingface.return_value = {}
        mock_uploader.upload_hf_metadata.return_value = {}
        mock_uploader.upload_summary.return_value = {
            "summary": {"url": "http://example.com/upload/summary", "path": "job-result.json"}
        }

        with patch("sys.argv", ["runner.py", "--config", "http://example.com/config.json"]):
            runner.main()

        reporter = mock_reporter_cls.return_value
        reporter.report_final.assert_called_once()
        final_call = reporter.report_final.call_args
        final_payload = final_call.args[0]

        self.assertEqual(final_call.kwargs["status"], "finished")
        self.assertEqual(final_payload["status"], "success")
        self.assertIn("merged_archive", final_payload["upload_errors"])


if __name__ == "__main__":
    unittest.main()