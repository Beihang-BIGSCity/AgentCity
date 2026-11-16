"""Regression checks for task-controlled compute policy and LibCity paths."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.core.config import AppPaths, build_context
from agents.adaptation.prompts.data_integrator import build_data_integrator_prompt
from agents.adaptation.prompts.effect_evaluator import build_effect_evaluator_prompt
from agents.adaptation.prompts.hyper_searcher import build_hyper_searcher_prompt
from agents.adaptation.prompts.lead import build_execution_lead_prompt, build_tuning_lead_prompt
from agents.adaptation.prompts.model_composer import build_model_composer_prompt
from agents.adaptation.prompts.perf_debugger import build_perf_debugger_prompt
from agents.adaptation.prompts.run_tester import build_run_tester_prompt
from agents.adaptation.prompts.task_planner import build_task_planning_prompt


BUILDERS = (
    build_task_planning_prompt, build_execution_lead_prompt, build_tuning_lead_prompt,
    build_data_integrator_prompt, build_model_composer_prompt, build_run_tester_prompt,
    build_hyper_searcher_prompt, build_perf_debugger_prompt, build_effect_evaluator_prompt,
)


class AdaptationComputePromptsTest(unittest.TestCase):
    def context(self, gpu=None):
        train = {"max_epoch": 15, "batch_size": 128, "extra_args": {"train_rate": 0.6}}
        if gpu is not None:
            train["gpu"] = gpu
        return build_context(AppPaths(), {
            "task_id": "prompt_policy_probe", "dataset": "PEMS_BAY",
            "task": "traffic_state_pred", "train": train,
        })

    def test_every_role_honors_gpu_budget(self):
        context = self.context(True)
        for builder in BUILDERS:
            with self.subTest(role=builder.__name__):
                prompt = builder(context)
                self.assertIn("gpu=true", prompt)
                for forbidden in ("gpu=false", "--gpu false", "CPU-only", "never request a GPU", "never request CUDA"):
                    self.assertNotIn(forbidden, prompt)
                self.assertIn("epoch", prompt)
                self.assertIn("budget", prompt)

    def test_every_role_defaults_to_cpu(self):
        for setting in (None, False):
            context = self.context(setting)
            for builder in BUILDERS:
                with self.subTest(gpu=setting, role=builder.__name__):
                    prompt = builder(context)
                    self.assertIn("gpu=false", prompt)
                    self.assertNotIn("gpu=true", prompt)
                    self.assertIn("epoch", prompt)
                    self.assertIn("budget", prompt)

    def test_paths_are_absolute_without_duplicate_checkout(self):
        context = self.context(True)
        for builder in BUILDERS:
            with self.subTest(role=builder.__name__):
                prompt = builder(context)
                self.assertIn(context.repo_path, prompt)
                self.assertIn(context.task_dir, prompt)
                self.assertNotIn("Bigscity-LibCity/Bigscity-LibCity", prompt)
                for token in ("__REPO__", "__TASK_DIR__", "__GPU_FLAG__", "__COMPUTE_POLICY__"):
                    self.assertNotIn(token, prompt)
        composer = build_model_composer_prompt(context)
        self.assertIn(f"{context.repo_path}/libcity/model/traffic_speed_prediction/<ModelName>.py", composer)
        self.assertIn(f"{context.repo_path}/libcity/config/task_config.json", composer)

    def test_composer_requires_package_export_and_registration(self):
        context = self.context(True)
        prompt = build_model_composer_prompt(context)
        self.assertIn(f"{context.repo_path}/libcity/model/traffic_speed_prediction/__init__.py", prompt)
        self.assertIn("from .<ModelName> import <ModelName>", prompt)
        self.assertIn("TrafficStatePointDataset", prompt)
        self.assertIn("allowed_model", prompt)


if __name__ == "__main__":
    unittest.main()
