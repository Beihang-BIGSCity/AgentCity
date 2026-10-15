"""Runs the adaptation pipeline, including the closed self-correction loop."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.adaptation.catalog import AdaptationCatalog
from agents.adaptation.context import ExecutionContext
from agents.adaptation.runtime import clear_active_task, set_active_task
from agents.adaptation.workflow import get_adaptation_workflow

from .config import AppPaths, build_context, load_task_spec
from .openjiuwen_backend import HarnessProfile, ensure_runner_started, shutdown_runner
from .pipeline import AgentPipeline
from .storage import StageStorage

ROUND_ARTIFACTS = (
    "dataset_profile.json",
    "plan.json",
    "data_integration.json",
    "composition.json",
    "run_report.json",
    "tuning.json",
    "verdict.json",
)

# Artifacts that describe the *dataset* rather than the round, so they survive
# across rounds and do not have to be recomputed by the next data-integration pass.
ROUND_CARRYOVER = ("dataset_profile.json",)


class AgentOrchestrator:
    """Coordinates adaptation runs for the CLI entry point."""

    def __init__(self, *, paths: AppPaths, logger) -> None:
        self.paths = paths
        self.logger = logger
        self._lock = asyncio.Lock()
        self._profile = HarnessProfile.load()
        self._last_notes: Dict[str, str] = {}

    async def run_task_file(self, spec_path: Path) -> Dict[str, Any]:
        """Load a task specification from disk and adapt it."""

        return await self.run_adaptation(load_task_spec(spec_path))

    async def run_adaptation(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt one scenario, looping until the target is met or rounds run out."""

        task_id = str(task_spec.get("task_id") or "task")
        max_rounds = int(task_spec.get("max_rounds") or 1)
        store = ExecutionContext.load(task_id, base_dir=self.paths.adaptation_dir)
        storage = StageStorage(self.paths.stage_log)
        storage.reset()
        catalog = AdaptationCatalog(
            index_path=self.paths.adaptation_index,
            documentation_dir=self.paths.documentation_dir,
            root=self.paths.root,
        )
        result: Dict[str, Any] = {
            "task_id": task_id,
            "max_rounds": max_rounds,
            "rounds": [],
            "passed": False,
            "verdict": None,
        }

        async with self._lock:
            set_active_task(task_spec)
            await ensure_runner_started()
            try:
                for round_index in range(1, max_rounds + 1):
                    self._archive_previous_round(store, round_index)
                    context = build_context(
                        self.paths,
                        task_spec,
                        round_index=round_index,
                        max_rounds=max_rounds,
                        feedback=store.feedback_note(),
                        dataset_profile=store.get("dataset_profile") or {},
                    )
                    workflow = get_adaptation_workflow()
                    workflow.name = f"adaptation#r{round_index}"
                    self.logger.info("=== Adaptation round %s/%s ===", round_index, max_rounds)
                    pipeline = AgentPipeline(
                        profile=self._profile,
                        workflows=[workflow],
                        storage=storage,
                        logger=self.logger,
                        context=context,
                    )
                    await pipeline.run()
                    self._last_notes = dict(context.stage_notes)

                    payload = self._collect_round(store, round_index, context)
                    store.record_round(payload)
                    verdict = payload.get("verdict") or {}
                    result["rounds"].append({"round": round_index, "verdict": verdict or None})
                    if verdict:
                        result["verdict"] = verdict
                    if verdict.get("passed"):
                        result["passed"] = True
                        self.logger.info("Adaptation target met on round %s", round_index)
                        break
                    if round_index < max_rounds:
                        self.logger.info(
                            "Round %s did not meet the target; feeding back into planning", round_index
                        )
            finally:
                await shutdown_runner()
                clear_active_task()

        summary_path = self._write_summary(catalog, task_spec, result, store)
        result["summary_path"] = summary_path
        result["context_path"] = str(store.path)
        result["artifact_dir"] = str(store.dir)
        return result

    @staticmethod
    def _archive_previous_round(store: ExecutionContext, round_index: int) -> None:
        """Move stale artifacts aside before a round starts.

        Without this the collector cannot tell "this round produced nothing"
        from "the previous round's file is still lying around", and a failed
        round would silently inherit the previous verdict. The artifact
        directory is also reused between CLI invocations, so round 1 of a new
        run stashes whatever an earlier session left behind (observed: a
        planner read the previous session's run_report.json).
        """

        stale = [
            name
            for name in ROUND_ARTIFACTS
            if name not in ROUND_CARRYOVER and (store.dir / name).exists()
        ]
        if not stale:
            return
        bucket = f"round_{round_index - 1}" if round_index > 1 else "previous_run"
        archive_dir = store.dir / "rounds" / bucket
        archive_dir.mkdir(parents=True, exist_ok=True)
        for name in stale:
            path = store.dir / name
            try:
                path.replace(archive_dir / name)
            except OSError:
                path.unlink(missing_ok=True)

    def _collect_round(
        self, store: ExecutionContext, round_index: int, context
    ) -> Dict[str, Any]:
        """Read the round artifacts and persist them into the shared context."""

        artifacts = {name: self._read_json(store.dir / name) for name in ROUND_ARTIFACTS}
        payload: Dict[str, Any] = {
            "round": round_index,
            "dataset_profile": artifacts["dataset_profile.json"],
            "plan": artifacts["plan.json"],
            "data_integration": artifacts["data_integration.json"],
            "composition": artifacts["composition.json"],
            "run_report": artifacts["run_report.json"],
            "runs": [item for item in [artifacts["run_report.json"]] if item],
            "tuning": artifacts["tuning.json"],
            "verdict": artifacts["verdict.json"],
            "artifacts": {name: str(store.dir / name) for name in ROUND_ARTIFACTS},
        }
        recorded_runs = self._read_json(store.dir / "runs.json")
        if recorded_runs:
            payload["recorded_runs"] = recorded_runs
        if not payload["verdict"]:
            failure = (
                context.stage_notes.get("tuning")
                or context.stage_notes.get("execution")
                or "no verdict was produced in this round"
            )
            payload["failure"] = str(failure)[:4000]

        for section in (
            "dataset_profile",
            "plan",
            "data_integration",
            "composition",
            "tuning",
            "verdict",
        ):
            if payload[section]:
                store.update(section, payload[section])
        return payload

    def _write_summary(
        self,
        catalog: AdaptationCatalog,
        task_spec: Dict[str, Any],
        result: Dict[str, Any],
        store: ExecutionContext,
    ) -> Optional[str]:
        """Render the adaptation summary document for this task."""

        task_id = str(task_spec.get("task_id"))
        target = task_spec.get("target") or {}
        train = task_spec.get("train") or {}
        model_name = task_spec.get("model_name") or (store.get("composition") or {}).get("model_name")
        lines: List[str] = [
            f"# {task_id} 模型适配总结",
            "",
            "## 任务规格",
            "",
            f"- 业务场景数据集: {task_spec.get('dataset')}",
            f"- 预测任务类型: {task_spec.get('task')}",
            f"- 适配目标: {target.get('metric')} {'≥' if target.get('maximize') else '≤'} {target.get('threshold')}",
            f"- 效率偏好: {task_spec.get('efficiency')}",
            f"- 训练预算: {train.get('max_epoch')} epochs / batch {train.get('batch_size')} / GPU {train.get('gpu')}",
            f"- 适配模型: {model_name or '未产出'}",
            f"- 迭代轮次: {len(result.get('rounds') or [])} / {result.get('max_rounds')}",
            f"- 最终结论: {'达标' if result.get('passed') else '未达标'}",
            "",
        ]

        profile = store.get("dataset_profile") or {}
        if profile:
            lines.extend(
                [
                    "## 数据特征画像",
                    "",
                    f"- CVglobal: {profile.get('cv_global')}",
                    f"- CVnode: {profile.get('cv_node')}",
                    f"- 空间异质性 SH: {profile.get('spatial_heterogeneity')}",
                    f"- 一阶自相关 rho1: {profile.get('rho1')}",
                    f"- 采样间隔(分钟): {profile.get('sampling_interval_min')}",
                    f"- 节点数 / 时间步: {profile.get('node_count')} / {profile.get('timesteps')}",
                    "",
                ]
            )

        composition = store.get("composition") or {}
        if composition:
            lines.extend(
                [
                    "## 模块组合方案",
                    "",
                    f"- hidden_dim: {composition.get('hidden_dim')}, depth: {composition.get('depth')}, loss: {composition.get('loss_type')}",
                ]
            )
            for category, module in (composition.get("modules") or {}).items():
                identifier = module.get("id") if isinstance(module, dict) else module
                lines.append(f"- {category}: {identifier}")
            lines.append("")

        lines.extend(["## 迭代轨迹", ""])
        for round_payload in result.get("rounds") or []:
            verdict = round_payload.get("verdict") or {}
            lines.append(
                f"- 第 {round_payload.get('round')} 轮: {verdict.get('metric')}="
                f"{verdict.get('value')} (阈值 {verdict.get('threshold')}) -> "
                f"{'达标' if verdict.get('passed') else '未达标'}"
            )
            if verdict.get("reason"):
                lines.append(f"  - 判定理由: {str(verdict['reason'])[:500]}")
            if verdict.get("next_direction"):
                lines.append(f"  - 下一步方向: {str(verdict['next_direction'])[:500]}")
        history = store.get("history") or []
        if history:
            last = history[-1]
            if last.get("failure"):
                lines.append(f"- 末轮失败原因: {str(last['failure'])[:800]}")
            # Prefer the round's own run_report.json. runs.json is an append-only
            # registry shared with earlier sessions, so its last entry can be a
            # stale record (observed: the summary reported another model's failed
            # run while this round succeeded).
            last_run = last.get("run_report") or {}
            if not last_run:
                recorded = last.get("recorded_runs") or []
                last_run = recorded[-1] if recorded else {}
            if last_run:
                detail = [
                    f"status={last_run.get('status')}",
                    f"metrics={json.dumps(last_run.get('metrics') or {}, ensure_ascii=False)}",
                ]
                if last_run.get("exp_id"):
                    detail.insert(1, f"exp_id={last_run['exp_id']}")
                if last_run.get("metrics_source"):
                    detail.insert(-1, f"metrics_source={last_run['metrics_source']}")
                lines.append(f"- 末次运行: {' '.join(detail)}")
        lines.append("")

        lines.extend(["## 阶段摘要", ""])
        for key, note in (self._last_notes or {}).items():
            lines.append(f"### {key}")
            lines.append("")
            lines.append(str(note)[:4000])
            lines.append("")

        lines.extend(
            [
                "## 产物",
                "",
                f"- 共享执行上下文: {store.path}",
                f"- 运行记录: {store.dir / 'runs.json'}",
                f"- 组合方案: {store.dir / 'composition.json'}",
                f"- 评估结论: {store.dir / 'verdict.json'}",
            ]
        )
        composition_model = store.get("composition") or {}
        if composition_model.get("model_name"):
            lines.append(
                f"- 适配模型代码: Bigscity-LibCity/libcity/model/traffic_speed_prediction/"
                f"{composition_model['model_name']}.py"
            )

        entry = catalog.record_summary(
            task_id,
            "\n".join(lines),
            task_spec=task_spec,
            verdict=result.get("verdict"),
        )
        return (entry or {}).get("summary_file")

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None


__all__ = ["AgentOrchestrator"]
