"""
Training checkpoint manager for end-to-end pipeline traceability.

Tracks model states, hyperparameters, metrics, and preprocessing contracts
at each pipeline stage with SHA256 verification for full reproducibility.
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StageCheckpoint:
    """Record of a single pipeline stage execution."""

    stage_name: str
    status: str  # "pending", "running", "completed", "failed", "skipped"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    config_hash: Optional[str] = None
    artifact_paths: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


class TrainingCheckpointManager:
    """
    Manages checkpoints across all pipeline stages for training traceability.

    Provides:
    - Stage-level checkpoint save/load with JSON persistence
    - SHA256 hashing of configs and data contracts
    - Model artifact path tracking
    - Metric logging per stage
    - Resume-from-checkpoint capability
    """

    STAGES = [
        "preprocessing",
        "data_contract",
        "baseline_training",
        "foundation_training",
        "fusion_training",
        "hyperparameter_optimization",
        "cross_study_evaluation",
        "reporting",
    ]

    def __init__(self, checkpoint_dir: Path, run_id: Optional[str] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.checkpoint_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: Dict[str, StageCheckpoint] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing checkpoints from disk."""
        manifest = self.run_dir / "manifest.json"
        if manifest.exists():
            with open(manifest, "r") as f:
                data = json.load(f)
            for stage_name, stage_data in data.get("stages", {}).items():
                self.checkpoints[stage_name] = StageCheckpoint(**stage_data)
            logger.info(
                "Resumed run %s with %d completed stages",
                self.run_id,
                sum(1 for c in self.checkpoints.values() if c.status == "completed"),
            )

    def _save_manifest(self) -> None:
        """Persist checkpoint manifest to disk."""
        manifest = self.run_dir / "manifest.json"
        data = {
            "run_id": self.run_id,
            "pipeline": "MM Transcriptomics Risk Signature Pipeline",
            "created": datetime.now().isoformat(),
            "stages": {
                name: asdict(cp) for name, cp in self.checkpoints.items()
            },
        }
        with open(manifest, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def is_completed(self, stage_name: str) -> bool:
        """Check if a stage has already been completed."""
        cp = self.checkpoints.get(stage_name)
        return cp is not None and cp.status == "completed"

    def begin_stage(self, stage_name: str) -> StageCheckpoint:
        """Mark a stage as running and record start time."""
        cp = StageCheckpoint(
            stage_name=stage_name,
            status="running",
            start_time=datetime.now().isoformat(),
        )
        self.checkpoints[stage_name] = cp
        self._save_manifest()
        logger.info("STAGE [%s] started", stage_name)
        return cp

    def complete_stage(
        self,
        stage_name: str,
        metrics: Optional[Dict[str, Any]] = None,
        artifact_paths: Optional[List[str]] = None,
        config_hash: Optional[str] = None,
    ) -> None:
        """Mark a stage as completed with results."""
        cp = self.checkpoints.get(stage_name)
        if cp is None:
            cp = StageCheckpoint(stage_name=stage_name, status="completed")
            self.checkpoints[stage_name] = cp

        cp.status = "completed"
        cp.end_time = datetime.now().isoformat()
        if cp.start_time:
            start = datetime.fromisoformat(cp.start_time)
            end = datetime.fromisoformat(cp.end_time)
            cp.duration_seconds = (end - start).total_seconds()
        if metrics:
            cp.metrics = metrics
        if artifact_paths:
            cp.artifact_paths = artifact_paths
        if config_hash:
            cp.config_hash = config_hash

        self._save_manifest()
        logger.info(
            "STAGE [%s] completed (%.1fs)",
            stage_name,
            cp.duration_seconds or 0,
        )

    def fail_stage(self, stage_name: str, error: str) -> None:
        """Mark a stage as failed."""
        cp = self.checkpoints.get(stage_name)
        if cp is None:
            cp = StageCheckpoint(stage_name=stage_name, status="failed")
            self.checkpoints[stage_name] = cp

        cp.status = "failed"
        cp.end_time = datetime.now().isoformat()
        cp.error_message = error
        if cp.start_time:
            start = datetime.fromisoformat(cp.start_time)
            end = datetime.fromisoformat(cp.end_time)
            cp.duration_seconds = (end - start).total_seconds()

        self._save_manifest()
        logger.error("STAGE [%s] FAILED: %s", stage_name, error)

    def skip_stage(self, stage_name: str, reason: str = "already completed") -> None:
        """Mark a stage as skipped."""
        if stage_name not in self.checkpoints:
            self.checkpoints[stage_name] = StageCheckpoint(
                stage_name=stage_name, status="skipped"
            )
        self.checkpoints[stage_name].status = "skipped"
        self._save_manifest()
        logger.info("STAGE [%s] skipped: %s", stage_name, reason)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all stage statuses."""
        summary = {"run_id": self.run_id, "stages": {}}
        for stage in self.STAGES:
            cp = self.checkpoints.get(stage)
            if cp:
                summary["stages"][stage] = {
                    "status": cp.status,
                    "duration": cp.duration_seconds,
                    "metrics": cp.metrics,
                }
            else:
                summary["stages"][stage] = {"status": "pending"}
        return summary

    def print_summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "",
            "=" * 72,
            f"  PIPELINE RUN SUMMARY  (run_id: {self.run_id})",
            "=" * 72,
        ]
        status_icons = {
            "completed": "[OK]",
            "failed": "[FAIL]",
            "running": "[...]",
            "skipped": "[SKIP]",
            "pending": "[    ]",
        }
        total_time = 0.0
        for stage in self.STAGES:
            cp = self.checkpoints.get(stage)
            if cp:
                icon = status_icons.get(cp.status, "[?]")
                dur = f"{cp.duration_seconds:.1f}s" if cp.duration_seconds else "---"
                total_time += cp.duration_seconds or 0
                metric_str = ""
                if cp.metrics:
                    key_metrics = {
                        k: v
                        for k, v in cp.metrics.items()
                        if isinstance(v, (int, float)) and not np.isnan(v)
                    }
                    if key_metrics:
                        metric_str = "  " + ", ".join(
                            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in list(key_metrics.items())[:3]
                        )
                lines.append(f"  {icon:>6}  {stage:<35} {dur:>8}{metric_str}")
            else:
                lines.append(
                    f"  {'[    ]':>6}  {stage:<35} {'---':>8}"
                )

        lines.append("-" * 72)
        lines.append(f"  Total wall time: {total_time:.1f}s")
        lines.append("=" * 72)
        return "\n".join(lines)

    @staticmethod
    def hash_config(config_dict: Dict[str, Any]) -> str:
        """Compute SHA256 hash of a configuration dictionary."""
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
