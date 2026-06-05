# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
import types
from pathlib import Path

from omegaconf import OmegaConf

from rlinf.utils.metric_logger import MetricLogger


class _FakeWandbModule:
    """Lightweight WandB stub for unit tests."""

    class Settings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Table:
        def __init__(self, dataframe=None):
            self.dataframe = dataframe

    def __init__(self, run_id: str = "wandb-run-1"):
        self.run_id = run_id
        self.init_calls: list[dict] = []
        self.log_calls: list[dict] = []
        self.define_metric_calls: list[tuple[tuple, dict]] = []
        self.run = types.SimpleNamespace(id=run_id, summary={})

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        run_id = kwargs.get("id", self.run_id)
        if "resume_from" in kwargs:
            run_id = kwargs["resume_from"].split("?", 1)[0]
        self.run = types.SimpleNamespace(id=run_id, summary={})
        return self.run

    def define_metric(self, *args, **kwargs):
        self.define_metric_calls.append((args, kwargs))

    def log(self, data, step=None, commit=None):
        self.log_calls.append({"data": data, "step": step, "commit": commit})

    def finish(self):
        return None


def _make_cfg(log_path: Path):
    return OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "log_path": str(log_path),
                    "project_name": "proj",
                    "experiment_name": "exp",
                    "logger_backends": ["wandb"],
                },
                "resume_dir": None,
            }
        }
    )


def test_metric_logger_reuses_wandb_run_and_logs_global_step(tmp_path, monkeypatch):
    fake_wandb = _FakeWandbModule(run_id="fresh-run")
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    cfg = _make_cfg(tmp_path)
    logger = MetricLogger(cfg)

    assert fake_wandb.init_calls[0]["project"] == "proj"
    assert fake_wandb.init_calls[0]["name"] == "exp"
    assert "id" not in fake_wandb.init_calls[0]
    assert fake_wandb.init_calls[0].get("resume") is None
    assert ("global_step",) in [args for args, _ in fake_wandb.define_metric_calls]
    assert ("*",) in [args for args, _ in fake_wandb.define_metric_calls]

    logger.log({"train/loss": 1.5}, step=42)
    logger.flush(42)
    logger.finish()

    assert fake_wandb.log_calls[0] == {
        "data": {"train/loss": 1.5, "global_step": 42},
        "step": None,
        "commit": False,
    }
    assert fake_wandb.log_calls[1] == {
        "data": {"global_step": 42},
        "step": None,
        "commit": True,
    }
    assert fake_wandb.run.summary["latest_global_step"] == 42


def test_metric_logger_ignores_saved_wandb_run_id_on_resume(tmp_path, monkeypatch):
    fake_wandb = _FakeWandbModule(run_id="new-run")
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    tmp_path.joinpath("wandb_run_id.txt").write_text("previous-run")
    cfg = _make_cfg(tmp_path)
    cfg.runner.resume_dir = f"{tmp_path}/checkpoints/global_step_7"

    logger = MetricLogger(cfg)

    assert "id" not in fake_wandb.init_calls[0]
    assert "resume" not in fake_wandb.init_calls[0]

    logger.log({"eval/score": 9.0}, step=7)
    logger.finish()

    assert fake_wandb.log_calls[0]["data"]["global_step"] == 7
    assert fake_wandb.run.summary["latest_global_step"] == 7
