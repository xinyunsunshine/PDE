"""CPU checks for the interactive demo's comparison and notebook cells."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pde.demo import find_microwave_task, run_episode


def test_notebook_cells_compile_and_preserve_recorded_outputs():
    notebook = json.loads(
        (
            Path(__file__).parents[1] / "notebooks/microwave_prompt_demo.ipynb"
        ).read_text()
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            if "recorded-output" in cell["metadata"].get("tags", []):
                assert cell["outputs"]
                assert cell["execution_count"] is not None
                assert all(o["output_type"] != "error" for o in cell["outputs"])
            else:
                assert cell["outputs"] == []
                assert cell["execution_count"] is None


def test_microwave_lookup_uses_language_not_hardcoded_index():
    suite = SimpleNamespace(
        get_num_tasks=lambda: 2,
        get_task=lambda i: SimpleNamespace(
            language=["open the drawer", "close the microwave"][i]
        ),
    )
    assert find_microwave_task(suite) == 1
    suite.get_num_tasks = lambda: 1
    with pytest.raises(ValueError, match="Expected one"):
        find_microwave_task(suite)


def test_comparison_resets_noise_and_keeps_original_goal(tmp_path, monkeypatch):
    # RLinf core imports require its optional runtime dependencies.
    pytest.importorskip("ray")
    from rlinf.envs import action_utils

    monkeypatch.setattr(
        action_utils, "prepare_actions", lambda **kw: kw["raw_chunk_actions"]
    )
    written = []

    class Writer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def append_data(self, frame):
            written.append(frame.copy())

    import imageio.v2 as imageio

    monkeypatch.setattr(imageio, "get_writer", lambda *a, **k: Writer())

    def observations():
        return {
            "main_images": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
            "task_descriptions": ["close the microwave"],
        }

    class Env:
        def reset(self):
            return observations(), {}

        def chunk_step(self, actions):
            return (
                [observations()] * 10,
                None,
                None,
                None,
                [{"episode": {"success_once": [True]}}],
            )

    noise, prompts = [], []

    class Model:
        def predict_action_batch(self, env_obs, mode):
            assert mode == "eval"
            assert not torch.is_grad_enabled()
            prompts.append(env_obs["task_descriptions"][0])
            noise.append(torch.randn(3))
            return torch.zeros(1, 10, 7), {}

    cfg = SimpleNamespace(num_action_chunks=10, action_dim=7, get=lambda key: None)
    first = run_episode(
        Env(), Model(), cfg, "close the microwave", 7, 20, tmp_path / "first.mp4"
    )
    second = run_episode(
        Env(), Model(), cfg, "push the door", 7, 20, tmp_path / "second.mp4"
    )
    assert first["initial_image_sha256"] == second["initial_image_sha256"]
    assert (
        first["canonical_prompt"] == second["canonical_prompt"] == "close the microwave"
    )
    assert first["success"] and second["success"]
    assert prompts == ["close the microwave"] * 2 + ["push the door"] * 2
    assert torch.equal(noise[0], noise[2]) and torch.equal(noise[1], noise[3])
    assert len(written) == 42
    assert np.load(tmp_path / "first.npz")["frames"].shape == (3, 8, 8, 3)


def test_feedback_uses_observed_frames_and_displays_response(
    tmp_path, monkeypatch, capsys
):
    pytest.importorskip("openai")
    from pde.demo import feedback

    np.savez_compressed(
        tmp_path / "reference.npz", frames=np.zeros((2, 8, 8, 3), dtype=np.uint8)
    )
    (tmp_path / "comparison.json").write_text(
        json.dumps(
            {
                "checkpoint": "frozen",
                "results": [
                    {
                        "frames": "reference.npz",
                        "canonical_prompt": "close the microwave",
                        "prompt": "close the microwave",
                        "success": False,
                    }
                ],
            }
        )
    )
    closed = []

    class Supervisor:
        provider = "openai"
        model = "test-model"
        client = SimpleNamespace(close=lambda: closed.append(True))

        def __init__(self, **kwargs):
            pass

        def summarize(self, goal, prompt, episodes):
            assert goal == prompt == "close the microwave"
            assert episodes[0].shape == (2, 8, 8, 3)
            return "The robot missed the door."

        def __call__(self, pool, candidates):
            assert pool.metadata["canonical_evaluation"]["success_rate"] == 0
            assert candidates == 3
            return ["push the door shut"]

    monkeypatch.setattr("pde.vlm.VLMSupervisor", Supervisor)
    feedback(
        SimpleNamespace(
            output=str(tmp_path), model="test-model", provider="openai", base_url=None
        )
    )
    result = json.loads((tmp_path / "vlm_feedback.json").read_text())
    assert result["new_prompts"] == ["push the door shut"]
    assert "The robot missed the door." in capsys.readouterr().out
    assert closed == [True]


def test_each_prompt_gets_a_fresh_simulator_but_the_same_policy(tmp_path, monkeypatch):
    from pde.demo import compare_prompts

    created, closed, policies = [], [], []
    policy = object()

    def factory():
        env = SimpleNamespace(env=SimpleNamespace(close=lambda: closed.append(True)))
        created.append(env)
        return env

    def episode(env, model, *args):
        policies.append(model)
        assert not hasattr(env, "used"), "Simulator was reused between prompts"
        env.used = True
        return {"initial_image_sha256": "identical"}

    monkeypatch.setattr("pde.demo.run_episode", episode)
    compare_prompts(factory, policy, None, "original", "rewrite", 0, 240, tmp_path)
    assert len(created) == len(closed) == 2
    assert policies == [policy, policy]


def test_simulator_closed_if_rollout_fails(tmp_path, monkeypatch):
    from pde.demo import compare_prompts

    closed = []
    env = SimpleNamespace(env=SimpleNamespace(close=lambda: closed.append(True)))

    def fail(*args):
        raise RuntimeError("rendering failed")

    monkeypatch.setattr("pde.demo.run_episode", fail)
    with pytest.raises(RuntimeError, match="rendering failed"):
        compare_prompts(
            lambda: env, None, None, "original", "rewrite", 0, 240, tmp_path
        )
    assert closed == [True]
