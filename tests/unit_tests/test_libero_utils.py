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

import importlib
import sys
from types import ModuleType


def _load_libero_utils_module(monkeypatch):
    benchmark_module = ModuleType("libero.libero.benchmark")
    benchmark_module.BENCHMARK_MAPPING = {}
    benchmark_module.libero_suites = []
    benchmark_module.task_maps = {}

    class FakeBenchmarkBase:
        def __init__(self, task_order_index=0):
            self.task_order_index = task_order_index

    class FakeTask:
        def __init__(
            self,
            name: str,
            language: str,
            problem: str,
            problem_folder: str,
            bddl_file: str,
            init_states_file: str,
        ):
            self.name = name
            self.language = language
            self.problem = problem
            self.problem_folder = problem_folder
            self.bddl_file = bddl_file
            self.init_states_file = init_states_file

    calls = {"get_benchmark": []}

    def fake_get_benchmark(name):
        calls["get_benchmark"].append(name)
        return f"fallback::{name}"

    def fake_grab_language_from_filename(filename):
        return f"lang::{filename}"

    benchmark_module.Benchmark = FakeBenchmarkBase
    benchmark_module.Task = FakeTask
    benchmark_module.get_benchmark = fake_get_benchmark
    benchmark_module.grab_language_from_filename = fake_grab_language_from_filename

    libero_pkg = ModuleType("libero")
    libero_subpkg = ModuleType("libero.libero")
    libero_pkg.libero = libero_subpkg
    libero_subpkg.benchmark = benchmark_module

    monkeypatch.setitem(sys.modules, "libero", libero_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_subpkg)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", benchmark_module)

    monkeypatch.setenv("LIBERO_TYPE", "standard")
    monkeypatch.delitem(sys.modules, "rlinf.envs.libero.utils", raising=False)

    utils_module = importlib.import_module("rlinf.envs.libero.utils")
    return utils_module, benchmark_module, calls


def test_get_benchmark_overridden_supports_libero_90_boost(monkeypatch):
    utils_module, benchmark_module, calls = _load_libero_utils_module(monkeypatch)

    benchmark_cls = utils_module.get_benchmark_overridden("libero_90_boost")
    benchmark_instance = benchmark_cls()

    assert benchmark_instance.name == "libero_90_boost"
    assert benchmark_instance.n_tasks == 2
    assert [task.name for task in benchmark_instance.tasks] == [
        "KITCHEN_SCENE6_close_the_microwave",
        "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
    ]
    assert benchmark_module.BENCHMARK_MAPPING["libero_90_boost"] is benchmark_cls
    assert calls["get_benchmark"] == []


def test_get_benchmark_overridden_delegates_non_special_suite(monkeypatch):
    utils_module, _, calls = _load_libero_utils_module(monkeypatch)

    result = utils_module.get_benchmark_overridden("libero_10")

    assert result == "fallback::libero_10"
    assert calls["get_benchmark"] == ["libero_10"]
