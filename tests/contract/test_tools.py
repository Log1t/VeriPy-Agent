from veripy_agent.environment.paths import SandboxPaths
from veripy_agent.tools.basic import inspect_files, inspect_table


def test_inputs_are_read_only_and_scratch_is_writable(tmp_path):
    (tmp_path / "inputs").mkdir()
    paths = SandboxPaths(tmp_path / "inputs", tmp_path / "scratch")
    (paths.inputs / "x.csv").write_text("a,b\n1,2\n")
    (paths.inputs / "x.json").write_text('[{"a":1}]')
    assert inspect_files(paths).status.value == "ok"
    assert inspect_table(paths, "x.csv").status.value == "ok"
    assert inspect_table(paths, "x.json").status.value == "ok"


def test_path_escape_fails_closed(tmp_path):
    (tmp_path / "inputs").mkdir()
    paths = SandboxPaths(tmp_path / "inputs", tmp_path / "scratch")
    assert inspect_table(paths, "../scratch/x").status.value == "error"
