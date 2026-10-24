import csv
import datetime as dt
import json
import os
import pathlib

import pytest

from llm_tel import persistence


@pytest.fixture
def workspace(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    class FrozenDateTime(dt.datetime):
        @classmethod
        def now(cls, tz: dt.tzinfo | None = None) -> dt.datetime:
            return cls(2026, 1, 1, tzinfo=tz)

    monkeypatch.setattr(persistence.dt, "datetime", FrozenDateTime)


@pytest.mark.parametrize("seed", [
    "../escape", "../../escape", "../../../escape", "a/../../escape",
    "/absolute", "//absolute", "///absolute", "/", "..", ".", "../..",
    r"..\escape", r"C:\absolute\seed", r"\\server\share", "a/b/c",
    "a / b", "a/../b", "a//b", "a\x00b", "\x00", "a\nb\tc",
    "a\u2215b", "a\u2044b", "a\uff0fb", "a\uff3cb", "\uff0e\uff0e/seed",
    "\u202eseed", "\u200b", "\u0301", "caf\u00e9", "\u4e2d\u6587", "\U0001f600",
    pytest.param("x" * 1000, id="long-seed"),
])
def test_hostile_seeds_stay_inside_experiments(
    seed: str, workspace: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = workspace / "experiments"
    original_mkdir = os.mkdir

    def guarded_mkdir(
        path: str | os.PathLike[str], mode: int = 0o777, *, dir_fd: int | None = None,
    ) -> None:
        target = pathlib.Path(path).resolve()
        # Keep the unfixed implementation's traversal inside the test sandbox.
        assert target == root or root in target.parents, (
            "allocation escaped experiments"
        )
        return original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", guarded_mkdir)
    try:
        allocated = persistence.mk_exp_dir(seed)
    except ValueError:
        assert list(workspace.iterdir()) == []
    else:
        assert isinstance(allocated, pathlib.Path)
        assert root in allocated.resolve().parents
        assert allocated.is_dir()


def test_same_seed_twice_is_distinct(
    workspace: pathlib.Path, frozen_clock: None,
) -> None:
    first = persistence.mk_exp_dir("same seed")
    second = persistence.mk_exp_dir("same seed")
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_existing_run_contents_survive(
    workspace: pathlib.Path, frozen_clock: None,
) -> None:
    first = persistence.mk_exp_dir("same seed")
    (first / "sentinel").write_bytes(b"preserve sentinel")
    (first / "chain.csv").write_bytes(b"preserve csv")
    (first / "chain.jsonl").write_bytes(b"preserve jsonl")
    before = {path.name: path.read_bytes() for path in first.iterdir()}

    second = persistence.mk_exp_dir("same seed")
    with persistence.Recorder(second) as recorder:
        recorder.log(cycle=1, phase="answer", model="test", text="new", score=3)

    assert {path.name: path.read_bytes() for path in first.iterdir()} == before
    assert second != first


@pytest.mark.parametrize(
    "seed", ["", " ", "\t\r\n", "\u00a0", "...", "///", "\x00", "\u200b"],
)
def test_empty_seed_rejected_before_writes(
    seed: str, workspace: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_mkdir(*args, **kwargs):
        pytest.fail("empty seed attempted a filesystem write")

    monkeypatch.setattr(os, "mkdir", forbid_mkdir)
    with pytest.raises(ValueError):
        persistence.mk_exp_dir(seed)
    assert list(workspace.iterdir()) == []


def test_recorder_round_trip_inside_allocated_directory(
    workspace: pathlib.Path,
) -> None:
    allocated = persistence.mk_exp_dir("ordinary seed")
    row = dict(cycle=2, phase="answer", model="test", text='a, "quote"\nline', score=4)
    with persistence.Recorder(allocated) as recorder:
        recorder.log(**row)

    assert (workspace / "experiments") in allocated.resolve().parents
    assert {path.name for path in allocated.iterdir()} == {"chain.csv", "chain.jsonl"}
    with (allocated / "chain.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["cycle", "phase", "model", "text", "score"]
        assert list(reader) == [{key: str(value) for key, value in row.items()}]
    with (allocated / "chain.jsonl").open() as handle:
        assert [json.loads(line) for line in handle] == [row]
