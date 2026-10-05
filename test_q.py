#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""Unit tests for qsched. Run: ./test_q.py  (or via pytest)"""

import argparse
import fcntl
import io
import random
import shlex
from datetime import datetime
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import q
from q import (
    Scheduler,
    _parse_hms,
    expand_sweep,
    run_probe,
    run_probes,
    split_top_level,
    tqdm_progress,
    validated_combos,
)

VALID_CMD = f"exit {q.VALIDATE_OK_CODE}"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("QSCHED_HOME", str(tmp_path))
    monkeypatch.setenv("QSCHED_NOTIFY", str(tmp_path / "absent.sh"))

    def record(title: str, body: str, kind: str = "failure") -> None:
        NOTIFICATIONS.append(title)
        KINDS.append(kind)
        BODIES.append(body)

    monkeypatch.setattr(q, "notify", record)
    NOTIFICATIONS.clear()
    KINDS.clear()
    BODIES.clear()
    yield tmp_path


NOTIFICATIONS: list[str] = []
KINDS: list[str] = []
BODIES: list[str] = []


def enqueue(conn: sqlite3.Connection, *argv: str) -> int:
    return q.insert_jobs(conn, [list(argv) or ["true"]], {})[0]


def scheduler(gpus: list[int] | None = None) -> Scheduler:
    return Scheduler(conn=q.db(), gpus=gpus if gpus is not None else [0])


def state_of(conn: sqlite3.Connection, job_id: int) -> str:
    row = conn.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
    return str(row["state"])


def run_to_completion(sched: Scheduler, timeout: float = 5.0) -> None:
    """Dispatch, then reap until nothing is running (real short-lived processes)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        now = time.time()
        sched.reap()
        sched.dispatch(now)
        sched.check_drain()
        if not sched.running and (
            sched.paused(now)
            or not sched.conn.execute(
                "SELECT 1 FROM jobs WHERE state='queued' LIMIT 1"
            ).fetchone()
        ):
            return
        time.sleep(0.02)
    raise AssertionError("jobs did not finish in time")


def test_quoted_value_is_single_variant() -> None:
    assert split_top_level('"96"') == ['"96"']


def test_plain_comma_sweep() -> None:
    assert split_top_level("0.1,0.01") == ["0.1", "0.01"]


def test_list_of_quoted_strings_vs_sweep_of_lists() -> None:
    assert split_top_level('["Day","key"]') == ['["Day","key"]']
    assert split_top_level('["Day","key"],["Day"]') == ['["Day","key"]', '["Day"]']


def test_quoted_comma_not_split() -> None:
    assert split_top_level('"a,b"') == ['"a,b"']
    assert split_top_level("'a,b',c") == ["'a,b'", "c"]


def test_nested_brackets() -> None:
    assert split_top_level("[[a,b],[c]],[[d]]") == ["[[a,b],[c]]", "[[d]]"]


def test_braces_and_parens() -> None:
    assert split_top_level("{a:1,b:2},{a:3}") == ["{a:1,b:2}", "{a:3}"]
    assert split_top_level("f(x,y),g(z)") == ["f(x,y)", "g(z)"]


def test_expand_cartesian_product() -> None:
    argv = ["python", "train.py", "model=a,b", "lr=0.1,0.01", "seed=1"]
    combos = expand_sweep(argv)
    assert len(combos) == 4
    assert ["python", "train.py", "model=a", "lr=0.1", "seed=1"] in combos
    assert ["python", "train.py", "model=b", "lr=0.01", "seed=1"] in combos


def test_expand_user_examples() -> None:
    argv = [
        "uv",
        "run",
        "python",
        "scripts/train_predict.py",
        'features_transform.grouping=["Day","key"],["Day"]',
        'training_window="96"',
    ]
    combos = expand_sweep(argv)
    assert len(combos) == 2
    assert combos[0][-2] == 'features_transform.grouping=["Day","key"]'
    assert combos[1][-2] == 'features_transform.grouping=["Day"]'
    assert all(c[-1] == 'training_window="96"' for c in combos)


def test_flags_and_non_overrides_untouched() -> None:
    argv = ["prog", "--config-name=a,b", "plainword", "+key=x,y"]
    combos = expand_sweep(argv)
    assert len(combos) == 2  # only +key expands
    assert all(c[1] == "--config-name=a,b" for c in combos)


def test_empty_variant_raises() -> None:
    with pytest.raises(ValueError, match="empty sweep variant"):
        expand_sweep(["key=a,,b"])
    with pytest.raises(ValueError, match="empty sweep variant"):
        expand_sweep(["key=a,"])


def test_no_expansion_is_single_job() -> None:
    assert expand_sweep(["python", "train.py", "model=a"]) == [
        ["python", "train.py", "model=a"]
    ]


BAR = "predict:  83%|████████▎ | 38/46 [3:13:17<58:20, 437.56s/it]"


def test_parse_hms() -> None:
    assert _parse_hms("58:20") == 3500.0
    assert _parse_hms("3:13:17") == 11597.0
    assert _parse_hms("07") == 7.0


def _log(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "1.log"
    path.write_text(text, encoding="utf-8")
    return path


def test_tqdm_progress_reads_last_bar(tmp_path: Path) -> None:
    text = "== header\n" + "\r".join(
        ["fit:   5%|▌ | 1/20 [00:10<03:10, 10.0s/it]", BAR]
    )
    assert tqdm_progress(_log(tmp_path, text)) == (83, 3500.0)


def test_tqdm_progress_ignores_unknown_eta(tmp_path: Path) -> None:
    unknown = "predict:   0%|  | 0/46 [00:00<?, ?it/s]"
    assert tqdm_progress(_log(tmp_path, f"{BAR}\r{unknown}")) == (83, 3500.0)


def test_tqdm_progress_without_bar(tmp_path: Path) -> None:
    assert tqdm_progress(_log(tmp_path, "epoch 3 loss 0.1\n")) is None
    assert tqdm_progress(tmp_path / "missing.log") is None


def test_tqdm_progress_ignores_totalless_bar(tmp_path: Path) -> None:
    assert tqdm_progress(_log(tmp_path, "38it [03:13, 5.09s/it]\n")) is None


def test_tqdm_progress_only_reads_tail(tmp_path: Path) -> None:
    assert tqdm_progress(_log(tmp_path, BAR + "\n" + "x" * 8192)) is None


# ------------------------------------------------------------ failure semantics


def test_failure_default_back_then_failed(home: Path) -> None:
    sched = scheduler()
    job = enqueue(sched.conn, "false")
    log = home / "logs" / f"{job}.log"
    sched.finalize(job, exit_code=1, log_path=log)
    assert state_of(sched.conn, job) == "queued"
    assert (
        sched.conn.execute("SELECT retries FROM jobs WHERE id=?", (job,)).fetchone()[
            "retries"
        ]
        == 1
    )
    sched.finalize(job, exit_code=1, log_path=log)
    assert state_of(sched.conn, job) == "failed"


def test_success_resets_failure_streak(home: Path) -> None:
    sched = scheduler()
    bad, good = enqueue(sched.conn, "false"), enqueue(sched.conn, "true")
    sched.finalize(bad, 1, home / "logs" / "1.log")
    assert q.ctl_get(sched.conn, "fail_streak", "0") == "1"
    sched.finalize(good, 0, home / "logs" / "2.log")
    assert q.ctl_get(sched.conn, "fail_streak", "0") == "0"


def test_halt_after_consecutive_failures(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(q, "HALT_AFTER", 3)
    monkeypatch.setattr(q, "PAUSE_SECONDS", 0.0)  # windows expire instantly
    sched = scheduler()
    for _ in range(3):
        job = enqueue(sched.conn, "false")
        sched.finalize(job, 1, home / "logs" / f"{job}.log")
    assert q.ctl_get(sched.conn, "halted", "0") == "1"
    assert sched.paused(time.time())
    assert NOTIFICATIONS[-1] == "qsched: dispatch halted"


def test_halt_not_triggered_when_successes_interleave(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(q, "HALT_AFTER", 3)
    monkeypatch.setattr(q, "PAUSE_SECONDS", 0.0)
    sched = scheduler()
    for exit_code in (1, 1, 0, 1, 1):
        job = enqueue(sched.conn, "false")
        sched.finalize(job, exit_code, home / "logs" / f"{job}.log")
    assert q.ctl_get(sched.conn, "halted", "0") == "0"


def test_halt_disabled_by_zero(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(q, "HALT_AFTER", 0)
    monkeypatch.setattr(q, "PAUSE_SECONDS", 0.0)
    sched = scheduler()
    for _ in range(20):
        job = enqueue(sched.conn, "false")
        sched.finalize(job, 1, home / "logs" / f"{job}.log")
    assert q.ctl_get(sched.conn, "halted", "0") == "0"


def test_pauses_do_not_stack(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(q, "PAUSE_SECONDS", 60.0)
    monkeypatch.setattr(q, "HALT_AFTER", 0)
    sched = scheduler()
    first = enqueue(sched.conn, "false")
    sched.finalize(first, 1, home / "logs" / f"{first}.log")
    pause_until = q.ctl_get(sched.conn, "pause_until", "0")
    second = enqueue(sched.conn, "false")
    sched.finalize(second, 1, home / "logs" / f"{second}.log")
    assert q.ctl_get(sched.conn, "pause_until", "0") == pause_until


def test_extend_converts_halt_and_resets_streak(home: Path) -> None:
    conn = q.db()
    q.ctl_set(conn, "halted", "1")
    q.ctl_set(conn, "fail_streak", "7")
    q.cmd_extend(1.0)
    conn = q.db()
    assert q.ctl_get(conn, "halted", "0") == "0"
    assert q.ctl_get(conn, "fail_streak", "0") == "0"
    assert float(q.ctl_get(conn, "pause_until", "0")) > time.time()


def test_paused_scheduler_dispatches_nothing(home: Path) -> None:
    sched = scheduler()
    enqueue(sched.conn, "true")
    q.ctl_set(sched.conn, "halted", "1")
    sched.dispatch(time.time())
    assert not sched.running


# --------------------------------------------------------- cancel / restart paths


def test_cancel_queued_is_immediate(home: Path) -> None:
    conn = q.db()
    job = enqueue(conn, "true")
    q.cmd_cancel([job])
    assert state_of(q.db(), job) == "canceled"


def test_finalize_honours_canceling_and_restarting(home: Path) -> None:
    sched = scheduler()
    canceled, restarted = enqueue(sched.conn, "true"), enqueue(sched.conn, "true")
    rank = sched.conn.execute(
        "SELECT rank FROM jobs WHERE id=?", (restarted,)
    ).fetchone()["rank"]
    sched.conn.execute("UPDATE jobs SET state='canceling' WHERE id=?", (canceled,))
    sched.conn.execute("UPDATE jobs SET state='restarting' WHERE id=?", (restarted,))
    sched.finalize(canceled, 143, home / "logs" / "1.log")
    sched.finalize(restarted, 143, home / "logs" / "2.log")
    assert state_of(sched.conn, canceled) == "canceled"
    row = sched.conn.execute(
        "SELECT state, rank, retries FROM jobs WHERE id=?", (restarted,)
    ).fetchone()
    assert (row["state"], row["rank"], row["retries"]) == ("queued", rank, 0)
    assert q.ctl_get(sched.conn, "fail_streak", "0") == "0"  # neither is a failure


def test_restart_requires_ids_or_a_selector(home: Path) -> None:
    conn = q.db()
    enqueue(conn, "true")
    with pytest.raises(SystemExit, match="no job ids given"):
        q.cmd_restart([], [])
    with pytest.raises(SystemExit, match="not both"):
        q.cmd_restart([1], ["running"])


def test_restart_running_defers_to_the_scheduler(home: Path) -> None:
    conn = q.db()
    job = enqueue(conn, "true")
    rank = conn.execute("SELECT rank FROM jobs WHERE id=?", (job,)).fetchone()["rank"]
    conn.execute("UPDATE jobs SET state='running', retries=1 WHERE id=?", (job,))
    conn.commit()
    q.cmd_restart([job], [])
    row = q.db().execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
    assert (row["state"], row["rank"], row["retries"]) == ("restarting", rank, 1)


def test_restart_undoes_an_in_flight_cancel(home: Path) -> None:
    conn = q.db()
    job = enqueue(conn, "true")
    conn.execute("UPDATE jobs SET state='running' WHERE id=?", (job,))
    conn.commit()
    q.cmd_cancel([job])
    q.cmd_restart([job], [])
    assert state_of(q.db(), job) == "restarting"


def test_restart_loses_the_race_against_a_settled_cancel(home: Path) -> None:
    """The scheduler finalized the cancel first: the terminal path takes over."""
    conn = q.db()
    job = enqueue(conn, "true")
    conn.execute(
        "UPDATE jobs SET state='canceled', finished_at=? WHERE id=?",
        (time.time(), job),
    )
    conn.commit()
    q.cmd_restart([job], [])
    assert state_of(q.db(), job) == "queued"


def test_restart_canceled_goes_to_the_back_and_resets_the_row(home: Path) -> None:
    conn = q.db()
    canceled, queued = enqueue(conn, "true"), enqueue(conn, "true")
    conn.execute(
        "UPDATE jobs SET state='canceled', retries=1, gpu=2, pid=999, exit_code=143, "
        "started_at=?, finished_at=? WHERE id=?",
        (time.time(), time.time(), canceled),
    )
    conn.commit()
    q.cmd_restart([canceled], [])
    row = q.db().execute("SELECT * FROM jobs WHERE id=?", (canceled,)).fetchone()
    assert (row["state"], row["retries"], row["exit_code"]) == ("queued", 0, None)
    assert (row["gpu"], row["pid"], row["started_at"], row["finished_at"]) == (
        None,
        None,
        None,
        None,
    )
    order = [
        int(r["id"]) for r in q.db().execute("SELECT id FROM jobs ORDER BY rank, id")
    ]
    assert order == [queued, canceled]


def test_restart_preserves_listed_order_at_the_back(home: Path) -> None:
    conn = q.db()
    ids = [enqueue(conn, "true") for _ in range(3)]
    conn.execute("UPDATE jobs SET state='done' WHERE id IN (?,?)", (ids[0], ids[1]))
    conn.commit()
    q.cmd_restart([ids[1], ids[0]], [])
    order = [
        int(r["id"]) for r in q.db().execute("SELECT id FROM jobs ORDER BY rank, id")
    ]
    assert order == [ids[2], ids[1], ids[0]]


def test_restart_queued_job_is_a_no_op(home: Path) -> None:
    conn = q.db()
    job, other = enqueue(conn, "true"), enqueue(conn, "true")
    rank = conn.execute("SELECT rank FROM jobs WHERE id=?", (job,)).fetchone()["rank"]
    q.cmd_restart([job], [])
    row = q.db().execute("SELECT state, rank FROM jobs WHERE id=?", (job,)).fetchone()
    assert (row["state"], row["rank"]) == ("queued", rank)
    assert state_of(q.db(), other) == "queued"


def test_restart_selectors_pick_exactly_those_states(home: Path) -> None:
    conn = q.db()
    states = ("done", "failed", "canceled", "queued")
    ids = {state: enqueue(conn, "true") for state in states}
    for state, job_id in ids.items():
        conn.execute("UPDATE jobs SET state=? WHERE id=?", (state, job_id))
    conn.commit()
    q.cmd_restart([], ["failed", "canceled"])
    settled = {
        state: state_of(q.db(), job_id)
        for state, job_id in ids.items()
        if state in ("done", "failed", "canceled")
    }
    assert settled == {"done": "done", "failed": "queued", "canceled": "queued"}


# ------------------------------------------------------------------ end-to-end


def test_dispatch_runs_jobs_and_notifies_on_drain(home: Path) -> None:
    sched = scheduler(gpus=[0, 1])
    good, bad = enqueue(sched.conn, "true"), enqueue(sched.conn, "false")
    run_to_completion(sched)
    assert state_of(sched.conn, good) == "done"
    assert state_of(sched.conn, bad) == "queued"  # first failure: back of the queue
    sched.conn.execute(
        "UPDATE jobs SET state='failed', finished_at=? WHERE id=?", (time.time(), bad)
    )
    sched.conn.commit()
    NOTIFICATIONS.clear()
    sched.check_drain()
    assert NOTIFICATIONS == ["qsched: queue drained — 1 done, 1 failed"]
    sched.check_drain()  # fires once per batch
    assert len(NOTIFICATIONS) == 1


def test_gpu_env_and_cwd_reach_the_job(home: Path, tmp_path: Path) -> None:
    sched = scheduler(gpus=[3])
    workdir = tmp_path / "work"
    workdir.mkdir()
    conn = sched.conn
    (job,) = q.insert_jobs(
        conn,
        [["sh", "-c", "echo $CUDA_VISIBLE_DEVICES $REGION $PWD"]],
        {"REGION": "US3"},
    )
    conn.execute("UPDATE jobs SET cwd=? WHERE id=?", (str(workdir), job))
    conn.commit()
    run_to_completion(sched)
    log = (home / "logs" / f"{job}.log").read_text()
    assert f"3 US3 {workdir}" in log


# ------------------------------------------------------------------ housekeeping


def test_front_and_back_preserve_listed_order(home: Path) -> None:
    conn = q.db()
    ids = [enqueue(conn, "true") for _ in range(4)]
    q.cmd_reorder([ids[2], ids[3]], to_front=True)
    q.cmd_reorder([ids[0]], to_front=False)
    order = [
        int(row["id"])
        for row in q.db().execute("SELECT id FROM jobs ORDER BY rank, id")
    ]
    assert order == [ids[2], ids[3], ids[1], ids[0]]


def test_fixed_puts_first_listed_first(home: Path) -> None:
    conn = q.db()
    ids = [enqueue(conn, "true") for _ in range(3)]
    conn.execute("UPDATE jobs SET state='failed' WHERE id IN (?,?)", (ids[1], ids[2]))
    conn.commit()
    q.cmd_fixed([ids[2], ids[1]])
    order = [
        int(row["id"])
        for row in q.db().execute("SELECT id FROM jobs ORDER BY rank, id")
    ]
    assert order == [ids[2], ids[1], ids[0]]


def test_clear_deletes_terminal_jobs_and_their_logs(home: Path) -> None:
    conn = q.db()
    done, queued = enqueue(conn, "true"), enqueue(conn, "true")
    conn.execute(
        "UPDATE jobs SET state='done', finished_at=? WHERE id=?", (time.time(), done)
    )
    conn.commit()
    logs = home / "logs"
    logs.mkdir()
    for job_id in (done, queued):
        (logs / f"{job_id}.log").write_text("x")
    (logs / "999.log").write_text("orphan")
    q.cmd_clear(wipe_all=False, older_than_days=None, dry_run=False)
    assert {p.name for p in logs.iterdir()} == {f"{queued}.log"}
    assert [int(r["id"]) for r in q.db().execute("SELECT id FROM jobs")] == [queued]


def test_clear_older_than_spares_recent_jobs(home: Path) -> None:
    conn = q.db()
    old, recent = enqueue(conn, "true"), enqueue(conn, "true")
    conn.execute(
        "UPDATE jobs SET state='done', finished_at=? WHERE id=?",
        (time.time() - 5 * 86400, old),
    )
    conn.execute(
        "UPDATE jobs SET state='done', finished_at=? WHERE id=?", (time.time(), recent)
    )
    conn.commit()
    q.cmd_clear(wipe_all=False, older_than_days=2.0, dry_run=False)
    assert [int(r["id"]) for r in q.db().execute("SELECT id FROM jobs")] == [recent]


def test_clear_dry_run_changes_nothing(home: Path) -> None:
    conn = q.db()
    job = enqueue(conn, "true")
    conn.execute(
        "UPDATE jobs SET state='done', finished_at=? WHERE id=?", (time.time(), job)
    )
    conn.commit()
    q.cmd_clear(wipe_all=False, older_than_days=None, dry_run=True)
    assert [int(r["id"]) for r in q.db().execute("SELECT id FROM jobs")] == [job]


def test_clear_all_wipes_queue_and_resets_ids(home: Path) -> None:
    conn = q.db()
    enqueue(conn, "true")
    q.ctl_set(conn, "halted", "1")
    q.cmd_clear(wipe_all=True, older_than_days=None, dry_run=False)
    conn = q.db()
    assert conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 0
    assert q.ctl_get(conn, "halted", "0") == "0"
    assert enqueue(conn, "true") == 1


def test_clear_all_refuses_while_scheduler_holds_lock(home: Path) -> None:
    conn = q.db()
    enqueue(conn, "true")
    home.mkdir(parents=True, exist_ok=True)
    with open(q.lock_path(), "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="a scheduler is running"):
            q.cmd_clear(wipe_all=True, older_than_days=None, dry_run=False)
    assert q.db().execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 1


def test_show_reports_env_and_cwd(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    conn = q.db()
    (job,) = q.insert_jobs(
        conn, [["python", "train.py", "model=a"]], {"REGION": "US3"}
    )
    q.cmd_show(job, resubmit=False)
    out = capsys.readouterr().out
    assert "env: REGION=US3" in out
    assert "cmd: python train.py model=a" in out
    q.cmd_show(job, resubmit=True)
    assert (
        capsys.readouterr().out.strip()
        == "q add --env REGION=US3 -- python train.py model=a"
    )


# -------------------------------------------------- pause / halt override commands


def test_failure_notification_is_throttled_by_the_pause(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(q, "PAUSE_SECONDS", 60.0)
    monkeypatch.setattr(q, "HALT_AFTER", 0)
    sched = scheduler()
    for _ in range(3):  # three failures, one open pause window
        job = enqueue(sched.conn, "false")
        sched.finalize(job, 1, home / "logs" / f"{job}.log")
    assert len(NOTIFICATIONS) == 1


def test_notification_kinds(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(q, "HALT_AFTER", 2)
    monkeypatch.setattr(q, "PAUSE_SECONDS", 0.0)
    sched = scheduler()
    for _ in range(2):
        job = enqueue(sched.conn, "false")
        sched.finalize(job, 1, home / "logs" / f"{job}.log")
    assert KINDS == ["failure", "halt"]


def test_resume_clears_pause_halt_and_streak(home: Path) -> None:
    conn = q.db()
    q.ctl_set(conn, "halted", "1")
    q.ctl_set(conn, "pause_until", str(time.time() + 600))
    q.ctl_set(conn, "fail_streak", "7")
    q.cmd_resume()
    conn = q.db()
    assert q.ctl_get(conn, "halted", "0") == "0"
    assert float(q.ctl_get(conn, "pause_until", "0")) == 0.0
    assert q.ctl_get(conn, "fail_streak", "0") == "0"


def test_resume_revives_no_jobs(home: Path) -> None:
    conn = q.db()
    failed = enqueue(conn, "false")
    conn.execute("UPDATE jobs SET state='failed', retries=1 WHERE id=?", (failed,))
    conn.commit()
    q.cmd_resume()
    assert state_of(q.db(), failed) == "failed"


def test_fixed_clears_a_halt(home: Path) -> None:
    conn = q.db()
    job = enqueue(conn, "false")
    conn.execute("UPDATE jobs SET state='failed' WHERE id=?", (job,))
    conn.commit()
    q.ctl_set(conn, "halted", "1")
    q.cmd_fixed([job])
    assert q.ctl_get(q.db(), "halted", "0") == "0"


def test_restart_failed_resets_retries_and_goes_to_the_back(home: Path) -> None:
    conn = q.db()
    failed, queued = enqueue(conn, "false"), enqueue(conn, "true")
    conn.execute(
        "UPDATE jobs SET state='failed', retries=1, exit_code=1, finished_at=? "
        "WHERE id=?",
        (time.time(), failed),
    )
    conn.commit()
    q.cmd_restart([], ["failed"])
    row = q.db().execute("SELECT * FROM jobs WHERE id=?", (failed,)).fetchone()
    assert (row["state"], row["retries"], row["exit_code"]) == ("queued", 0, None)
    order = [
        int(r["id"]) for r in q.db().execute("SELECT id FROM jobs ORDER BY rank, id")
    ]
    assert order == [queued, failed]


def test_main_maps_restart_flags_to_states(home: Path) -> None:
    conn = q.db()
    failed, done = enqueue(conn, "false"), enqueue(conn, "true")
    conn.execute("UPDATE jobs SET state='failed' WHERE id=?", (failed,))
    conn.execute("UPDATE jobs SET state='done' WHERE id=?", (done,))
    conn.commit()
    q.main(["restart", "--failed"])
    assert (state_of(q.db(), failed), state_of(q.db(), done)) == ("queued", "done")


def test_restart_keeps_a_halt(home: Path) -> None:
    conn = q.db()
    failed = enqueue(conn, "false")
    conn.execute("UPDATE jobs SET state='failed' WHERE id=?", (failed,))
    conn.commit()
    q.ctl_set(conn, "halted", "1")
    q.cmd_restart([], ["failed"])
    assert q.ctl_get(q.db(), "halted", "0") == "1"  # restarting is not looking


def test_front_ranks_degenerate_when_queued_outranks_running(home: Path) -> None:
    conn = q.db()
    running, queued = enqueue(conn, "true"), enqueue(conn, "true")
    conn.execute("UPDATE jobs SET state='running', rank=9 WHERE id=?", (running,))
    conn.execute("UPDATE jobs SET rank=1 WHERE id=?", (queued,))
    conn.commit()
    ranks = q.queue_front_ranks(conn, 1)
    assert ranks[0] < 1.0  # queue position wins; may precede the running job


# ------------------------------------------------------------------- submission


def test_parse_env_pairs_rejects_malformed(home: Path) -> None:
    assert q.parse_env_pairs(["A=1", "B="]) == {"A": "1", "B": ""}
    with pytest.raises(SystemExit, match="--env expects KEY=VALUE"):
        q.parse_env_pairs(["NOEQUALS"])
    with pytest.raises(SystemExit, match="--env expects KEY=VALUE"):
        q.parse_env_pairs(["=novalue"])


def test_sweep_dry_run_lines_round_trip(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    command = ["python", "t.py", 'grouping=["Day","key"],["Day"]', "lr=0.1,0.01"]
    q.cmd_sweep([], command, dry_run=True)
    lines = capsys.readouterr().out.splitlines()
    assert [shlex.split(line) for line in lines] == expand_sweep(command)
    assert q.db().execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 0


def test_base_env_drops_uv_script_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/script-venv")
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    monkeypatch.setenv("PATH", "/tmp/script-venv/bin:/usr/bin")
    env = q.base_env()
    assert "VIRTUAL_ENV" not in env and "UV_RUN_RECURSION_DEPTH" not in env
    assert env["PATH"] == "/usr/bin"


def test_spawn_failure_is_a_normal_failure(home: Path) -> None:
    sched = scheduler()
    (job,) = q.insert_jobs(sched.conn, [["definitely-not-a-real-binary"]], {})
    sched.dispatch(time.time())
    row = sched.conn.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
    assert (row["state"], row["retries"], row["exit_code"]) == ("queued", 1, 127)
    assert "spawn failed" in (home / "logs" / f"{job}.log").read_text()


def test_cancel_running_defers_to_the_scheduler(home: Path) -> None:
    conn = q.db()
    job = enqueue(conn, "true")
    conn.execute("UPDATE jobs SET state='running' WHERE id=?", (job,))
    conn.commit()
    q.cmd_cancel([job])
    assert state_of(q.db(), job) == "canceling"


# ------------------------------------------------------------ validation probes


def test_probe_passes_only_on_the_agreed_exit_code(home: Path) -> None:
    assert run_probe(["sh", "-c", VALID_CMD], {}, str(home)).ok
    # a target that ignores QSCHED_VALIDATE exits 0: never silently a pass
    unaware = run_probe(["sh", "-c", "echo ran the real job"], {}, str(home))
    assert not unaware.ok and "expected 80" in unaware.reason
    assert "ran the real job" in unaware.output


def test_probe_reports_nonzero_exit_with_output(home: Path) -> None:
    result = run_probe(["sh", "-c", "echo bad config >&2; exit 3"], {}, str(home))
    assert not result.ok and result.reason == "exit 3"
    assert "bad config" in result.output


def test_probe_ok_code_is_not_confused_with_failure(home: Path) -> None:
    assert q.VALIDATE_OK_CODE not in (0, 1, 2, 126, 127)  # no convention claims it
    assert run_probe(["sh", "-c", "exit 79"], {}, str(home)).reason == "exit 79"
    assert run_probe(["sh", "-c", "exit 81"], {}, str(home)).reason == "exit 81"


def test_probe_times_out(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(q, "VALIDATE_TIMEOUT", 0.3)
    result = run_probe(["sh", "-c", "sleep 30"], {}, str(home))
    assert not result.ok and "no answer" in result.reason


def test_probe_spawn_error(home: Path) -> None:
    assert not run_probe(["definitely-not-a-real-binary"], {}, str(home)).ok


def test_probe_env_hides_gpus_and_carries_job_env(home: Path) -> None:
    argv = [
        "sh",
        "-c",
        f'echo "V=$QSCHED_VALIDATE CUDA=[$CUDA_VISIBLE_DEVICES] R=$REGION P=$PWD"; '
        f"exit 1",
    ]
    result = run_probe(argv, {"REGION": "US3"}, str(home))
    assert f"V=1 CUDA=[] R=US3 P={home}" in result.output


def test_probes_keep_combo_order_when_parallel(home: Path) -> None:
    combos = [
        ["sh", "-c", f"sleep 0.3; {VALID_CMD}"],
        ["sh", "-c", "exit 1"],
        ["sh", "-c", VALID_CMD],
    ]
    results = run_probes(combos, {})
    assert [r.ok for r in results] == [True, False, True]


def test_validated_combos_all_bad_aborts(home: Path) -> None:
    with pytest.raises(SystemExit, match=r"all 2 job\(s\) rejected"):
        validated_combos([["false"], ["false"]], {}, "skip")


def test_validated_combos_keeps_survivors_with_yes(home: Path) -> None:
    combos = [["sh", "-c", VALID_CMD], ["false"]]
    assert validated_combos(combos, {}, "skip") == [combos[0]]


def test_validated_combos_declined_enqueues_nothing(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        q, "prompt_confirm", lambda question, timeout, *, default: False
    )
    with pytest.raises(SystemExit, match="nothing enqueued"):
        validated_combos([["sh", "-c", VALID_CMD], ["false"]], {}, "ask")


def test_add_with_validate_enqueues_nothing_on_rejection(home: Path) -> None:
    with pytest.raises(SystemExit, match=r"all 1 job\(s\) rejected"):
        q.cmd_add([], ["false"], validate=True, on_reject="skip")
    assert q.db().execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 0


def test_sweep_with_validate_enqueues_only_survivors(home: Path) -> None:
    script = f'test "$1" = "V=ok" && {VALID_CMD}'
    command = ["sh", "-c", script, "probe", "V=ok,bad"]
    q.cmd_sweep([], command, dry_run=False, validate=True, on_reject="skip")
    rows = q.db().execute("SELECT argv FROM jobs").fetchall()
    assert len(rows) == 1 and "V=ok" in rows[0]["argv"]


def test_sweep_dry_run_with_validate_reports_and_exits_nonzero(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="1"):
        q.cmd_sweep([], ["false"], dry_run=True, validate=True)
    captured = capsys.readouterr()
    assert captured.out.strip() == "false"  # stdout stays paste-able
    assert "0 passed, 1 rejected" in captured.err
    assert q.db().execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 0


# --------------------------------------------------------------- status digest


def test_next_digest_after_picks_the_next_listed_hour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(q, "STATUS_HOURS", [8, 14, 20])
    at = datetime(2026, 6, 1, 9, 30).timestamp()
    assert datetime.fromtimestamp(q.next_digest_after(at)).hour == 14
    at = datetime(2026, 6, 1, 21, 0).timestamp()
    tomorrow = datetime.fromtimestamp(q.next_digest_after(at))
    assert (tomorrow.day, tomorrow.hour) == (2, 8)


def test_digest_off_when_no_hours_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(q, "STATUS_HOURS", [])
    assert q.next_digest_after(time.time()) == float("inf")
    assert q._status_hours("") == []


def test_status_hours_rejects_nonsense() -> None:
    assert q._status_hours("20,8,8") == [8, 20]
    with pytest.raises(SystemExit, match="hours must be 0-23"):
        q._status_hours("8,24")
    with pytest.raises(SystemExit, match="not a list of hours"):
        q._status_hours("8pm")


def test_digest_waits_until_due(home: Path) -> None:
    sched = scheduler()
    enqueue(sched.conn, "true")
    now = time.time()
    sched.digest_due = now + 3600
    sched.maybe_digest(now)
    assert NOTIFICATIONS == []
    sched.digest_due = now - 1
    sched.maybe_digest(now)
    assert len(NOTIFICATIONS) == 1


def test_digest_carries_the_standard_status_view(home: Path) -> None:
    sched = scheduler()
    job = enqueue(sched.conn, "true")
    sched.digest_due = time.time() - 1
    sched.maybe_digest(time.time())
    assert KINDS == ["digest"]
    assert "queued 1  (1 total)" in NOTIFICATIONS[0]
    assert BODIES[0] == q.render_status(False)
    assert f"{job:>5} queued" in BODIES[0]


def test_digest_reschedules_to_the_next_hour(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(q, "STATUS_HOURS", [8, 14, 20])
    sched = scheduler()
    enqueue(sched.conn, "true")
    sched.digest_due = time.time() - 1
    sched.maybe_digest(time.time())
    assert datetime.fromtimestamp(sched.digest_due).hour in (8, 14, 20)
    assert sched.digest_due > time.time()


def test_digest_stays_silent_while_idle(home: Path) -> None:
    sched = scheduler()
    job = enqueue(sched.conn, "true")
    sched.conn.execute(
        "UPDATE jobs SET state='done', finished_at=? WHERE id=?", (time.time(), job)
    )
    sched.conn.commit()
    sched.digest_due = time.time() - 1
    sched.maybe_digest(time.time())  # reports the job that settled
    assert len(NOTIFICATIONS) == 1
    sched.digest_due = time.time() - 1
    sched.maybe_digest(time.time())  # nothing since: silent
    assert len(NOTIFICATIONS) == 1


def test_hook_receives_title_body_and_kind(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()  # restore the real notify()
    monkeypatch.setenv("QSCHED_HOME", str(home))
    hook = home / "notify.sh"
    record = home / "hook-args"
    hook.write_text(f'#!/bin/sh\nprintf "%s|%s" "$2" "$3" > {record}\n')
    hook.chmod(0o755)
    monkeypatch.setenv("QSCHED_NOTIFY", str(hook))
    q.notify("t", "b", kind="digest")
    assert record.read_text() == "[DIGEST] t|b"


# ------------------------------------------------- helpers for the sections below


def enqueue_job_in_state(conn: sqlite3.Connection, state: str, *argv: str) -> int:
    job_id = enqueue(conn, *argv)
    conn.execute("UPDATE jobs SET state=? WHERE id=?", (state, job_id))
    conn.commit()
    return job_id


def queue_order_of_job_ids(conn: sqlite3.Connection) -> list[int]:
    return [
        int(row["id"]) for row in conn.execute("SELECT id FROM jobs ORDER BY rank, id")
    ]


def set_state_from_another_connection(job_id: int, state: str) -> None:
    """What a concurrent CLI process does while the code under test is mid-flight."""
    other = sqlite3.connect(q.db_path(), timeout=10.0)  # not q.db(): tests patch it
    other.execute("UPDATE jobs SET state=? WHERE id=?", (state, job_id))
    other.commit()
    other.close()


class ConnectionWithRaceHook:
    """Wraps a connection and runs `hook` once before the first matching statement.

    A statement matches when it contains `trigger`; the hook stands in for a
    concurrent writer, deterministically.
    """

    def __init__(
        self, real: sqlite3.Connection, trigger: str, hook: Callable[[], None]
    ) -> None:
        self._real = real
        self._trigger = trigger
        self._hook = hook
        self._armed = True

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
        if self._armed and self._trigger in sql:
            self._armed = False
            self._hook()
        return self._real.execute(sql, params)

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


def mark_job_done_at(conn: sqlite3.Connection, job_id: int, finished_at: float) -> None:
    conn.execute(
        "UPDATE jobs SET state='done', exit_code=0, started_at=?, finished_at=? "
        "WHERE id=?",
        (finished_at - 1.0, finished_at, job_id),
    )
    conn.commit()


def insert_done_job_with_wall_time(
    conn: sqlite3.Connection, wall_seconds: float, finished_ago: float = 10.0
) -> int:
    job_id = enqueue(conn, "true")
    now = time.time()
    conn.execute(
        "UPDATE jobs SET state='done', exit_code=0, started_at=?, finished_at=? "
        "WHERE id=?",
        (now - finished_ago - wall_seconds, now - finished_ago, job_id),
    )
    conn.commit()
    return job_id


def insert_done_job_between(
    conn: sqlite3.Connection, started_at: float, finished_at: float
) -> int:
    job_id = enqueue(conn, "true")
    conn.execute(
        "UPDATE jobs SET state='done', exit_code=0, started_at=?, finished_at=? "
        "WHERE id=?",
        (started_at, finished_at, job_id),
    )
    conn.commit()
    return job_id


def insert_running_job_started_ago(
    conn: sqlite3.Connection, seconds: float, log_path: Path | None = None, gpu: int = 0
) -> int:
    job_id = enqueue(conn, "true")
    conn.execute(
        "UPDATE jobs SET state='running', gpu=?, started_at=?, log_path=? WHERE id=?",
        (
            gpu,
            time.time() - seconds,
            None if log_path is None else str(log_path),
            job_id,
        ),
    )
    conn.commit()
    return job_id


def insert_job_with_log_bytes(home: Path, content: bytes) -> int:
    conn = q.db()
    job_id = enqueue(conn, "true")
    log = home / "logs" / f"{job_id}.log"
    log.parent.mkdir(exist_ok=True)
    log.write_bytes(content)
    conn.execute("UPDATE jobs SET log_path=? WHERE id=?", (str(log), job_id))
    conn.commit()
    return job_id


def numbered_log_lines(count: int) -> bytes:
    return "".join(f"line {i}\n" for i in range(1, count + 1)).encode()


def drain_separator_line() -> str:
    return "-" * q.STATUS_TABLE_WIDTH


# ------------------------------------------------------------- job id selectors


@pytest.mark.parametrize(
    ("token", "expected"),
    [("7", (7, 7)), ("120-160", (120, 160)), ("5-5", (5, 5)), ("0", (0, 0))],
)
def test_parse_id_spec_reads_plain_ids_and_ranges(
    token: str, expected: tuple[int, int]
) -> None:
    assert q.parse_id_spec(token) == expected


@pytest.mark.parametrize("token", ["x", "", "5-", "-5", "1-2-3", "5-x", "3.5"])
def test_parse_id_spec_rejects_malformed_tokens(token: str) -> None:
    with pytest.raises(
        argparse.ArgumentTypeError, match="expected a job id or a range"
    ):
        q.parse_id_spec(token)


def test_parse_id_spec_rejects_reversed_ranges() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="empty range"):
        q.parse_id_spec("8-7")


def test_resolve_ids_keeps_listed_order_and_drops_repeats(home: Path) -> None:
    conn = q.db()
    for _ in range(6):
        enqueue(conn)
    assert q.resolve_ids([(5, 5), (2, 4), (3, 3)]) == [5, 2, 3, 4]


def test_resolve_ids_skips_gaps_in_ranges_but_keeps_missing_plain_ids(
    home: Path,
) -> None:
    conn = q.db()
    for _ in range(5):
        enqueue(conn)
    conn.execute("DELETE FROM jobs WHERE id=3")
    conn.commit()
    assert q.resolve_ids([(1, 5)]) == [1, 2, 4, 5]
    assert q.resolve_ids([(99, 99)]) == [99]  # the command reports "not found"


def test_resolve_ids_reports_a_range_that_matches_no_job(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    enqueue(q.db())
    assert q.resolve_ids([(100, 200)]) == []
    assert capsys.readouterr().out.strip() == "no jobs in range 100-200"


def test_resolve_ids_leaves_the_database_alone_for_plain_ids(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_db() -> sqlite3.Connection:
        raise AssertionError("db() must not be opened for plain ids")

    monkeypatch.setattr(q, "db", forbidden_db)
    assert q.resolve_ids([(4, 4), (2, 2)]) == [4, 2]


def test_main_cancel_accepts_ranges_and_plain_ids(home: Path) -> None:
    conn = q.db()
    ids = [enqueue(conn) for _ in range(5)]
    q.main(["cancel", "2-3", "5"])
    assert [state_of(q.db(), job_id) for job_id in ids] == [
        "queued",
        "canceled",
        "canceled",
        "queued",
        "canceled",
    ]


def test_main_rejects_a_reversed_range_with_a_usage_error(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="2"):
        q.main(["cancel", "5-3"])
    assert "empty range" in capsys.readouterr().err


def test_main_front_expands_a_range_in_ascending_order(home: Path) -> None:
    conn = q.db()
    ids = [enqueue(conn) for _ in range(4)]
    q.main(["front", "3-4"])
    assert queue_order_of_job_ids(q.db()) == [ids[2], ids[3], ids[0], ids[1]]


def test_main_restart_accepts_ranges(home: Path) -> None:
    conn = q.db()
    ids = [enqueue_job_in_state(conn, "done") for _ in range(3)]
    q.main(["restart", "1-2"])
    assert [state_of(q.db(), job_id) for job_id in ids] == ["queued", "queued", "done"]


def test_main_fixed_accepts_ranges(home: Path) -> None:
    conn = q.db()
    ids = [enqueue_job_in_state(conn, "failed") for _ in range(3)]
    q.main(["fixed", "2-3"])
    assert [state_of(q.db(), job_id) for job_id in ids] == [
        "failed",
        "queued",
        "queued",
    ]


# ------------------------------------------- one-transaction submission (sweeps)


def test_insert_jobs_ids_are_contiguous_and_ranks_ascend(home: Path) -> None:
    conn = q.db()
    enqueue(conn)  # an earlier submission
    assert q.insert_jobs(conn, [["true"]] * 3, {}) == [2, 3, 4]
    ranks = [row["rank"] for row in conn.execute("SELECT rank FROM jobs ORDER BY id")]
    assert ranks == sorted(ranks) and len(set(ranks)) == 4


def test_insert_jobs_records_state_env_and_cwd(home: Path) -> None:
    conn = q.db()
    (job_id,) = q.insert_jobs(conn, [["true"]], {"REGION": "US3"}, "backlog")
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    assert row["state"] == "backlog"
    assert row["env"] == '{"REGION": "US3"}'
    assert row["cwd"] == str(Path.cwd())


def test_insert_jobs_is_all_or_nothing(home: Path) -> None:
    conn = q.db()
    with pytest.raises(TypeError, match="not JSON serializable"):
        q.insert_jobs(conn, [["true"], ["true"], [object()]], {})  # type: ignore[list-item]
    assert conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 0
    assert q.insert_jobs(conn, [["true"]], {}) == [1]  # the rollback burned no ids


def test_insert_jobs_batch_is_not_interleaved_with_concurrent_submitters(
    home: Path,
) -> None:
    sweep_ids: list[int] = []

    def submit_sweep() -> None:
        sweep_ids.extend(
            q.insert_jobs(q.db(), [["echo", f"sweepmark={i}"] for i in range(300)], {})
        )

    def submit_singles() -> None:
        for _ in range(30):
            q.insert_jobs(q.db(), [["echo", "single"]], {})

    threads = [threading.Thread(target=submit_sweep)] + [
        threading.Thread(target=submit_singles) for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sweep_ids == list(range(sweep_ids[0], sweep_ids[0] + 300))
    argvs = q.db().execute(
        "SELECT argv FROM jobs WHERE id BETWEEN ? AND ?", (sweep_ids[0], sweep_ids[-1])
    )
    assert all("sweepmark" in row["argv"] for row in argvs)


def test_format_id_range() -> None:
    assert q.format_id_range([7]) == "7"
    assert q.format_id_range([120, 121, 131]) == "120-131"


def test_add_prints_the_legacy_line_for_a_queued_job(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    q.cmd_add([], ["echo", "hi"])
    assert capsys.readouterr().out == "enqueued job 1: echo hi\n"


def test_sweep_prints_the_id_range_it_enqueued(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    q.cmd_sweep([], ["echo", "k=1,2,3"], dry_run=False)
    assert capsys.readouterr().out == "enqueued 3 job(s): ids 1-3\n"


def test_sweep_of_one_combination_prints_a_bare_id(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    q.cmd_sweep([], ["echo", "k=1"], dry_run=False)
    assert capsys.readouterr().out == "enqueued 1 job(s): ids 1\n"


# ------------------------------------------------------------- the drain line


def test_check_drain_stamps_a_time_not_before_the_last_finish(home: Path) -> None:
    sched = scheduler(gpus=[0, 1])
    for _ in range(3):
        enqueue(sched.conn, "true")
    run_to_completion(sched)
    drained_at = float(q.ctl_get(sched.conn, "last_drain_at", "0"))
    finished = [
        row["finished_at"] for row in sched.conn.execute("SELECT finished_at FROM jobs")
    ]
    assert drained_at > 0 and all(moment <= drained_at for moment in finished)


@pytest.mark.parametrize("show_all", [False, True])
def test_status_puts_the_drain_line_below_the_final_job(
    home: Path, show_all: bool
) -> None:
    sched = scheduler(gpus=[0, 1])
    ids = [enqueue(sched.conn, "true") for _ in range(3)]
    run_to_completion(sched)
    lines = q.render_status(show_all, sched.conn).splitlines()
    assert lines.count(drain_separator_line()) == 1
    assert lines[-1] == drain_separator_line()
    assert lines[-2].lstrip().startswith(f"{ids[-1]} ")


def test_status_drain_line_separates_old_rows_from_new_ones(home: Path) -> None:
    conn = q.db()
    now = time.time()
    first, second = enqueue(conn), enqueue(conn)
    third = enqueue(conn)
    mark_job_done_at(conn, first, now - 100)
    mark_job_done_at(conn, second, now - 90)
    q.ctl_set(conn, "last_drain_at", str(now - 50))
    lines = q.render_status(False, conn).splitlines()
    at = lines.index(drain_separator_line())
    assert lines[at - 1].lstrip().startswith(f"{second} ")
    assert lines[at + 1].lstrip().startswith(f"{third} ")
    assert lines.count(drain_separator_line()) == 1


def test_status_has_no_drain_line_before_any_drain(home: Path) -> None:
    conn = q.db()
    mark_job_done_at(conn, enqueue(conn), time.time() - 5)
    assert drain_separator_line() not in q.render_status(False, conn).splitlines()


def test_status_keeps_the_line_above_rows_that_finished_after_the_recorded_drain(
    home: Path,
) -> None:
    """Databases drained before the fix keep the line above the last rows.

    Their recorded drain time is earlier than the last finish; the line stays
    there until the next drain overwrites it.
    """
    conn = q.db()
    now = time.time()
    early, late = enqueue(conn), enqueue(conn)
    mark_job_done_at(conn, early, now - 100)
    mark_job_done_at(conn, late, now - 10)
    q.ctl_set(conn, "last_drain_at", str(now - 50))
    lines = q.render_status(False, conn).splitlines()
    at = lines.index(drain_separator_line())
    assert lines[at - 1].lstrip().startswith(f"{early} ")
    assert lines[at + 1].lstrip().startswith(f"{late} ")


# --------------------------------------------------------------------- backlog


def test_add_backlog_enqueues_into_the_backlog_state(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    q.cmd_add([], ["echo", "parked"], backlog=True)
    assert state_of(q.db(), 1) == "backlog"
    assert capsys.readouterr().out == "enqueued job 1 to backlog: echo parked\n"


def test_sweep_backlog_enqueues_every_combination_into_the_backlog(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    q.cmd_sweep([], ["echo", "k=1,2,3"], dry_run=False, backlog=True)
    assert capsys.readouterr().out == "enqueued 3 job(s) to backlog: ids 1-3\n"
    assert [state_of(q.db(), job_id) for job_id in (1, 2, 3)] == ["backlog"] * 3


def test_main_submits_into_the_backlog(home: Path) -> None:
    q.main(["add", "--backlog", "--", "true"])
    q.main(["sweep", "--backlog", "--", "echo", "k=1,2"])
    assert [state_of(q.db(), job_id) for job_id in (1, 2, 3)] == ["backlog"] * 3


def test_backlog_command_parks_only_queued_jobs(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    conn = q.db()
    queued = enqueue(conn)
    running = enqueue_job_in_state(conn, "running")
    done = enqueue_job_in_state(conn, "done")
    q.cmd_backlog([queued, running, done, 99])
    out = capsys.readouterr().out
    assert state_of(q.db(), queued) == "backlog"
    assert state_of(q.db(), running) == "running"
    assert state_of(q.db(), done) == "done"
    assert f"job {queued}: moved to backlog" in out
    assert f"job {running}: state running, only queued jobs can be backlogged" in out
    assert "job 99: not found" in out


def test_backlog_command_reports_a_job_that_is_already_parked(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = enqueue_job_in_state(q.db(), "backlog")
    q.cmd_backlog([job])
    assert capsys.readouterr().out.strip() == f"job {job}: already in backlog"


def test_backlog_command_accepts_ranges(home: Path) -> None:
    conn = q.db()
    ids = [enqueue(conn) for _ in range(4)]
    q.main(["backlog", "2-3"])
    assert [state_of(q.db(), job_id) for job_id in ids] == [
        "queued",
        "backlog",
        "backlog",
        "queued",
    ]


def test_backlog_command_loses_gracefully_when_the_scheduler_started_the_job(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observer = q.db()
    job = enqueue(observer)
    racing = ConnectionWithRaceHook(
        q.db(),
        "SET state='backlog'",
        lambda: set_state_from_another_connection(job, "running"),
    )
    monkeypatch.setattr(q, "db", lambda: racing)
    q.cmd_backlog([job])
    assert state_of(observer, job) == "running"  # the scheduler's start was kept
    out = capsys.readouterr().out
    assert "state running, only queued jobs can be backlogged" in out


def test_release_moves_jobs_to_the_back_in_listed_order(home: Path) -> None:
    conn = q.db()
    queued = enqueue(conn)
    parked = [enqueue_job_in_state(conn, "backlog") for _ in range(3)]
    q.cmd_release([parked[2], parked[0]], release_all=False)
    assert queue_order_of_job_ids(q.db()) == [queued, parked[1], parked[2], parked[0]]
    assert state_of(q.db(), parked[2]) == state_of(q.db(), parked[0]) == "queued"
    assert state_of(q.db(), parked[1]) == "backlog"


def test_release_all_keeps_the_backlogs_own_order_behind_queued_work(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    conn = q.db()
    first_parked = enqueue_job_in_state(conn, "backlog")
    queued = enqueue(conn)
    second_parked = enqueue_job_in_state(conn, "backlog")
    q.cmd_release([], release_all=True)
    assert queue_order_of_job_ids(q.db()) == [queued, first_parked, second_parked]
    assert capsys.readouterr().out.strip() == "released 2 job(s) to back of queue"


def test_release_accepts_ranges(home: Path) -> None:
    conn = q.db()
    ids = [enqueue_job_in_state(conn, "backlog") for _ in range(4)]
    q.main(["release", "2-3"])
    assert [state_of(q.db(), job_id) for job_id in ids] == [
        "backlog",
        "queued",
        "queued",
        "backlog",
    ]


def test_release_requires_ids_or_all(home: Path) -> None:
    with pytest.raises(SystemExit, match="pass ids or --all"):
        q.cmd_release([], release_all=False)
    with pytest.raises(SystemExit, match="not both"):
        q.cmd_release([1], release_all=True)


def test_release_refuses_jobs_that_are_not_in_the_backlog(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    queued = enqueue(q.db())
    q.cmd_release([queued, 99], release_all=False)
    out = capsys.readouterr().out
    assert f"job {queued}: state queued, not in backlog" in out
    assert "job 99: not found" in out


def test_release_all_on_an_empty_backlog(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    q.cmd_release([], release_all=True)
    assert capsys.readouterr().out.strip() == "backlog is empty"


def test_release_does_not_resurrect_a_job_canceled_meanwhile(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observer = q.db()
    job = enqueue_job_in_state(observer, "backlog")
    racing = ConnectionWithRaceHook(
        q.db(),
        "SET state='queued', rank=?",
        lambda: set_state_from_another_connection(job, "canceled"),
    )
    monkeypatch.setattr(q, "db", lambda: racing)
    q.cmd_release([job], release_all=False)
    assert f"job {job}: state canceled, not in backlog" in capsys.readouterr().out
    assert state_of(observer, job) == "canceled"  # not resurrected as queued


def test_dispatch_never_starts_backlog_jobs(home: Path) -> None:
    sched = scheduler(gpus=[0, 1])
    parked = enqueue_job_in_state(sched.conn, "backlog")
    sched.dispatch(time.time())
    assert not sched.running
    assert state_of(sched.conn, parked) == "backlog"


def test_dispatch_runs_queued_jobs_around_the_backlog(home: Path) -> None:
    sched = scheduler(gpus=[0])
    parked = enqueue_job_in_state(sched.conn, "backlog")
    queued = enqueue(sched.conn)
    run_to_completion(sched)
    assert state_of(sched.conn, queued) == "done"
    assert state_of(sched.conn, parked) == "backlog"
    assert not (home / "logs" / f"{parked}.log").exists()


def test_drain_fires_while_the_backlog_waits_and_mentions_it(home: Path) -> None:
    sched = scheduler()
    enqueue(sched.conn)
    enqueue_job_in_state(sched.conn, "backlog")
    enqueue_job_in_state(sched.conn, "backlog")
    run_to_completion(sched)
    assert NOTIFICATIONS == ["qsched: queue drained — 1 done; 2 in backlog"]
    assert KINDS == ["drain"]
    assert "2 in backlog" in BODIES[-1] and "q release --all" in BODIES[-1]


def test_drain_notification_has_no_backlog_text_without_a_backlog(home: Path) -> None:
    sched = scheduler()
    enqueue(sched.conn)
    run_to_completion(sched)
    assert NOTIFICATIONS == ["qsched: queue drained — 1 done"]
    assert "backlog" not in BODIES[-1]


def test_cancel_works_on_backlog_jobs(home: Path) -> None:
    job = enqueue_job_in_state(q.db(), "backlog")
    q.cmd_cancel([job])
    assert state_of(q.db(), job) == "canceled"


def test_cancel_queued_flag_leaves_the_backlog_alone(home: Path) -> None:
    conn = q.db()
    queued, parked = enqueue(conn), enqueue_job_in_state(conn, "backlog")
    q.cmd_cancel([], queued=True)
    assert (state_of(q.db(), queued), state_of(q.db(), parked)) == (
        "canceled",
        "backlog",
    )


def test_cancel_rejects_ids_together_with_the_queued_flag(home: Path) -> None:
    with pytest.raises(SystemExit, match="not both"):
        q.cmd_cancel([1], queued=True)


def test_restart_of_a_backlog_job_points_at_release(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = enqueue_job_in_state(q.db(), "backlog")
    q.cmd_restart([job], [])
    assert state_of(q.db(), job) == "backlog"
    assert f"`q release {job}`" in capsys.readouterr().out


def test_restart_state_selectors_ignore_the_backlog(home: Path) -> None:
    conn = q.db()
    parked = enqueue_job_in_state(conn, "backlog")
    failed = enqueue_job_in_state(conn, "failed")
    q.cmd_restart([], ["running", "failed", "canceled", "done"])
    assert state_of(q.db(), parked) == "backlog"
    assert state_of(q.db(), failed) == "queued"


def test_fixed_stays_failure_only_for_backlog_jobs(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = enqueue_job_in_state(q.db(), "backlog")
    q.cmd_fixed([job])
    assert state_of(q.db(), job) == "backlog"
    assert f"job {job}: state backlog, cannot retry" in capsys.readouterr().out


def test_front_and_back_refuse_backlog_jobs(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = enqueue_job_in_state(q.db(), "backlog")
    q.cmd_reorder([job], to_front=True)
    q.cmd_reorder([job], to_front=False)
    assert state_of(q.db(), job) == "backlog"
    assert capsys.readouterr().out.count("only queued jobs can be moved") == 2


def test_clear_keeps_the_backlog_but_clear_all_wipes_it(home: Path) -> None:
    conn = q.db()
    parked = enqueue_job_in_state(conn, "backlog")
    mark_job_done_at(conn, enqueue(conn), time.time() - 5)
    q.cmd_clear(wipe_all=False, older_than_days=None, dry_run=False)
    assert [int(r["id"]) for r in q.db().execute("SELECT id FROM jobs")] == [parked]
    q.cmd_clear(wipe_all=True, older_than_days=None, dry_run=False)
    assert q.db().execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 0


def test_state_order_lists_the_backlog_between_active_and_terminal_states() -> None:
    assert (*q.ACTIVE_STATES, "backlog", *q.TERMINAL_STATES) == q.STATE_ORDER
    assert "backlog" not in q.ACTIVE_STATES  # parked jobs are never "pending"


def test_status_tally_counts_the_backlog(home: Path) -> None:
    conn = q.db()
    enqueue(conn)
    enqueue_job_in_state(conn, "backlog")
    enqueue_job_in_state(conn, "backlog")
    assert q.status_tally(conn) == "queued 1 · backlog 2  (3 total)"


def test_status_default_and_all_views_hide_backlog_rows(home: Path) -> None:
    conn = q.db()
    enqueue(conn, "echo", "visible")
    enqueue_job_in_state(conn, "backlog", "echo", "parked")
    for show_all in (False, True):
        view = q.render_status(show_all, conn)
        assert "visible" in view and "parked" not in view
        assert "backlog 1" in view.splitlines()[0]  # counted, not listed


def test_status_backlog_view_lists_only_backlog_rows(home: Path) -> None:
    conn = q.db()
    enqueue(conn, "echo", "visible")
    enqueue_job_in_state(conn, "backlog", "echo", "parked")
    view = q.render_status(False, conn, backlog_only=True)
    assert "parked" in view and "visible" not in view
    assert "queued 1 · backlog 1" in view.splitlines()[0]


def test_status_flags_all_and_backlog_conflict(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="2"):
        q.main(["status", "--all", "--backlog"])
    assert "not allowed with argument" in capsys.readouterr().err


def test_main_status_backlog_prints_the_backlog_view(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    enqueue_job_in_state(q.db(), "backlog", "echo", "parked")
    q.main(["status", "--backlog"])
    assert "parked" in capsys.readouterr().out


def test_status_empty_queue_messages(home: Path) -> None:
    conn = q.db()
    assert q.render_status(False, conn) == "queue is empty"
    assert q.render_status(False, conn, backlog_only=True) == "backlog is empty"
    enqueue_job_in_state(conn, "backlog")
    enqueue_job_in_state(conn, "backlog")
    assert q.render_status(False, conn).splitlines()[-1] == (
        "queue is empty — 2 in backlog (`q status --backlog`)"
    )


def test_main_backlog_and_release_commands(home: Path) -> None:
    conn = q.db()
    ids = [enqueue(conn) for _ in range(3)]
    q.main(["backlog", "1-3"])
    assert [state_of(q.db(), job_id) for job_id in ids] == ["backlog"] * 3
    q.main(["release", "--all"])
    assert [state_of(q.db(), job_id) for job_id in ids] == ["queued"] * 3


# --------------------------------------------------- cancel and the scheduler claim


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ("queued", "job 1: canceled"),
        ("backlog", "job 1: canceled"),
        ("running", "job 1: cancel requested (scheduler will SIGTERM)"),
        ("canceling", "job 1: state canceling, nothing to cancel"),
        ("done", "job 1: state done, nothing to cancel"),
        ("failed", "job 1: state failed, nothing to cancel"),
        ("canceled", "job 1: state canceled, nothing to cancel"),
    ],
)
def test_cancel_job_reports_what_it_did_per_state(
    home: Path, state: str, message: str
) -> None:
    conn = q.db()
    job = enqueue_job_in_state(conn, state)
    assert q.cancel_job(conn, job) == message


def test_cancel_job_of_an_unknown_id(home: Path) -> None:
    assert q.cancel_job(q.db(), 99) == "job 99: not found"


def test_cancel_job_rereads_when_the_job_started_in_between(home: Path) -> None:
    conn = q.db()
    job = enqueue(conn)
    racing = ConnectionWithRaceHook(
        q.db(),
        "SET state='canceled'",
        lambda: set_state_from_another_connection(job, "running"),
    )
    assert (
        q.cancel_job(racing, job)  # type: ignore[arg-type]
        == f"job {job}: cancel requested (scheduler will SIGTERM)"
    )
    racing.commit()  # cmd_cancel commits after its loop; do the same
    assert state_of(conn, job) == "canceling"


@pytest.mark.parametrize("state", ["canceled", "backlog", "running", "done"])
def test_claim_refuses_jobs_that_are_not_queued(home: Path, state: str) -> None:
    sched = scheduler()
    job = enqueue_job_in_state(sched.conn, state)
    assert sched.claim(job, 0, home / "logs" / "x.log") is False
    assert state_of(sched.conn, job) == state


def test_claim_takes_a_queued_job_exactly_once(home: Path) -> None:
    sched = scheduler()
    job = enqueue(sched.conn)
    log = home / "logs" / f"{job}.log"
    assert sched.claim(job, 2, log) is True
    row = sched.conn.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
    assert (row["state"], row["gpu"], row["log_path"]) == ("running", 2, str(log))
    assert row["started_at"] is not None and row["pid"] is None
    assert sched.claim(job, 3, log) is False


@pytest.mark.parametrize("winner", ["canceled", "backlog"])
def test_dispatch_keeps_a_command_that_lands_before_the_claim(
    home: Path, monkeypatch: pytest.MonkeyPatch, winner: str
) -> None:
    sched = scheduler(gpus=[0])
    first, second = enqueue(sched.conn), enqueue(sched.conn)
    real_claim = Scheduler.claim

    def claim_after_a_concurrent_command(
        self: Scheduler, job_id: int, gpu: int, log_path: Path
    ) -> bool:
        if job_id == first:
            set_state_from_another_connection(first, winner)
        return real_claim(self, job_id, gpu, log_path)

    monkeypatch.setattr(Scheduler, "claim", claim_after_a_concurrent_command)
    sched.dispatch(time.time())
    assert state_of(sched.conn, first) == winner  # not overwritten with 'running'
    assert not (home / "logs" / f"{first}.log").exists()  # and never started
    assert set(sched.running) == {second}  # the same pass moved on to the next job
    run_to_completion(sched)
    assert state_of(sched.conn, second) == "done"


def test_cancel_request_between_claim_and_pid_update_survives(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sched = scheduler()
    job = enqueue(sched.conn, "sleep", "30")
    real_popen = q.subprocess.Popen

    def popen_then_a_concurrent_cancel(*args: object, **kwargs: object) -> object:
        proc = real_popen(*args, **kwargs)  # type: ignore[call-overload]
        set_state_from_another_connection(job, "canceling")
        return proc

    monkeypatch.setattr(q.subprocess, "Popen", popen_then_a_concurrent_cancel)
    try:
        sched.dispatch(time.time())
        row = sched.conn.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
        assert row["state"] == "canceling"  # the pid update left the state alone
        assert row["pid"] == sched.running[job].proc.pid
        sched.process_termination_requests(time.time())
        deadline = time.time() + 5.0
        while sched.running and time.time() < deadline:
            sched.reap()
            time.sleep(0.02)
        assert state_of(sched.conn, job) == "canceled"
    finally:
        sched.shutdown()


def test_spawn_of_a_job_that_lost_its_claim_does_nothing(home: Path) -> None:
    sched = scheduler()
    job = enqueue(sched.conn)
    stale_row = sched.conn.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
    set_state_from_another_connection(job, "canceled")
    assert sched.spawn(stale_row, 0) is False
    assert not sched.running
    assert not (home / "logs" / f"{job}.log").exists()


# --------------------------------------------------------------- drain estimate


@pytest.mark.parametrize(
    ("free_in", "queued", "duration", "expected"),
    [
        ([3600.0, 0.0, 0.0], 0, 100.0, 3600.0),  # bounded by the slowest running job
        ([0.0, 0.0, 0.0], 6, 100.0, 200.0),  # two full waves
        ([0.0, 0.0, 0.0], 7, 100.0, 300.0),  # a partial last wave still costs a wave
        ([50.0, 0.0], 2, 100.0, 150.0),  # the slot that frees first takes the job
        ([60.0, 0.0], 6, 100.0, 360.0),
        ([0.0], 0, 5.0, 0.0),
    ],
)
def test_simulate_drain_places_equal_jobs_on_the_slot_that_frees_first(
    free_in: list[float], queued: int, duration: float, expected: float
) -> None:
    assert q.simulate_drain(free_in, queued, duration) == pytest.approx(expected)


def test_simulate_drain_does_not_mutate_its_input() -> None:
    free_in = [10.0, 0.0]
    q.simulate_drain(free_in, 5, 1.0)
    assert free_in == [10.0, 0.0]


def test_typical_runtime_needs_three_samples(home: Path) -> None:
    conn = q.db()
    for _ in range(2):
        insert_done_job_with_wall_time(conn, 100.0)
    assert q.typical_runtime(conn) is None
    insert_done_job_with_wall_time(conn, 100.0)
    assert q.typical_runtime(conn) == pytest.approx(100.0)


def test_typical_runtime_is_the_median_not_the_mean(home: Path) -> None:
    conn = q.db()
    for wall in (100.0, 200.0, 300.0, 400.0, 100_000.0):
        insert_done_job_with_wall_time(conn, wall)
    assert q.typical_runtime(conn) == pytest.approx(300.0)


def test_typical_runtime_averages_the_middle_pair_for_even_counts(home: Path) -> None:
    conn = q.db()
    for wall in (100.0, 200.0, 300.0, 400.0):
        insert_done_job_with_wall_time(conn, wall)
    assert q.typical_runtime(conn) == pytest.approx(250.0)


def test_typical_runtime_only_looks_at_the_twenty_most_recent_jobs(
    home: Path,
) -> None:
    conn = q.db()
    for index in range(15):  # older history: all cheap
        insert_done_job_with_wall_time(conn, 30.0, finished_ago=1000.0 + index)
    for index in range(20):  # the window: 9 cheap, 11 expensive -> median 300
        wall = 300.0 if index >= 9 else 30.0
        insert_done_job_with_wall_time(conn, wall, finished_ago=1.0 + index)
    assert q.typical_runtime(conn) == pytest.approx(300.0)


def test_typical_runtime_ignores_jobs_that_are_not_cleanly_done(home: Path) -> None:
    conn = q.db()
    for _ in range(3):
        insert_done_job_with_wall_time(conn, 100.0)
    for state in ("failed", "canceled"):
        for _ in range(4):
            job = insert_done_job_with_wall_time(conn, 9999.0)
            conn.execute("UPDATE jobs SET state=? WHERE id=?", (state, job))
    never_started = insert_done_job_with_wall_time(conn, 9999.0)
    conn.execute("UPDATE jobs SET started_at=NULL WHERE id=?", (never_started,))
    conn.commit()
    assert q.typical_runtime(conn) == pytest.approx(100.0)


def test_typical_runtime_excludes_runs_shorter_than_twenty_seconds(home: Path) -> None:
    conn = q.db()
    for _ in range(3):
        insert_done_job_with_wall_time(conn, 100.0)
    for _ in range(5):  # instant skips: without the rule they would set the median
        insert_done_job_with_wall_time(conn, 1.0)
    assert q.typical_runtime(conn) == pytest.approx(100.0)


def test_typical_runtime_threshold_is_exactly_twenty_seconds(home: Path) -> None:
    conn = q.db()
    for _ in range(3):
        insert_done_job_between(conn, 1000.0, 1020.0)  # exactly 20 s: counts
    assert q.typical_runtime(conn) == 20.0
    for _ in range(3):
        insert_done_job_between(conn, 2000.0, 2019.99)  # just under: ignored
    assert q.typical_runtime(conn) == 20.0


def test_typical_runtime_needs_three_runs_that_qualify(home: Path) -> None:
    conn = q.db()
    for _ in range(2):
        insert_done_job_with_wall_time(conn, 100.0)
    for _ in range(10):
        insert_done_job_with_wall_time(conn, 1.0)
    assert q.typical_runtime(conn) is None


def test_typical_runtime_window_is_filled_with_runs_that_qualify(home: Path) -> None:
    """Skips are dropped before the window is cut, so they cannot crowd out runs."""
    conn = q.db()
    for index in range(10):  # older, expensive
        insert_done_job_with_wall_time(conn, 1000.0, finished_ago=500.0 + index)
    for index in range(10):  # newer, cheaper
        insert_done_job_with_wall_time(conn, 100.0, finished_ago=100.0 + index)
    for index in range(15):  # the newest of all: instant skips
        insert_done_job_with_wall_time(conn, 1.0, finished_ago=1.0 + index)
    # the 20 qualifying runs are ten 100 s and ten 1000 s: median (100 + 1000) / 2
    assert q.typical_runtime(conn) == pytest.approx(550.0)


def test_estimate_drain_ignores_instant_skips_when_judging_cost(home: Path) -> None:
    conn = q.db()
    for _ in range(3):
        insert_done_job_with_wall_time(conn, 100.0)
    for _ in range(5):
        insert_done_job_with_wall_time(conn, 1.0)
    for _ in range(4):
        enqueue(conn)
    q.ctl_set(conn, "gpus", "0,1")
    assert q.estimate_drain_seconds(conn, time.time()) == pytest.approx(200.0)


def test_estimate_drain_is_unavailable_when_every_finished_run_was_instant(
    home: Path,
) -> None:
    conn = q.db()
    for _ in range(10):
        insert_done_job_with_wall_time(conn, 5.0)
    enqueue(conn)
    q.ctl_set(conn, "gpus", "0")
    assert q.estimate_drain_seconds(conn, time.time()) is None
    assert "drain" not in q.render_status(False, conn).splitlines()[0]


def test_estimate_drain_is_unavailable_without_history(home: Path) -> None:
    conn = q.db()
    enqueue(conn)
    q.ctl_set(conn, "gpus", "0,1")
    assert q.estimate_drain_seconds(conn, time.time()) is None


def test_estimate_drain_is_unavailable_for_an_idle_queue(home: Path) -> None:
    conn = q.db()
    for _ in range(3):
        insert_done_job_with_wall_time(conn, 100.0)
    q.ctl_set(conn, "gpus", "0,1")
    assert q.estimate_drain_seconds(conn, time.time()) is None


def test_estimate_drain_is_unavailable_when_no_gpus_are_known(home: Path) -> None:
    conn = q.db()
    for _ in range(3):
        insert_done_job_with_wall_time(conn, 100.0)
    enqueue(conn)
    assert q.estimate_drain_seconds(conn, time.time()) is None


def test_estimate_drain_spreads_queued_jobs_over_the_gpus(home: Path) -> None:
    conn = q.db()
    for _ in range(3):
        insert_done_job_with_wall_time(conn, 100.0)
    for _ in range(6):
        enqueue(conn)
    q.ctl_set(conn, "gpus", "0,1,2")
    assert q.estimate_drain_seconds(conn, time.time()) == pytest.approx(200.0)


def test_estimate_drain_excludes_the_backlog(home: Path) -> None:
    conn = q.db()
    for _ in range(3):
        insert_done_job_with_wall_time(conn, 100.0)
    enqueue(conn)
    for _ in range(10):
        enqueue_job_in_state(conn, "backlog")
    q.ctl_set(conn, "gpus", "0,1")
    assert q.estimate_drain_seconds(conn, time.time()) == pytest.approx(100.0)


def test_estimate_drain_counts_running_jobs_down_from_the_median(home: Path) -> None:
    conn = q.db()
    for _ in range(3):
        insert_done_job_with_wall_time(conn, 100.0)
    now = time.time()
    insert_running_job_started_ago(conn, 40.0)  # 60 s left by the median
    for _ in range(6):
        enqueue(conn)
    q.ctl_set(conn, "gpus", "0,1")
    assert q.estimate_drain_seconds(conn, now) == pytest.approx(360.0, abs=1.0)


def test_estimate_drain_prefers_the_tqdm_eta_of_a_running_job(home: Path) -> None:
    conn = q.db()
    for _ in range(3):
        insert_done_job_with_wall_time(conn, 100.0)
    log = home / "bar.log"
    log.write_text("50%|#####     | 5/10 [00:10<00:10, 1.00it/s]")
    insert_running_job_started_ago(conn, 40.0, log_path=log)
    for _ in range(6):
        enqueue(conn)
    q.ctl_set(conn, "gpus", "0,1")
    assert q.estimate_drain_seconds(conn, time.time()) == pytest.approx(310.0, abs=1.0)


def test_estimate_drain_with_an_empty_queue_is_the_slowest_running_job(
    home: Path,
) -> None:
    conn = q.db()
    for _ in range(3):
        insert_done_job_with_wall_time(conn, 100.0)
    log = home / "bar.log"
    log.write_text("10%|#         | 1/10 [00:10<1:00:00, 0.10it/s]")
    insert_running_job_started_ago(conn, 10.0, log_path=log)
    q.ctl_set(conn, "gpus", "0,1,2")
    assert q.estimate_drain_seconds(conn, time.time()) == pytest.approx(3600.0)


def test_estimate_drain_never_goes_negative_for_an_overrunning_job(home: Path) -> None:
    conn = q.db()
    for _ in range(3):
        insert_done_job_with_wall_time(conn, 100.0)
    insert_running_job_started_ago(conn, 500.0)  # long past the median
    q.ctl_set(conn, "gpus", "0")
    assert q.estimate_drain_seconds(conn, time.time()) == pytest.approx(0.0)


def test_status_tally_line_carries_the_drain_estimate(home: Path) -> None:
    conn = q.db()
    for _ in range(3):
        insert_done_job_with_wall_time(conn, 600.0)
    for _ in range(4):
        enqueue(conn)
    q.ctl_set(conn, "gpus", "0,1")
    first_line = q.render_status(False, conn).splitlines()[0]
    assert first_line.endswith("  ·  drain ~20m")  # 2 waves of 600 s


def test_status_tally_line_has_no_estimate_without_history(home: Path) -> None:
    conn = q.db()
    enqueue(conn)
    q.ctl_set(conn, "gpus", "0")
    assert "drain" not in q.render_status(False, conn).splitlines()[0]


def test_status_tally_function_and_digest_subject_exclude_the_estimate(
    home: Path,
) -> None:
    sched = scheduler(gpus=[0, 1])
    for _ in range(3):
        insert_done_job_with_wall_time(sched.conn, 600.0)
    enqueue(sched.conn)
    q.ctl_set(sched.conn, "gpus", "0,1")
    assert "drain" not in q.status_tally(sched.conn)
    sched.digest_due = time.time() - 1
    sched.maybe_digest(time.time())
    assert "drain" not in NOTIFICATIONS[0]
    assert "drain ~" in BODIES[0]  # the body is the full status view


# ---------------------------------------------------------------------- q logs


@pytest.mark.parametrize("text", ["0", "1", "50"])
def test_parse_line_count_accepts_counts(text: str) -> None:
    assert q.parse_line_count(text) == int(text)


def test_parse_line_count_accepts_all() -> None:
    assert q.parse_line_count("all") == q.ALL_LINES


@pytest.mark.parametrize("text", ["-3", "abc", "3.5", "", "ALL"])
def test_parse_line_count_rejects_everything_else(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="expected a line count"):
        q.parse_line_count(text)


@pytest.mark.parametrize("block_bytes", [1, 2, 3, 5, 64])
def test_tail_start_matches_the_splitlines_reference(
    monkeypatch: pytest.MonkeyPatch, block_bytes: int
) -> None:
    """Tiny blocks force every boundary case, including a CRLF pair split in two."""
    monkeypatch.setattr(q, "TAIL_BLOCK_BYTES", block_bytes)
    rng = random.Random(block_bytes)
    for _ in range(1500):
        data = bytes(rng.choice(b"ab\r\n") for _ in range(rng.randint(0, 40)))
        count = rng.randint(0, 12)
        expected = b"".join(data.splitlines(keepends=True)[-count:]) if count else b""
        assert data[q.tail_start(io.BytesIO(data), count) :] == expected, (data, count)


@pytest.mark.parametrize(
    ("data", "count", "expected"),
    [
        (b"", 5, b""),
        (b"a\nb\n", 10, b"a\nb\n"),
        (b"a\nb\nc", 2, b"b\nc"),
        (b"a\nb\nc\n", 2, b"b\nc\n"),  # a final newline ends a line, it is not one
        (b"a\n\n\nb\n", 2, b"\nb\n"),  # blank lines count
        (b"a\nb\n", 0, b""),
        (b"1%\r2%\r3%\r4%", 2, b"3%\r4%"),  # tqdm redraws are line breaks
        (b"a\r\nb\r\nc\r\n", 2, b"b\r\nc\r\n"),  # CRLF counts once
    ],
)
def test_tail_start_hand_picked_cases(data: bytes, count: int, expected: bytes) -> None:
    assert data[q.tail_start(io.BytesIO(data), count) :] == expected


def test_logs_without_options_prints_everything(
    home: Path, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    job = insert_job_with_log_bytes(home, numbered_log_lines(200))
    q.cmd_logs(job, follow=False)
    assert capsysbinary.readouterr().out == numbered_log_lines(200)


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (3, [b"line 198", b"line 199", b"line 200"]),
        (0, []),
        (9999, [f"line {i}".encode() for i in range(1, 201)]),
        (q.ALL_LINES, [f"line {i}".encode() for i in range(1, 201)]),
    ],
)
def test_logs_n_selects_the_tail(
    home: Path,
    capsysbinary: pytest.CaptureFixture[bytes],
    lines: int,
    expected: list[bytes],
) -> None:
    job = insert_job_with_log_bytes(home, numbered_log_lines(200))
    q.cmd_logs(job, follow=False, lines=lines)
    assert capsysbinary.readouterr().out.splitlines() == expected


def test_logs_follow_starts_at_the_last_fifty_lines_then_follows(
    home: Path,
    capsysbinary: pytest.CaptureFixture[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = insert_job_with_log_bytes(home, numbered_log_lines(200))
    log = home / "logs" / f"{job}.log"
    sleeps: list[float] = []

    def append_once_then_stop(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) > 1:
            raise KeyboardInterrupt
        with log.open("ab") as handle:
            handle.write(b"late arrival\n")

    monkeypatch.setattr(q.time, "sleep", append_once_then_stop)
    q.cmd_logs(job, follow=True)
    out = capsysbinary.readouterr().out.splitlines()
    assert out[:-1] == [f"line {i}".encode() for i in range(151, 201)]
    assert out[-1] == b"late arrival"
    assert len(out) == q.FOLLOW_TAIL_LINES + 1


def test_logs_follow_with_zero_lines_shows_only_new_output(
    home: Path,
    capsysbinary: pytest.CaptureFixture[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = insert_job_with_log_bytes(home, numbered_log_lines(200))
    log = home / "logs" / f"{job}.log"
    sleeps: list[float] = []

    def append_once_then_stop(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) > 1:
            raise KeyboardInterrupt
        with log.open("ab") as handle:
            handle.write(b"only new\n")

    monkeypatch.setattr(q.time, "sleep", append_once_then_stop)
    q.cmd_logs(job, follow=True, lines=0)
    assert capsysbinary.readouterr().out == b"only new\n"


def test_logs_follow_with_all_replays_everything_first(
    home: Path,
    capsysbinary: pytest.CaptureFixture[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = insert_job_with_log_bytes(home, numbered_log_lines(200))

    def stop_following(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(q.time, "sleep", stop_following)
    q.cmd_logs(job, follow=True, lines=q.ALL_LINES)
    assert capsysbinary.readouterr().out == numbered_log_lines(200)


def test_logs_tail_of_a_progress_bar_log_counts_carriage_returns(
    home: Path, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    bars = b"".join(f"{i}%|#| {i}/1000 [00:01<00:02]\r".encode() for i in range(1000))
    job = insert_job_with_log_bytes(home, b"start\n" + bars + b"done\n")
    q.cmd_logs(job, follow=False, lines=3)
    assert capsysbinary.readouterr().out == (
        b"998%|#| 998/1000 [00:01<00:02]\r999%|#| 999/1000 [00:01<00:02]\rdone\n"
    )


def test_logs_of_a_job_without_a_log_exits(home: Path) -> None:
    job = enqueue(q.db())
    with pytest.raises(SystemExit, match=f"no log for job {job}"):
        q.cmd_logs(job, follow=False)


def test_main_logs_passes_the_line_count_through(
    home: Path, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    job = insert_job_with_log_bytes(home, numbered_log_lines(10))
    q.main(["logs", str(job), "-n", "2"])
    assert capsysbinary.readouterr().out.splitlines() == [b"line 9", b"line 10"]
    q.main(["logs", str(job), "--lines", "all"])
    assert len(capsysbinary.readouterr().out.splitlines()) == 10


def test_main_logs_rejects_a_negative_line_count(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = insert_job_with_log_bytes(home, b"x\n")
    with pytest.raises(SystemExit, match="2"):
        q.main(["logs", str(job), "-n", "-3"])
    assert "expected a line count or 'all'" in capsys.readouterr().err


# ---------------------------------------------------------- validation confirmation


@pytest.mark.parametrize("default", [True, False])
def test_prompt_confirm_returns_the_default_when_the_timeout_expires(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    default: bool,
) -> None:
    monkeypatch.setattr(q.select, "select", lambda r, w, x, t: ([], [], []))
    assert q.prompt_confirm("go?", 0.01, default=default) is default
    assert "timed out, defaulting to" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("answer", "default", "expected"),
    [
        ("", True, True),  # a closed stdin
        ("", False, False),
        ("\n", True, True),  # just Enter
        ("\n", False, False),
        ("y\n", False, True),
        ("YES\n", False, True),
        ("n\n", True, False),
        ("maybe\n", True, False),
    ],
)
def test_prompt_confirm_interprets_the_answer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: str,
    default: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(q.select, "select", lambda r, w, x, t: (r, [], []))
    monkeypatch.setattr(q.sys, "stdin", io.StringIO(answer))
    assert q.prompt_confirm("go?", 1.0, default=default) is expected
    capsys.readouterr()


def test_validated_combos_ask_defaults_to_no_when_nobody_answers(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(q.select, "select", lambda r, w, x, t: ([], [], []))
    with pytest.raises(SystemExit, match="nothing enqueued"):
        validated_combos([["sh", "-c", VALID_CMD], ["false"]], {}, "ask")


def test_validated_combos_ask_enqueues_the_survivors_on_yes(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(q.select, "select", lambda r, w, x, t: (r, [], []))
    monkeypatch.setattr(q.sys, "stdin", io.StringIO("y\n"))
    combos = [["sh", "-c", VALID_CMD], ["false"]]
    assert validated_combos(combos, {}, "ask") == [combos[0]]


def test_validated_combos_abort_enqueues_nothing(home: Path) -> None:
    with pytest.raises(SystemExit, match="1 rejected"):
        validated_combos([["sh", "-c", VALID_CMD], ["false"]], {}, "abort")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
