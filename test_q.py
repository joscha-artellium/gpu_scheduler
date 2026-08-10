#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""Unit tests for qsched. Run: ./test_q.py  (or via pytest)"""

import fcntl
import shlex
import sqlite3
import time
from collections.abc import Iterator
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

SENTINEL_CMD = f"echo {q.VALIDATE_SENTINEL}"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("QSCHED_HOME", str(tmp_path))
    monkeypatch.setenv("QSCHED_NOTIFY", str(tmp_path / "absent.sh"))

    def record(title: str, body: str, urgency: str = "normal") -> None:
        NOTIFICATIONS.append(title)
        URGENCIES.append(urgency)

    monkeypatch.setattr(q, "notify", record)
    NOTIFICATIONS.clear()
    URGENCIES.clear()
    yield tmp_path


NOTIFICATIONS: list[str] = []
URGENCIES: list[str] = []


def enqueue(conn: sqlite3.Connection, *argv: str) -> int:
    return q.insert_job(conn, list(argv) or ["true"], {})


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
        sched.check_drain(now)
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
    with pytest.raises(ValueError):
        expand_sweep(["key=a,,b"])
    with pytest.raises(ValueError):
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


def test_restart_requires_ids_or_all(home: Path) -> None:
    conn = q.db()
    enqueue(conn, "true")
    with pytest.raises(SystemExit):
        q.cmd_restart([], restart_all=False)
    with pytest.raises(SystemExit):
        q.cmd_restart([1], restart_all=True)


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
    sched.check_drain(time.time())
    assert NOTIFICATIONS == ["qsched: queue drained — 1 done, 1 failed"]
    sched.check_drain(time.time())  # fires once per batch
    assert len(NOTIFICATIONS) == 1


def test_gpu_env_and_cwd_reach_the_job(home: Path, tmp_path: Path) -> None:
    sched = scheduler(gpus=[3])
    workdir = tmp_path / "work"
    workdir.mkdir()
    conn = sched.conn
    job = q.insert_job(
        conn, ["sh", "-c", "echo $CUDA_VISIBLE_DEVICES $REGION $PWD"], {"REGION": "US3"}
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
        with pytest.raises(SystemExit):
            q.cmd_clear(wipe_all=True, older_than_days=None, dry_run=False)
    assert q.db().execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 1


def test_show_reports_env_and_cwd(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    conn = q.db()
    job = q.insert_job(conn, ["python", "train.py", "model=a"], {"REGION": "US3"})
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


def test_only_the_halt_is_critical(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(q, "HALT_AFTER", 2)
    monkeypatch.setattr(q, "PAUSE_SECONDS", 0.0)
    sched = scheduler()
    for _ in range(2):
        job = enqueue(sched.conn, "false")
        sched.finalize(job, 1, home / "logs" / f"{job}.log")
    assert URGENCIES == ["normal", "critical"]


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


def test_requeue_failed_resets_retries_and_goes_to_the_back(home: Path) -> None:
    conn = q.db()
    failed, queued = enqueue(conn, "false"), enqueue(conn, "true")
    conn.execute(
        "UPDATE jobs SET state='failed', retries=1, exit_code=1, finished_at=? "
        "WHERE id=?",
        (time.time(), failed),
    )
    conn.commit()
    q.cmd_requeue_failed()
    row = q.db().execute("SELECT * FROM jobs WHERE id=?", (failed,)).fetchone()
    assert (row["state"], row["retries"], row["exit_code"]) == ("queued", 0, None)
    order = [
        int(r["id"]) for r in q.db().execute("SELECT id FROM jobs ORDER BY rank, id")
    ]
    assert order == [queued, failed]


def test_requeue_failed_spares_canceled_and_keeps_a_halt(home: Path) -> None:
    conn = q.db()
    canceled = enqueue(conn, "true")
    conn.execute("UPDATE jobs SET state='canceled' WHERE id=?", (canceled,))
    conn.commit()
    q.ctl_set(conn, "halted", "1")
    q.cmd_requeue_failed()
    assert state_of(q.db(), canceled) == "canceled"
    assert q.ctl_get(q.db(), "halted", "0") == "1"  # requeueing is not looking


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
    with pytest.raises(SystemExit):
        q.parse_env_pairs(["NOEQUALS"])
    with pytest.raises(SystemExit):
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
    job = q.insert_job(sched.conn, ["definitely-not-a-real-binary"], {})
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


def test_probe_passes_only_with_the_sentinel(home: Path) -> None:
    good = run_probe(["sh", "-c", SENTINEL_CMD], {}, str(home))
    assert (good.status, good.ok) == ("pass", True)
    quiet = run_probe(["sh", "-c", "echo nothing to say"], {}, str(home))
    assert quiet.status == "unsupported"  # never silently a pass
    assert "nothing to say" in quiet.output


def test_probe_reports_nonzero_exit_with_output(home: Path) -> None:
    result = run_probe(["sh", "-c", "echo bad config >&2; exit 3"], {}, str(home))
    assert (result.status, result.reason) == ("failed", "exit 3")
    assert "bad config" in result.output


def test_probe_times_out(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(q, "VALIDATE_TIMEOUT", 0.3)
    result = run_probe(["sh", "-c", f"{SENTINEL_CMD}; sleep 30"], {}, str(home))
    assert result.status == "timeout"


def test_probe_spawn_error(home: Path) -> None:
    assert run_probe(["definitely-not-a-real-binary"], {}, str(home)).status == "error"


def test_probe_env_hides_gpus_and_carries_job_env(home: Path) -> None:
    argv = [
        "sh",
        "-c",
        f'echo "V=$QSCHED_VALIDATE CUDA=[$CUDA_VISIBLE_DEVICES] R=$REGION P=$PWD"; '
        f"{SENTINEL_CMD}; exit 1",
    ]
    result = run_probe(argv, {"REGION": "US3"}, str(home))
    assert f"V=1 CUDA=[] R=US3 P={home}" in result.output


def test_probes_keep_combo_order_when_parallel(home: Path) -> None:
    combos = [
        ["sh", "-c", f"sleep 0.3; {SENTINEL_CMD}"],
        ["sh", "-c", "exit 1"],
        ["sh", "-c", SENTINEL_CMD],
    ]
    results = run_probes(combos, {}, workers=3)
    assert [r.status for r in results] == ["pass", "failed", "pass"]


def test_validated_combos_all_bad_aborts(home: Path) -> None:
    with pytest.raises(SystemExit):
        validated_combos([["false"], ["false"]], {}, workers=2, assume_yes=True)


def test_validated_combos_keeps_survivors_with_yes(home: Path) -> None:
    combos = [["sh", "-c", SENTINEL_CMD], ["false"]]
    assert validated_combos(combos, {}, workers=2, assume_yes=True) == [combos[0]]


def test_validated_combos_declined_enqueues_nothing(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(q, "prompt_yes", lambda question, timeout: False)
    with pytest.raises(SystemExit):
        validated_combos(
            [["sh", "-c", SENTINEL_CMD], ["false"]], {}, workers=2, assume_yes=False
        )


def test_add_with_validate_enqueues_nothing_on_rejection(home: Path) -> None:
    with pytest.raises(SystemExit):
        q.cmd_add([], ["false"], validate=True, assume_yes=True)
    assert q.db().execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 0


def test_sweep_with_validate_enqueues_only_survivors(home: Path) -> None:
    script = f'test "$1" = "V=ok" && {SENTINEL_CMD}'
    command = ["sh", "-c", script, "probe", "V=ok,bad"]
    q.cmd_sweep([], command, dry_run=False, validate=True, assume_yes=True)
    rows = q.db().execute("SELECT argv FROM jobs").fetchall()
    assert len(rows) == 1 and "V=ok" in rows[0]["argv"]


def test_sweep_dry_run_with_validate_reports_and_exits_nonzero(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        q.cmd_sweep([], ["false"], dry_run=True, validate=True)
    captured = capsys.readouterr()
    assert captured.out.strip() == "false"  # stdout stays paste-able
    assert "0 passed, 1 rejected" in captured.err
    assert q.db().execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
