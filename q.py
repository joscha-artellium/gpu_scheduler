#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""qsched — minimal GPU job queue for a single machine.

One SQLite database holds the queue; `q run` is a foreground scheduler that
dispatches queued jobs onto free GPUs (one GPU per job, exact bookkeeping, no
nvidia-smi heuristics). The CLI works whether or not the scheduler is running.

Commands:
    q add   [--env K=V ...] -- <command ...>     enqueue one job
    q sweep [--env K=V ...] [-n] -- <command ...> expand hydra-style sweeps,
                                                  enqueue one job per combo
    q run   [--gpus 0,1,2]                        start the scheduler
    q status [--all]                              show the queue
    q logs <id> [-f]                              show/follow a job's log
    q cancel <id> [...]                           cancel queued/running jobs
    q fixed <id> [...]                            "I fixed it": retry at FRONT of queue
    q extend [minutes]                            extend the failure pause (default 5)
    q resume                                      end the failure pause / halt now
    q requeue-failed                              re-enqueue all failed jobs (back)

QUOTING RULES
=============
`q` executes commands directly — there is NO second shell. Your interactive
shell strips exactly one layer of quoting when you type `q add ...`; whatever
lands in q's argv is stored verbatim and later passed byte-for-byte to
execvp(). Rule of thumb: quote exactly as you would when invoking the script
directly, no extra escaping.

`q add` never interprets anything after `--`.

`q sweep` splits each token of the form KEY=VALUE on *top-level* commas —
commas outside (), [], {}, single and double quotes — into sweep variants,
then enqueues the cartesian product over all tokens:

    'training_window="96"'                               -> 1 variant
    'lr=0.1,0.01'                                        -> 2 variants
    'features_transform.grouping=["Day","key"],["Day"]'  -> 2 variants
    'key="a,b"'                                          -> 1 variant (quoted comma)
    'model=xgb01,xgb02' 'lr=0.1,0.01'                    -> 4 jobs

Tokens starting with '-' and tokens without '=' are never split. Backslash
escapes are not understood by the splitter — use quotes. Hydra's range()/
glob()/choice() sweeps are not expanded; do that expansion in your submission
script. Use `q sweep -n -- ...` (dry run) to inspect the expansion.

Environment variables go through --env (repeatable), never shell prefixes:

    q add --env REGION=US3 -- uv run python scripts/train_predict.py model=xgb01

Need pipes or &&?  q add -- bash -c '<script>'

The submission-time working directory is recorded and jobs run in it.

FAILURE SEMANTICS
=================
A job exiting nonzero is immediately given the unattended default (its GPU
frees up): if it has never had an unattended retry it is requeued at the BACK
of the queue with retries=1; otherwise it is marked `failed` permanently.
You are notified (hook/notify-send/mail) and dispatch of NEW jobs pauses for
QSCHED_PAUSE_SECONDS (default 300) — running jobs continue. During (or after)
the pause you can override the default:

  1. `q fixed <id> [...]` — you fixed it: the job moves to the FRONT of the
     queue, regardless of retry/failure count (the count is NOT reset — a
     fixed job that fails unattended again follows the default for its
     count). Also clears any pause/halt.
  2. `q extend [minutes]` — buy more time before dispatch resumes.
  3. Do nothing — the pause expires and dispatch resumes; the default
     disposition is already in place.

Pauses do not stack; if QSCHED_HALT_AFTER (default 3) failures land within
one window the pause becomes indefinite (halt) until `q fixed`/`q resume`.
Cancels never count as failures.

On scheduler shutdown (Ctrl-C / SIGTERM / SIGHUP) running jobs are terminated
and requeued in place — with an idempotent framework a rerun skips completed
work. On startup, stale `running` rows are requeued after killing verified
orphans.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from shutil import which
from typing import IO, Any

QSCHED_HOME = Path(os.environ.get("QSCHED_HOME", str(Path.home() / ".local/share/qsched")))
DB_PATH = QSCHED_HOME / "queue.db"
LOG_DIR = QSCHED_HOME / "logs"
NOTIFY_HOOK = Path(os.environ.get("QSCHED_NOTIFY", str(Path.home() / ".config/qsched/notify.sh")))

PAUSE_SECONDS = float(os.environ.get("QSCHED_PAUSE_SECONDS", "300"))
HALT_AFTER = int(os.environ.get("QSCHED_HALT_AFTER", "3"))
POLL_SECONDS = float(os.environ.get("QSCHED_POLL_SECONDS", "2"))
CANCEL_GRACE_SECONDS = 10.0

ACTIVE_STATES = ("queued", "running", "canceling")


# --------------------------------------------------------------------------- db

def db() -> sqlite3.Connection:
    QSCHED_HOME.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            argv TEXT NOT NULL,
            env TEXT NOT NULL DEFAULT '{}',
            cwd TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'queued',
            rank REAL NOT NULL,
            gpu INTEGER,
            pid INTEGER,
            retries INTEGER NOT NULL DEFAULT 0,
            exit_code INTEGER,
            submitted_at REAL NOT NULL,
            started_at REAL,
            finished_at REAL,
            log_path TEXT
        )"""
    )
    conn.execute("CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.commit()
    return conn


def front_rank(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT COALESCE(MIN(rank), 0) - 1 AS r FROM jobs").fetchone()
    return float(row["r"])


def back_rank(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT COALESCE(MAX(rank), 0) + 1 AS r FROM jobs").fetchone()
    return float(row["r"])


def ctl_get(conn: sqlite3.Connection, key: str, default: str) -> str:
    row = conn.execute("SELECT value FROM control WHERE key=?", (key,)).fetchone()
    return str(row["value"]) if row else default


def ctl_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO control(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


# ----------------------------------------------------------------- sweep expansion

OVERRIDE_RE = re.compile(r"^(?P<key>[+~]{0,2}[\w.@/:]+)=(?P<value>.*)$", re.S)
OPENERS, CLOSERS = "([{", ")]}"


def split_top_level(value: str) -> list[str]:
    """Split on commas that are outside all brackets and quotes."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: str | None = None
    for ch in value:
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in OPENERS:
            depth += 1
        elif ch in CLOSERS:
            depth = max(depth - 1, 0)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf.clear()
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return parts


def expand_sweep(argv: list[str]) -> list[list[str]]:
    """Cartesian product over top-level comma variants of KEY=VALUE tokens."""
    per_token_variants: list[list[str]] = []
    for token in argv:
        match = None if token.startswith("-") else OVERRIDE_RE.match(token)
        if match is None:
            per_token_variants.append([token])
            continue
        variants = split_top_level(match["value"])
        if any(v.strip() == "" for v in variants) and len(variants) > 1:
            raise ValueError(f"empty sweep variant in token: {token!r}")
        per_token_variants.append([f"{match['key']}={v}" for v in variants])
    return [list(combo) for combo in product(*per_token_variants)]


# -------------------------------------------------------------------- submission

def parse_env_pairs(pairs: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--env expects KEY=VALUE, got {pair!r}")
        env[key] = value
    return env


def insert_job(conn: sqlite3.Connection, argv: list[str], env: dict[str, str]) -> int:
    cur = conn.execute(
        "INSERT INTO jobs(argv, env, cwd, submitted_at, rank) "
        "VALUES(?,?,?,?, (SELECT COALESCE(MAX(rank),0)+1 FROM jobs))",
        (json.dumps(argv), json.dumps(env), os.getcwd(), time.time()),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


def resolve_failure_default(conn: sqlite3.Connection, job_id: int, exit_code: int, now: float) -> str:
    """Unattended default, applied eagerly at failure time: back once, then failed."""
    row = conn.execute("SELECT retries FROM jobs WHERE id=?", (job_id,)).fetchone()
    if int(row["retries"]) == 0:
        conn.execute(
            "UPDATE jobs SET state='queued', retries=1, rank=?, exit_code=?, gpu=NULL, "
            "pid=NULL, started_at=NULL, finished_at=NULL WHERE id=?",
            (back_rank(conn), exit_code, job_id),
        )
        outcome = "requeued at back"
    else:
        conn.execute(
            "UPDATE jobs SET state='failed', exit_code=?, finished_at=?, gpu=NULL, pid=NULL "
            "WHERE id=?",
            (exit_code, now, job_id),
        )
        outcome = "failed permanently"
    conn.commit()
    return outcome


def clear_pause(conn: sqlite3.Connection) -> None:
    ctl_set(conn, "pause_until", "0")
    ctl_set(conn, "halted", "0")
    ctl_set(conn, "window_failures", "0")


def cmd_add(env_pairs: list[str], command: list[str]) -> None:
    if not command:
        raise SystemExit("no command given (usage: q add [--env K=V] -- cmd ...)")
    conn = db()
    job_id = insert_job(conn, command, parse_env_pairs(env_pairs))
    print(f"enqueued job {job_id}: {shlex.join(command)}")


def cmd_sweep(env_pairs: list[str], command: list[str], dry_run: bool) -> None:
    if not command:
        raise SystemExit("no command given (usage: q sweep [--env K=V] [-n] -- cmd ...)")
    combos = expand_sweep(command)
    if dry_run:
        for combo in combos:
            print(shlex.join(combo))
        print(f"-- dry run: {len(combos)} job(s), nothing enqueued", file=sys.stderr)
        return
    conn = db()
    env = parse_env_pairs(env_pairs)
    ids = [insert_job(conn, combo, env) for combo in combos]
    print(f"enqueued {len(ids)} job(s): ids {ids[0]}..{ids[-1]}")


# ------------------------------------------------------------------ notifications

def notify(title: str, body: str) -> None:
    try:
        if NOTIFY_HOOK.is_file() and os.access(NOTIFY_HOOK, os.X_OK):
            subprocess.run([str(NOTIFY_HOOK), title, body], timeout=30, check=False)
            return
        if which("notify-send"):
            subprocess.run(["notify-send", "-u", "critical", title, body], timeout=10, check=False)
        email = os.environ.get("QSCHED_EMAIL")
        if email and which("mail"):
            subprocess.run(["mail", "-s", title, email], input=body.encode(), timeout=30, check=False)
    except Exception as exc:  # notification failure must never take down dispatch
        print(f"[qsched] notify failed: {exc}", file=sys.stderr)


def log_tail(path: Path, max_bytes: int = 800) -> str:
    try:
        data = path.read_bytes()[-max_bytes:]
        return data.decode(errors="replace").strip()
    except OSError:
        return "<no log>"


# --------------------------------------------------------------------- scheduler

@dataclass
class RunningJob:
    job_id: int
    gpu: int
    proc: subprocess.Popen[bytes]
    log_file: IO[bytes]
    log_path: Path
    kill_deadline: float | None = None


@dataclass
class Scheduler:
    conn: sqlite3.Connection
    gpus: list[int]
    running: dict[int, RunningJob] = field(default_factory=dict)
    stop_requested: bool = False

    # -- helpers ------------------------------------------------------------
    def event(self, message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def paused(self, now: float) -> bool:
        if ctl_get(self.conn, "halted", "0") == "1":
            return True
        return now < float(ctl_get(self.conn, "pause_until", "0"))

    # -- startup ------------------------------------------------------------
    def cleanup_stale(self) -> None:
        rows = self.conn.execute(
            "SELECT * FROM jobs WHERE state IN ('running','canceling')"
        ).fetchall()
        for row in rows:
            pid = row["pid"]
            if pid and _pid_matches(int(pid), json.loads(row["argv"])):
                self.event(f"killing orphan of job {row['id']} (pid {pid})")
                _kill_process_group(int(pid))
            new_state = "canceled" if row["state"] == "canceling" else "queued"
            self.conn.execute(
                "UPDATE jobs SET state=?, gpu=NULL, pid=NULL, started_at=NULL WHERE id=?",
                (new_state, row["id"]),
            )
            if new_state == "queued":
                self.event(f"requeued stale job {row['id']}")
        self.conn.commit()

    # -- dispatch -----------------------------------------------------------
    def spawn(self, row: sqlite3.Row, gpu: int) -> None:
        job_id = int(row["id"])
        argv: list[str] = json.loads(row["argv"])
        job_env: dict[str, str] = json.loads(row["env"])
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{job_id}.log"
        log_file: IO[bytes] = open(log_path, "ab", buffering=0)
        env = {**os.environ, **job_env, "CUDA_VISIBLE_DEVICES": str(gpu)}
        header = f"== qsched job {job_id} on gpu {gpu} :: {shlex.join(argv)}\n"
        log_file.write(header.encode())
        try:
            proc = subprocess.Popen(
                argv,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=row["cwd"],
                start_new_session=True,
            )
        except OSError as exc:
            log_file.write(f"== spawn failed: {exc}\n".encode())
            log_file.close()
            self.conn.execute(
                "UPDATE jobs SET state='running', gpu=?, started_at=?, log_path=? WHERE id=?",
                (gpu, time.time(), str(log_path), job_id),
            )
            self.conn.commit()
            self.finalize(job_id, exit_code=127, was_canceling=False, log_path=log_path)
            return
        self.conn.execute(
            "UPDATE jobs SET state='running', gpu=?, pid=?, started_at=?, log_path=? WHERE id=?",
            (gpu, proc.pid, time.time(), str(log_path), job_id),
        )
        self.conn.commit()
        self.running[job_id] = RunningJob(job_id, gpu, proc, log_file, log_path)
        self.event(f"job {job_id} started on gpu {gpu}: {shlex.join(argv)}")

    def dispatch(self, now: float) -> None:
        busy = {rj.gpu for rj in self.running.values()}
        for gpu in self.gpus:
            if gpu in busy:
                continue
            if self.paused(time.time()):  # re-check: a spawn failure can pause mid-pass
                return
            row = self.conn.execute(
                "SELECT * FROM jobs WHERE state='queued' ORDER BY rank, id LIMIT 1"
            ).fetchone()
            if row is None:
                return
            self.spawn(row, gpu)
            busy.add(gpu)

    # -- reaping / failure path ----------------------------------------------
    def finalize(self, job_id: int, exit_code: int, was_canceling: bool, log_path: Path) -> None:
        now = time.time()
        if was_canceling:
            self.conn.execute(
                "UPDATE jobs SET state='canceled', exit_code=?, finished_at=?, pid=NULL WHERE id=?",
                (exit_code, now, job_id),
            )
            self.conn.commit()
            self.event(f"job {job_id} canceled")
            return
        if exit_code == 0:
            self.conn.execute(
                "UPDATE jobs SET state='done', exit_code=0, finished_at=?, pid=NULL WHERE id=?",
                (now, job_id),
            )
            self.conn.commit()
            self.event(f"job {job_id} done")
            return
        outcome = resolve_failure_default(self.conn, job_id, exit_code, now)
        self.event(f"job {job_id} exited {exit_code} — {outcome} (override with `q fixed {job_id}`)")
        self.register_failure(job_id, exit_code, outcome, log_path, now)

    def register_failure(
        self, job_id: int, exit_code: int, outcome: str, log_path: Path, now: float
    ) -> None:
        if ctl_get(self.conn, "halted", "0") == "1":
            return
        pause_until = float(ctl_get(self.conn, "pause_until", "0"))
        if now < pause_until:
            window_failures = int(ctl_get(self.conn, "window_failures", "0")) + 1
            ctl_set(self.conn, "window_failures", str(window_failures))
            if window_failures >= HALT_AFTER:
                ctl_set(self.conn, "halted", "1")
                self.event(f"{window_failures} failures in one window — dispatch HALTED")
                notify(
                    "qsched: dispatch halted",
                    f"{window_failures} job failures within one pause window; dispatch stopped "
                    f"until `q fixed <id> [...]` or `q resume`.\n"
                    f"Latest: job {job_id} (exit {exit_code}), {outcome}.",
                )
            return
        ctl_set(self.conn, "pause_until", str(now + PAUSE_SECONDS))
        ctl_set(self.conn, "window_failures", "1")
        row = self.conn.execute("SELECT argv FROM jobs WHERE id=?", (job_id,)).fetchone()
        argv: list[str] = json.loads(row["argv"])
        self.event(f"dispatch paused for {PAUSE_SECONDS:.0f}s")
        notify(
            f"qsched: job {job_id} failed (exit {exit_code})",
            f"cmd: {shlex.join(argv)}\nlog: {log_path}\n"
            f"Default applied: {outcome}. Dispatch paused {PAUSE_SECONDS / 60:.0f} min.\n"
            f"  `q fixed {job_id}` -> retry at FRONT instead\n"
            f"  `q extend [min]`   -> more time before dispatch resumes\n\n"
            f"--- log tail ---\n{log_tail(log_path)}",
        )

    def reap(self) -> None:
        for job_id, rj in list(self.running.items()):
            exit_code = rj.proc.poll()
            if exit_code is None:
                continue
            rj.log_file.close()
            was_canceling = rj.kill_deadline is not None or self._state(job_id) == "canceling"
            del self.running[job_id]
            self.finalize(job_id, exit_code, was_canceling, rj.log_path)

    def _state(self, job_id: int) -> str:
        row = self.conn.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
        return str(row["state"]) if row else "?"

    def process_cancel_requests(self, now: float) -> None:
        rows = self.conn.execute("SELECT id FROM jobs WHERE state='canceling'").fetchall()
        for row in rows:
            rj = self.running.get(int(row["id"]))
            if rj is None:
                continue
            if rj.kill_deadline is None:
                self.event(f"canceling job {rj.job_id} (SIGTERM)")
                _kill_process_group(rj.proc.pid, signal.SIGTERM)
                rj.kill_deadline = now + CANCEL_GRACE_SECONDS
            elif now > rj.kill_deadline:
                _kill_process_group(rj.proc.pid, signal.SIGKILL)

    # -- shutdown -------------------------------------------------------------
    def shutdown(self) -> None:
        if self.running:
            self.event(f"shutting down — terminating {len(self.running)} running job(s)")
        for rj in self.running.values():
            _kill_process_group(rj.proc.pid, signal.SIGTERM)
        deadline = time.time() + 5.0
        while self.running and time.time() < deadline:
            for job_id, rj in list(self.running.items()):
                if rj.proc.poll() is not None:
                    rj.log_file.close()
                    self._requeue_after_kill(job_id)
                    del self.running[job_id]
            time.sleep(0.1)
        for job_id, rj in list(self.running.items()):
            _kill_process_group(rj.proc.pid, signal.SIGKILL)
            rj.proc.wait()
            rj.log_file.close()
            self._requeue_after_kill(job_id)
        self.running.clear()
        self.event("scheduler stopped; interrupted jobs requeued")

    def _requeue_after_kill(self, job_id: int) -> None:
        new_state = "canceled" if self._state(job_id) == "canceling" else "queued"
        self.conn.execute(
            "UPDATE jobs SET state=?, gpu=NULL, pid=NULL, started_at=NULL WHERE id=?",
            (new_state, job_id),
        )
        self.conn.commit()

    # -- main loop --------------------------------------------------------------
    def loop(self) -> None:
        self.cleanup_stale()
        self.event(f"scheduler up — gpus {self.gpus}, db {DB_PATH}")
        try:
            while not self.stop_requested:
                now = time.time()
                self.reap()
                self.process_cancel_requests(now)
                self.dispatch(now)
                slept = 0.0
                while slept < POLL_SECONDS and not self.stop_requested:
                    time.sleep(0.25)
                    slept += 0.25
        finally:
            self.shutdown()


def _kill_process_group(pid: int, sig: signal.Signals = signal.SIGTERM) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass


def _pid_matches(pid: int, argv: list[str]) -> bool:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        current = [part.decode(errors="replace") for part in cmdline if part]
        return current == argv
    except OSError:
        return False


def cmd_run(gpus: list[int]) -> None:
    QSCHED_HOME.mkdir(parents=True, exist_ok=True)
    lock_file = open(QSCHED_HOME / "scheduler.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another scheduler is already running for this QSCHED_HOME")
    scheduler = Scheduler(conn=db(), gpus=gpus)

    def _request_stop(signum: int, _frame: Any) -> None:
        scheduler.stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _request_stop)
    scheduler.loop()


# ---------------------------------------------------------------- status & friends

def _format_age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def cmd_status(show_all: bool) -> None:
    conn = db()
    now = time.time()
    halted = ctl_get(conn, "halted", "0") == "1"
    pause_until = float(ctl_get(conn, "pause_until", "0"))
    if halted:
        print("!! HALTED (repeated failures) — `q fixed <id> [...]` or `q resume`")
    elif now < pause_until:
        print(
            f"!! dispatch paused {pause_until - now:.0f}s more — "
            f"`q fixed <id>` / `q extend [min]` / `q resume`"
        )
    rows = conn.execute("SELECT * FROM jobs ORDER BY rank, id").fetchall()
    if not show_all:
        finished = [r for r in rows if r["state"] not in ACTIVE_STATES]
        rows = finished[-5:] + [r for r in rows if r["state"] in ACTIVE_STATES]
    if not rows:
        print("queue is empty")
        return
    print(f"{'ID':>5} {'STATE':<9} {'GPU':>3} {'TRY':>3} {'TIME':>6}  COMMAND")
    for row in rows:
        if row["started_at"] and row["state"] in ("running", "canceling"):
            runtime = _format_age(now - row["started_at"])
        elif row["started_at"] and row["finished_at"]:
            runtime = _format_age(row["finished_at"] - row["started_at"])
        else:
            runtime = "-"
        argv: list[str] = json.loads(row["argv"])
        command = shlex.join(argv)
        command = command if len(command) <= 90 else command[:87] + "..."
        gpu = row["gpu"] if row["gpu"] is not None else "-"
        print(f"{row['id']:>5} {row['state']:<9} {gpu!s:>3} {row['retries']:>3} {runtime:>6}  {command}")


def cmd_logs(job_id: int, follow: bool) -> None:
    conn = db()
    row = conn.execute("SELECT log_path FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None or not row["log_path"]:
        raise SystemExit(f"no log for job {job_id}")
    path = Path(row["log_path"])
    print(f"-- {path}", file=sys.stderr)
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read()
            if chunk:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            elif follow:
                time.sleep(0.5)
            else:
                break


def cmd_cancel(job_ids: list[int]) -> None:
    conn = db()
    for job_id in job_ids:
        row = conn.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            print(f"job {job_id}: not found")
        elif row["state"] == "queued":
            conn.execute(
                "UPDATE jobs SET state='canceled', finished_at=? WHERE id=?",
                (time.time(), job_id),
            )
            print(f"job {job_id}: canceled")
        elif row["state"] == "running":
            conn.execute("UPDATE jobs SET state='canceling' WHERE id=?", (job_id,))
            print(f"job {job_id}: cancel requested (scheduler will SIGTERM)")
        else:
            print(f"job {job_id}: state {row['state']}, nothing to cancel")
    conn.commit()


def cmd_fixed(job_ids: list[int]) -> None:
    conn = db()
    for job_id in job_ids:  # first listed gets the lowest rank -> runs first
        row = conn.execute("SELECT state, retries FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            print(f"job {job_id}: not found")
            continue
        if row["state"] not in ("queued", "failed"):
            print(f"job {job_id}: state {row['state']}, cannot retry")
            continue
        conn.execute(
            "UPDATE jobs SET state='queued', rank=?, gpu=NULL, pid=NULL, "
            "started_at=NULL, finished_at=NULL, exit_code=NULL WHERE id=?",
            (front_rank(conn), job_id),
        )
        print(f"job {job_id}: requeued at front (unattended retries used: {row['retries']})")
    conn.commit()
    clear_pause(conn)
    print("dispatch resumed")


def cmd_extend(minutes: float) -> None:
    conn = db()
    now = time.time()
    base = max(now, float(ctl_get(conn, "pause_until", "0")))
    ctl_set(conn, "pause_until", str(base + minutes * 60))
    ctl_set(conn, "halted", "0")  # a halt becomes a timed pause
    print(f"pause extended until {time.strftime('%H:%M:%S', time.localtime(base + minutes * 60))}")


def cmd_resume() -> None:
    conn = db()
    clear_pause(conn)
    print("dispatch resumed")


def cmd_requeue_failed() -> None:
    conn = db()
    rows = conn.execute("SELECT id FROM jobs WHERE state='failed' ORDER BY id").fetchall()
    for row in rows:
        conn.execute(
            "UPDATE jobs SET state='queued', retries=0, rank=?, gpu=NULL, pid=NULL, "
            "started_at=NULL, finished_at=NULL, exit_code=NULL WHERE id=?",
            (back_rank(conn), row["id"]),
        )
    conn.commit()
    print(f"requeued {len(rows)} failed job(s) at back")


# ------------------------------------------------------------------------- main

def _split_at_double_dash(args: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in args:
        return args, []
    idx = args.index("--")
    return args[:idx], args[idx + 1 :]


def main(argv: list[str] | None = None) -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # allow `q status | head`
    args = list(sys.argv[1:] if argv is None else argv)
    before, command = _split_at_double_dash(args)

    parser = argparse.ArgumentParser(prog="q", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="enqueue one job (command after --)")
    p_add.add_argument("--env", action="append", default=[], metavar="K=V")

    p_sweep = sub.add_parser("sweep", help="expand comma sweeps, enqueue jobs (command after --)")
    p_sweep.add_argument("--env", action="append", default=[], metavar="K=V")
    p_sweep.add_argument("-n", "--dry-run", action="store_true")

    p_run = sub.add_parser("run", help="start the scheduler (foreground)")
    p_run.add_argument("--gpus", default="0,1,2", help="comma-separated GPU ids")

    p_status = sub.add_parser("status", help="show the queue")
    p_status.add_argument("--all", action="store_true")

    p_logs = sub.add_parser("logs", help="print (or follow) a job's log")
    p_logs.add_argument("job_id", type=int)
    p_logs.add_argument("-f", "--follow", action="store_true")

    p_cancel = sub.add_parser("cancel", help="cancel queued or running jobs")
    p_cancel.add_argument("job_ids", type=int, nargs="+")

    p_fixed = sub.add_parser("fixed", help="mark failure fixed: retry at front of queue")
    p_fixed.add_argument("job_ids", type=int, nargs="+")

    p_extend = sub.add_parser("extend", help="extend the failure pause")
    p_extend.add_argument("minutes", type=float, nargs="?", default=5.0)

    sub.add_parser("resume", help="end the failure pause / halt now")
    sub.add_parser("requeue-failed", help="re-enqueue all failed jobs at the back")

    ns = parser.parse_args(before)
    match ns.cmd:
        case "add":
            cmd_add(ns.env, command)
        case "sweep":
            cmd_sweep(ns.env, command, ns.dry_run)
        case "run":
            cmd_run([int(g) for g in str(ns.gpus).split(",") if g != ""])
        case "status":
            cmd_status(ns.all)
        case "logs":
            cmd_logs(ns.job_id, ns.follow)
        case "cancel":
            cmd_cancel(ns.job_ids)
        case "fixed":
            cmd_fixed(ns.job_ids)
        case "extend":
            cmd_extend(ns.minutes)
        case "resume":
            cmd_resume()
        case "requeue-failed":
            cmd_requeue_failed()


if __name__ == "__main__":
    main()
