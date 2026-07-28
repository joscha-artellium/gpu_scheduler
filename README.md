# qsched

Minimal GPU job queue for a single multi-GPU machine. One SQLite database
holds the queue; `q run` is a foreground scheduler you own in a terminal that
dispatches queued jobs onto free GPUs — exact bookkeeping (the scheduler owns
GPU assignment), no nvidia-smi heuristics, one GPU per job. The CLI works
whether or not the scheduler is running: jobs enqueued while it is down are
picked up when it starts.

Stdlib-only, Python ≥ 3.12, Linux.

## Install

```bash
chmod +x q.py
ln -s "$PWD/q.py" ~/.local/bin/q     # or any dir on PATH
```

State lives in `~/.local/share/qsched/` (`queue.db`, `logs/<id>.log`);
override with `QSCHED_HOME`.

## Daily workflow

```bash
# terminal 1 (ideally inside tmux):
q run                                # scheduler for GPUs 0,1,2 (--gpus to change)

# anywhere else, any time — your submission scripts become lists of these:
q sweep --env REGION=US3 -- uv run python scripts/train_predict.py \
    model=xgb01,xgb02 'training_window="96"'
q status                             # counts, queue, and a live ETA per running job
q logs 17 -f                         # follow a job's progress bars
```

Your submission scripts are now declarative manifests: since the framework is
idempotent, re-running a whole (edited) script is safe — already-computed
configs enqueue, run, detect their artifacts, and exit in seconds.

## Quoting rules — read this once

**There is no second shell.** `q` stores the argv your interactive shell
hands it and later passes it byte-for-byte to `execvp()`. Your shell strips
exactly one layer of quoting — the same layer it would strip if you invoked
the training script directly. Therefore:

> Quote exactly as you would for a direct invocation. No extra escaping.

- `q add -- <cmd ...>` — nothing after `--` is interpreted. Ever.
- `q sweep -- <cmd ...>` — each token of the form `KEY=VALUE` is split on
  **top-level commas** (commas outside `()`, `[]`, `{}`, `'…'`, `"…"`) into
  sweep variants; the cartesian product over all tokens is enqueued.

| You type (bash)                                       | hydra receives         | sweep variants                   |
|-------------------------------------------------------|------------------------|----------------------------------|
| `'training_window="96"'`                              | `training_window="96"` | 1                                |
| `lr=0.1,0.01`                                         | —                      | 2                                |
| `'features_transform.grouping=["Day","key"],["Day"]'` | —                      | 2: `["Day","key"]` and `["Day"]` |
| `'key=[a,b]'`                                         | `key=[a,b]`            | 1 (commas bracketed)             |
| `'key="a,b"'`                                         | `key="a,b"`            | 1 (comma quoted)                 |
| `model=a,b lr=0.1,0.01`                               | —                      | 4 (product)                      |

Not interpreted / not supported:

- Tokens starting with `-` and tokens without `=` are never split.
- Backslash escapes are not understood by the splitter — use quotes.
- Hydra's `range()` / `glob()` / sweeper plugins are not expanded. Do that
  expansion in your submission script (a loop emitting `q add` lines).
- Environment variables go through `--env K=V` (repeatable), never shell
  prefixes — `REGION=US3 q add ...` sets it for `q`, not for the job.
- Need pipes, `&&`, or globs at run time: `q add -- bash -c '<script>'`.

**Always check nontrivial sweeps with a dry run** — it prints one
shell-quoted line per job that you can eyeball or even paste back:

```bash
q sweep -n -- uv run python scripts/train_predict.py \
    'features_transform.grouping=["Day","key"],["Day"]' model=xgb01,xgb02
```

The submission-time working directory is recorded per job and the job runs
in it, so relative paths (`scripts/…`, hydra config dirs) behave as if you
had launched from where you typed `q add`.

## Status

`q status` prints a count line (`queued 41 · running 3 · done 38 · failed 2`),
then the last 5 finished jobs plus everything active (`--all` for the lot).
For running jobs the `ETA` column parses the **last tqdm bar** in the tail of
the log (`83% 58m`). Caveats: with nested bars that's the innermost loop, not
the job; bars without a total (`38it [03:13, ...]`) and bars that haven't
produced an estimate yet show `-`.

## Scheduling & failure semantics

Dispatch is FIFO by queue rank; each job gets `CUDA_VISIBLE_DEVICES=<gpu>`.

A job exiting nonzero gets the **unattended default immediately** (its GPU
frees up): first unattended failure → requeued at the **back** of the queue
with `retries=1`; second → marked `failed` permanently. You get a
notification stating which happened, and dispatch of *new* jobs pauses for
5 min (`QSCHED_PAUSE_SECONDS`) — running jobs continue. During the pause you
can override the default:

1. **`q fixed <id> [...]`** — "I attempted a fix": the job moves to the
   **front** of the queue, allowed regardless of retry/failure count. The
   retry counter is **not** reset — a fixed job that fails *unattended* again
   follows the default for its count. Clears the pause/halt.
2. **`q extend [minutes]`** — buy more time before dispatch resumes (default
   5 min). Also converts a halt back into a timed pause.
3. **Do nothing** — the pause expires and dispatch resumes; the default is
   already in place. (`q fixed` still works later.)

Guard rails: pauses don't stack, and behind them sits a circuit breaker —
`QSCHED_HALT_AFTER` (default 12, `0` disables) **consecutive** failures with no
job exiting 0 in between turn the pause indefinite (halt). Any successful job
resets the streak, so a long sweep with the odd OOM never halts, while a bad
commit that kills every job stops the queue after 12 instead of burning 200.
`q fixed`/`q resume` clear a halt; `q extend` converts it back to a timed pause
and resets the streak. Cancels never count as failures.

Other mechanics:

- `q restart <id> ...` — SIGTERM the running job(s) and requeue **in place**
  (same rank, retry counter untouched; not a failure). Use after a code/config
  fix when you don't want to bounce the scheduler or disturb the rest of the
  queue; SIGKILL after 10 s for jobs that ignore SIGTERM. `q restart --all`
  does this for every running job — the ids are required otherwise, since this
  throws away in-flight progress on every GPU. It only ever touches `running`
  jobs: nothing you cancelled or that already failed is resurrected by it.
- `q cancel <id> ...` — queued jobs go straight to `canceled`; running jobs get
  SIGTERM (then SIGKILL after 10 s). Never counted as a failure, never triggers
  a pause. A cancel is final: use `q fixed` (failed jobs) or resubmit by hand.
- `q front <id> ...` / `q back <id> ...` — reorder **queued** jobs; the listed
  order is preserved. Same for the ids you pass `q fixed`.
- `q show <id>` — prints `cwd:` / `env:` / `cmd:` for a job; `q show <id> -r`
  prints a paste-able `q add` line (prefixed with `cd <cwd> &&` when you are
  somewhere else), which is how you resurrect a cancelled job.
- `q clear` — delete finished (`done`/`failed`/`canceled`) jobs and their log
  files; `--older-than DAYS` scopes it, `-n` shows what would go. `q clear
  --all` wipes the queue and restarts ids at 1, and refuses while a scheduler
  holds the lock. Log files always follow the db: any `logs/<id>.log` without a
  matching row is removed too.
- Ctrl-C / SIGTERM / SIGHUP on the scheduler: running jobs are terminated and
  requeued. On the next `q run`, stale `running` rows are requeued (verified
  orphans from a SIGKILLed scheduler are killed first via /proc cmdline
  match). With an idempotent framework, restart is always safe.
- A lock file prevents two schedulers per `QSCHED_HOME` (double-booking).
- `q requeue-failed` resets all `failed` jobs to the back of the queue.

Known limitations (by design — migrate to Slurm if these start to hurt):
single machine, single GPU per job, no priorities, the queue must be the only
entry point (a manually launched job is invisible and will be double-booked
against), and nothing contains a job that ignores `CUDA_VISIBLE_DEVICES`.

## Notifications

You get a notification on failure, on halt, and once per batch when the queue
drains (the last running job finishes and nothing is queued) with the tally —
`2 done, 1 failed` plus the failed ids. The scheduler runs, in order of
preference:

1. `~/.config/qsched/notify.sh <title> <body>` if present+executable
   (override path with `QSCHED_NOTIFY`), else
2. `notify-send` (desktop), **and** `mail -s <title> $QSCHED_EMAIL` if
   `QSCHED_EMAIL` is set and `mail` exists.

For desktop + email, either export `QSCHED_EMAIL=you@example.com` (with a
working `mail`/msmtp setup, e.g. `~/.msmtprc` + `mailx`), or drop in a hook:

```bash
# ~/.config/qsched/notify.sh   (chmod +x)
#!/usr/bin/env bash
notify-send -u critical "$1" "$2"
printf '%s\n' "$2" | mail -s "$1" you@example.com
```

The body includes the command, log path, and the log tail.

## Tunables (env vars)

| var                    | default                      |                                 |
|------------------------|------------------------------|---------------------------------|
| `QSCHED_HOME`          | `~/.local/share/qsched`      | db + logs                       |
| `QSCHED_PAUSE_SECONDS` | 300                          | pause after a failure           |
| `QSCHED_HALT_AFTER`    | 12                           | consecutive failures before halt (0 = never) |
| `QSCHED_POLL_SECONDS`  | 2                            | scheduler poll interval         |
| `QSCHED_NOTIFY`        | `~/.config/qsched/notify.sh` | hook path                       |
| `QSCHED_EMAIL`         | unset                        | enables mail fallback           |

## Tests

```bash
./test_q.py        # self-contained: uv provisions python>=3.12 and pytest
```