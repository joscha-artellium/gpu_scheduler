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
q run                                # scheduler for GPUs 0,2,1 (--gpus to change)

# anywhere else, any time — your submission scripts become lists of these:
q sweep --env REGION=US3 -- uv run python scripts/train_predict.py \
    model=xgb01,xgb02 'training_window="96"'
q status                             # counts, queue, and a live ETA per running job
q logs 17 -f                         # follow a job's progress bars
```

Your submission scripts are now declarative manifests: since the framework is
idempotent, re-running a whole (edited) script is safe — already-computed
configs enqueue, run, detect their artifacts, and exit in seconds.

`--gpus` is a **preference order, not a set**: the scheduler fills GPUs in the
order you list them. The default `0,2,1` deprioritises GPU 1, which drives the
system display and is therefore slower and louder under load — it only picks up
work once 0 and 2 are busy.

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
- **The key must match `[+~]{0,2}[\w.@/:]+`** — letters, digits, `_`, `.`,
  `@`, `/`, `:`, optionally prefixed by `+` or `~`. Anything else is not a
  sweep token and is passed through whole, so `'my-key=a,b'` (hyphen) and
  `'paths[0]=a,b'` (index) each stay **one** variant. A dry run will show you.
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

## Validation probes (`--validate`)

`q sweep -n` checks that *q* understood your command line. `--validate` also
asks the **target** whether it understood it, before anything is enqueued — so
a typo costs you seconds instead of a queue full of jobs that each die on
startup.

```bash
q sweep --validate -- uv run python scripts/train_predict.py model=xgb01,xgb02
```

### What your target has to do

`q` runs each job once with `QSCHED_VALIDATE=1` in the environment. **Your
argv is not changed** — no extra flag to declare in your config schema. Read
the variable, check the config, print `qsched: validated`, exit 0:

```python
if os.environ.get("QSCHED_VALIDATE"):
    cfg = compose_and_check()          # raise / sys.exit(1) if anything is wrong
    print("qsched: validated")
    raise SystemExit(0)
```

**Both parts are required.** Exit 0 without the sentinel counts as a rejection
("never printed 'qsched: validated'") — otherwise a target that has never heard
of `QSCHED_VALIDATE` would run its real workload as its own probe and be
declared valid.

Probes get `CUDA_VISIBLE_DEVICES=""` (a probe can never take VRAM from a
running job), your recorded cwd, and your `--env` values. They run in parallel,
each capped at `QSCHED_VALIDATE_TIMEOUT` seconds (default 60).

### What happens when something is rejected

Rejections print to stderr with the reason and up to 40 lines of the probe's
output. What happens next is `--on-reject`:

| flag                  | behaviour                                          |
|-----------------------|----------------------------------------------------|
| *(default)* `ask`     | prompt; **yes after 100 s** if you don't answer     |
| `--on-reject skip`    | enqueue the jobs that passed, no prompt             |
| `--on-reject abort`   | enqueue nothing, no prompt                          |

Use `skip` in submission scripts, where nobody is watching a prompt. Use
`abort` when a sweep only makes sense whole.

Exit code is 0 if anything was enqueued, 1 otherwise (all rejected, `abort`, or
you answered no) — so a `set -e` script stops on `abort` and carries on after
`skip`.

`q sweep -n --validate` probes and reports but never enqueues and never
prompts, exiting 1 if anything was rejected. That is your standalone "is this
sweep valid?" check. stdout stays paste-able either way; verdicts go to stderr.

## Status

`q status` prints a count line
(`queued 41 · running 3 · done 38 · failed 2  (84 total)`), then the last 5
finished jobs plus everything active (`--all` for the lot). "Last 5" is by
**queue rank**, not by finish time, so jobs you reordered can show up out of
chronological order. For running jobs the `ETA` column parses the **last tqdm
bar** in the tail of the log (`83% 58m`). Caveats: with nested bars that's the
innermost loop, not the job; bars without a total (`38it [03:13, ...]`) and
bars that haven't produced an estimate yet show `-`.

## Scheduling & failure semantics

Dispatch is FIFO by queue rank; each job gets `CUDA_VISIBLE_DEVICES=<gpu>`.

A job exiting nonzero gets the **unattended default immediately** (its GPU
frees up): first unattended failure → requeued at the **back** of the queue
with `retries=1`; second → marked `failed` permanently. You get a
notification stating which happened (throttled — see Notifications), and
dispatch of *new* jobs pauses for 3 min (`QSCHED_PAUSE_SECONDS`) — running jobs
continue. During the pause you can override the default:

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
`q fixed`, `q resume` and `q extend` all reset the streak as well; `q fixed`
and `q resume` clear a halt outright, `q extend` converts it back into a timed
pause. Cancels never count as failures.

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
- `q resume` — end the pause or halt **now**. It writes only the pause/halt/
  streak state and touches no job: nothing is reordered and nothing already
  marked `failed` comes back. Use it when you've looked at the failure and
  decided the queue should carry on — a flaky OOM you don't care about, or a
  cause you fixed out of band (jobs re-read your code at spawn time, so a
  revert fixes everything still queued). `q fixed <id>` is the same thing
  *plus* moving those ids to the front and reviving them from `failed`.
- `q requeue-failed` — every `failed` job goes back to the queue, appended at
  the **back** in ascending id order, with **`retries` reset to 0**: a genuine
  second chance, each job getting its "one free unattended retry" budget again.
  `canceled` jobs are untouched. This is the end-of-sweep tool — the drain
  notification hands you the failed ids, you fix the root cause, and one
  command reruns exactly the stragglers. Note it does **not** clear a pause or
  a halt, so if the breaker tripped you need `q resume` as well before anything
  dispatches. Run it before `q clear`, which deletes failed jobs outright.
- `q front <id> ...` / `q back <id> ...` — reorder **queued** jobs; the listed
  order is preserved. Same for the ids you pass `q fixed`.
- `q show <id>` — prints `cwd:` / `env:` / `cmd:` for a job; `q show <id> -r`
  prints a paste-able `q add` line (prefixed with `cd <cwd> &&` when you are
  somewhere else), which is how you resurrect a cancelled job.
- `q clear` — delete finished (`done`/`failed`/`canceled`) jobs and their log
  files; `--older-than DAYS` scopes it, `-n` shows what would go. `q clear
  --all` wipes the queue, resets the pause/halt/streak state, restarts ids at
  1, and refuses while a scheduler holds the lock. Log files always follow the
  db: any `logs/<id>.log` without a matching row is removed too.
- Ctrl-C / SIGTERM / SIGHUP on the scheduler: running jobs are SIGTERMed (then
  SIGKILLed after 5 s) and requeued in place. On the next `q run`, stale
  `running` rows are requeued (verified orphans from a SIGKILLed scheduler are
  killed first via /proc cmdline match). With an idempotent framework, restart
  is always safe.
- A lock file prevents two schedulers per `QSCHED_HOME` (double-booking).

Known limitations (by design — migrate to Slurm if these start to hurt):
single machine, single GPU per job, no priorities, the queue must be the only
entry point (a manually launched job is invisible and will be double-booked
against), and nothing contains a job that ignores `CUDA_VISIBLE_DEVICES`.

## Notifications

You get a notification on failure, on halt, and once per batch when the queue
drains (the last running job finishes and nothing is queued) with the tally —
`2 done, 1 failed` plus the failed ids.

**Failure alerts are throttled to at most one per pause window**, and none are
sent at all while halted. With the default 3 min pause you therefore see at
most one failure alert every 3 minutes, and a run that dies job after job stops
alerting entirely once the breaker trips — the halt notification is the last
word. Fast failures are heavily coalesced this way; failures spaced further
apart than the pause each get their own alert. Nothing is lost either way: the
drain notification lists every failed id, and `q status` has the full tally.

Only the halt is sent at `notify-send` urgency `critical` (which on most
desktops means it never auto-expires); per-failure and drain alerts are
`normal` and self-dismiss.

The scheduler runs, in order of preference:

1. `~/.config/qsched/notify.sh <title> <body> <kind>` if present+executable
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

The hook's third argument is the `kind`: `failure`, `halt`, `drain` or
`digest`. Older two-argument hooks keep working — shell scripts ignore the
extra argument. The body includes the command, log path, and the log tail.

## Periodic status digest

The scheduler emails the standard `q status` view — the same table, so it
carries live ETAs — at fixed local hours. Default `8,14,20`: morning, after
lunch, and after dinner.

```bash
QSCHED_STATUS_AT=8,14,20   # default
QSCHED_STATUS_AT=20        # just the evening one
QSCHED_STATUS_AT=          # off
```

- **An idle queue is silent.** A digest is skipped if nothing is active *and*
  nothing finished since the previous one, so a drained queue doesn't mail you
  three times a day about nothing.
- Only `q run` sends digests: no scheduler, no mail.
- They never reach `notify-send` — a status table as a desktop popup is the
  opposite of useful. They go to the hook (with `kind=digest`) or to email.
- Missed hours don't stack: if the scheduler was down at 14:00 it does not
  send a late one, it just waits for 20:00.
- The subject is the count line (`qsched: queued 41 · running 3 · done 38
  (82 total)`), often the whole answer on a phone.
- Overhead is one float comparison per poll; the next due time is computed
  once per digest.

Emailing these needs a hook, since the built-in `mail` fallback is plain
`mail(1)`. This one mails everything and pops up only the interactive kinds:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["python-dotenv"]
# ///
import subprocess, sys
sys.path.insert(0, "/abs/path/to/your/mailer")
from notification import send_email

title, body = sys.argv[1], sys.argv[2]
kind = sys.argv[3] if len(sys.argv) > 3 else "failure"

if kind != "digest":
    urgency = "critical" if kind == "halt" else "normal"
    subprocess.run(["notify-send", "-u", urgency, title, body], check=False)
send_email(title, body, raise_unconfigured=False)
```

A hook replaces both built-in channels, so this one owns the desktop popup too.
Keep the mailer's SMTP timeout short: the scheduler calls the hook
synchronously between polls, under a 30 s cap.

## Tunables (env vars)

| var                       | default                      |                                 |
|---------------------------|------------------------------|---------------------------------|
| `QSCHED_HOME`             | `~/.local/share/qsched`      | db + logs                       |
| `QSCHED_PAUSE_SECONDS`    | 180                          | pause after a failure; also caps the failure-alert rate |
| `QSCHED_HALT_AFTER`       | 12                           | consecutive failures before halt (0 = never) |
| `QSCHED_POLL_SECONDS`     | 2                            | scheduler poll interval         |
| `QSCHED_STATUS_AT`        | `8,14,20`                    | local hours for the status digest (empty = off) |
| `QSCHED_VALIDATE_TIMEOUT` | 60                           | per-probe timeout for `--validate` |
| `QSCHED_NOTIFY`           | `~/.config/qsched/notify.sh` | hook path                       |
| `QSCHED_EMAIL`            | unset                        | enables mail fallback           |

`QSCHED_VALIDATE=1` is set by `q` *into a probe's* environment; it is not a
tunable you set yourself.

## Tests

```bash
./test_q.py        # self-contained: uv provisions python>=3.12 and pytest
```