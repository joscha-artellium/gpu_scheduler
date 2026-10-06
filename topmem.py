#!/usr/bin/env python3
import glob
import os
import re
import sys

# --- Color Configuration (Standard Library Only) ---
USE_COLOR = sys.stdout.isatty()

RESET = "\033[0m" if USE_COLOR else ""
BOLD = "\033[1m" if USE_COLOR else ""
DIM = "\033[2m" if USE_COLOR else ""

RED = "\033[31m" if USE_COLOR else ""
GREEN = "\033[32m" if USE_COLOR else ""
YELLOW = "\033[33m" if USE_COLOR else ""
BLUE = "\033[34m" if USE_COLOR else ""
MAGENTA = "\033[35m" if USE_COLOR else ""
CYAN = "\033[36m" if USE_COLOR else ""

BOLD_CYAN = "\033[1;36m" if USE_COLOR else ""
BOLD_YELLOW = "\033[1;33m" if USE_COLOR else ""
BOLD_WHITE = "\033[1;37m" if USE_COLOR else ""

# --- Memory summary layout ---
KIB_PER_GIB = 1024 * 1024
BAR_WIDTH = 40
LABEL_WIDTH = 6
WARN_FRACTION = 0.60
CRITICAL_FRACTION = 0.85
ANSI_ESCAPE = re.compile(r"\033\[[0-9;]*m")

# Processes that act as containers/supervisors for separate user workloads
BOUNDARIES = {
    # System & Session Managers
    "systemd",
    # Desktop Environments & Window Managers
    "gnome-shell",
    "gnome-session",
    # Terminal Emulators & Multiplexers
    "gnome-terminal",
    "tmux",
    "screen",
    # Interactive Shells
    "bash",
}


def is_boundary(pdata):
    """Check if a process is a shell, terminal emulator, etc."""
    cmd = pdata.get("cmdline", "").lower()
    name = pdata.get("name", "").lower()

    if "systemd --user" in cmd:
        return True

    binary = os.path.basename(cmd.split()[0]) if cmd else ""

    for b in BOUNDARIES:
        if binary == b or name == b or binary.startswith(b) or name.startswith(b):
            return True

    return False


def get_meminfo() -> dict[str, int]:
    """Parse /proc/meminfo into {field: value}. Values are kB or count."""
    info: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            info[key] = int(rest.split()[0])
    return info


def visible_len(text: str) -> int:
    return len(ANSI_ESCAPE.sub("", text))


def gib(kib: int) -> float:
    return kib / KIB_PER_GIB


def utilization_color(fraction: float) -> str:
    if fraction >= CRITICAL_FRACTION:
        return RED
    if fraction >= WARN_FRACTION:
        return YELLOW
    return GREEN


def render_bar(used: int, reclaimable: int, total: int, used_color: str) -> str:
    # Cumulative rounding so the three segments always sum to BAR_WIDTH exactly.
    used_cells = round(used / total * BAR_WIDTH)
    through_reclaimable = round((used + reclaimable) / total * BAR_WIDTH)
    reclaimable_cells = through_reclaimable - used_cells
    free_cells = BAR_WIDTH - through_reclaimable
    return (
        f"[{used_color}{'█' * used_cells}{RESET}"
        f"{CYAN}{'▒' * reclaimable_cells}{RESET}"
        f"{DIM}{'░' * free_cells}{RESET}]"
    )


def usage_line(label: str, bar: str, used: int, total: int, color: str) -> str:
    return (
        f"{label:<{LABEL_WIDTH}}{bar}  {gib(used):5.1f} / {gib(total):5.1f} GB "
        f"{color}{used / total:>5.0%}{RESET}"
    )


def print_memory_summary(meminfo: dict[str, int]) -> None:
    total = meminfo["MemTotal"]
    free = meminfo["MemFree"]
    # Pre-3.14 kernels lack MemAvailable; fall back to "used = total - free".
    available = meminfo.get("MemAvailable", free)
    # Clamped: on tiny systems MemAvailable can dip below MemFree (watermark reserve).
    reclaimable = max(0, available - free)
    used = total - free - reclaimable

    apps = meminfo.get("AnonPages", 0)
    shared = meminfo.get("Shmem", 0)
    kernel_other = max(0, used - apps - shared)

    ram_color = utilization_color(used / total)
    indent = " " * LABEL_WIDTH
    ram_bar = render_bar(used, reclaimable, total, ram_color)
    lines = [
        usage_line("RAM", ram_bar, used, total, ram_color),
        (
            f"{indent}used {ram_color}{gib(used):.1f}{RESET} · "
            f"cache {CYAN}{gib(reclaimable):.1f}{RESET} · "
            f"free {GREEN}{gib(free):.1f}{RESET}"
            f"   available {GREEN}{gib(available):.1f}{RESET}"
        ),
        (
            f"{indent}{DIM}of used:{RESET} apps {gib(apps):.1f} · "
            f"shared {gib(shared):.1f} · kernel/other {gib(kernel_other):.1f}"
        ),
    ]

    swap_total = meminfo.get("SwapTotal", 0)
    if swap_total == 0:
        lines.append(f"{'Swap':<{LABEL_WIDTH}}none")
    else:
        swap_used = swap_total - meminfo.get("SwapFree", 0)
        swap_color = utilization_color(swap_used / swap_total)
        swap_bar = render_bar(swap_used, 0, swap_total, swap_color)
        lines.append(usage_line("Swap", swap_bar, swap_used, swap_total, swap_color))

    width = max(visible_len(line) for line in lines)
    print(f"{DIM}{'=' * width}{RESET}")
    print(f"{BOLD_CYAN}MEMORY UTILIZATION{RESET}")
    print(f"{DIM}{'-' * width}{RESET}")
    for line in lines:
        print(line)


def find_ancestor(pid, raw_procs):
    """Walk up PPID chain until hitting root or a boundary process."""
    curr = pid
    visited = set()
    while curr in raw_procs and curr not in visited:
        visited.add(curr)
        ppid = raw_procs[curr]["ppid"]

        if ppid in ("0", "1", "") or ppid not in raw_procs:
            return curr

        if is_boundary(raw_procs[ppid]):
            return curr

        curr = ppid
    return curr


def print_table(
    title, proc_list, top_n, pid_w=7, mem_w=8, cmd_max_len=75, kb_to_gb=1024 * 1024
):
    print(f"{BOLD_CYAN}{title}{RESET}")

    # Headers for width calculation
    fmt_hdr = (
        f"%-{pid_w}s | %-{pid_w}s | %-{mem_w}s | %-{mem_w}s | %-{mem_w}s | COMMAND"
    )
    header_str = fmt_hdr % ("PID", "PPID", "MEM(GB)", "RAM(GB)", "SWAP(GB)")

    raw_rows = []
    printable_rows = []

    for total, rss, swap, pid, ppid, cmd in proc_list[:top_n]:
        cmd_truncated = cmd[:cmd_max_len]

        pid_s = f"{pid:<{pid_w}}"
        ppid_s = f"{ppid:<{pid_w}}"
        mem_s = f"{total / kb_to_gb:<{mem_w}.1f}"
        ram_s = f"{rss / kb_to_gb:<{mem_w}.2f}"
        swap_s = f"{swap / kb_to_gb:<{mem_w}.2f}"

        # Raw plain string to accurately compute column widths
        raw_row = f"{pid_s} | {ppid_s} | {mem_s} | {ram_s} | {swap_s} | {cmd_truncated}"
        raw_rows.append(raw_row)

        # Highlight [X procs] tag if present
        if cmd_truncated.startswith("[") and " procs] " in cmd_truncated:
            proc_tag, rest_cmd = cmd_truncated.split(" procs] ", 1)
            colored_cmd = f"{BOLD_YELLOW}{proc_tag} procs]{RESET} {rest_cmd}"
        else:
            colored_cmd = cmd_truncated

        sep = f"{DIM}|{RESET}"
        printable_row = (
            f"{CYAN}{pid_s}{RESET} {sep} "
            f"{ppid_s} {sep} "
            f"{BOLD_YELLOW}{mem_s}{RESET} {sep} "
            f"{GREEN}{ram_s}{RESET} {sep} "
            f"{MAGENTA}{swap_s}{RESET} {sep} "
            f"{colored_cmd}"
        )
        printable_rows.append(printable_row)

    content_width = (
        max([len(header_str)] + [len(s) for s in raw_rows])
        if raw_rows
        else len(header_str)
    )

    hdr_sep = f"{DIM}|{RESET}"
    colored_header = (
        f"{BOLD_WHITE}{'PID':<{pid_w}}{RESET} {hdr_sep} "
        f"{BOLD_WHITE}{'PPID':<{pid_w}}{RESET} {hdr_sep} "
        f"{BOLD_WHITE}{'MEM(GB)':<{mem_w}}{RESET} {hdr_sep} "
        f"{BOLD_WHITE}{'RAM(GB)':<{mem_w}}{RESET} {hdr_sep} "
        f"{BOLD_WHITE}{'SWAP(GB)':<{mem_w}}{RESET} {hdr_sep} "
        f"{BOLD_WHITE}COMMAND{RESET}"
    )

    print(colored_header)
    print(f"{DIM}{'-' * content_width}{RESET}")
    for p_row in printable_rows:
        print(p_row)


def main():
    # --- Configurable Constants ---
    TOP_N_PROCESSES = 10
    TOP_N_ANCESTORS = 3
    CMDLINE_MAX_LEN = 75
    KB_TO_GB = 1024 * 1024

    raw_procs = {}

    # 1. Gather process data
    for p in glob.glob("/proc/[0-9]*"):
        try:
            pid = os.path.basename(p)
            with open(os.path.join(p, "status"), "r") as f:
                status = f.read()

            name, ppid, rss, swap = "unknown", "1", 0, 0
            for line in status.splitlines():
                if line.startswith("Name:"):
                    name = line.split()[1]
                elif line.startswith("PPid:"):
                    ppid = line.split()[1]
                elif line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("VmSwap:"):
                    swap = int(line.split()[1])

            try:
                with open(os.path.join(p, "cmdline"), "r") as f:
                    cmdline = f.read().replace("\x00", " ").strip()
            except Exception:
                cmdline = ""

            if not cmdline:
                cmdline = name

            raw_procs[pid] = {
                "ppid": ppid,
                "name": name,
                "rss": rss,
                "swap": swap,
                "cmdline": cmdline,
            }

        except Exception:
            continue

    # 2. Individual process list
    procs = []
    for pid, pdata in raw_procs.items():
        total = pdata["rss"] + pdata["swap"]
        if total > 0:
            procs.append(
                (
                    total,
                    pdata["rss"],
                    pdata["swap"],
                    pid,
                    pdata["ppid"],
                    pdata["cmdline"],
                )
            )

    procs.sort(key=lambda x: x[0], reverse=True)

    # 3. Aggregate process list by application root ancestor
    anc_map = {}
    for pid, pdata in raw_procs.items():
        anc_pid = find_ancestor(pid, raw_procs)
        if anc_pid not in anc_map:
            anc_info = raw_procs.get(anc_pid, pdata)
            anc_map[anc_pid] = {
                "ppid": anc_info["ppid"],
                "cmdline": anc_info["cmdline"],
                "rss": 0,
                "swap": 0,
                "count": 0,
            }
        anc_map[anc_pid]["rss"] += pdata["rss"]
        anc_map[anc_pid]["swap"] += pdata["swap"]
        anc_map[anc_pid]["count"] += 1

    ancestor_procs = []
    for anc_pid, adata in anc_map.items():
        total = adata["rss"] + adata["swap"]
        if total > 0:
            cmd_display = (
                f"[{adata['count']} procs] {adata['cmdline']}"
                if adata["count"] > 1
                else adata["cmdline"]
            )
            ancestor_procs.append(
                (
                    total,
                    adata["rss"],
                    adata["swap"],
                    anc_pid,
                    adata["ppid"],
                    cmd_display,
                )
            )

    ancestor_procs.sort(key=lambda x: x[0], reverse=True)

    # 4. Print Tables
    print_table(
        "TOP INDIVIDUAL PROCESSES",
        procs,
        TOP_N_PROCESSES,
        cmd_max_len=CMDLINE_MAX_LEN,
        kb_to_gb=KB_TO_GB,
    )
    print()
    print_table(
        "TOP PROCESS TREES (AGGREGATED)",
        ancestor_procs,
        TOP_N_ANCESTORS,
        cmd_max_len=CMDLINE_MAX_LEN,
        kb_to_gb=KB_TO_GB,
    )

    # 5. Memory summary
    print()
    print_memory_summary(get_meminfo())
    print()


if __name__ == "__main__":
    main()
