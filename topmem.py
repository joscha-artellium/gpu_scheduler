#!/usr/bin/env python3
import glob
import os
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

# Processes that act as containers/supervisors for separate user workloads
BOUNDARIES = {
    # System & Session Managers
    "systemd",
    "init",
    "sshd",
    "containerd",
    "dockerd",
    "lightdm",
    "gdm",
    "gdm3",
    "sddm",
    # Desktop Environments & Window Managers
    "gnome-shell",
    "gnome-session",
    "mutter",
    "kwin_x11",
    "kwin_wayland",
    "plasmashell",
    "plasmawindowed",
    "sway",
    "i3",
    "xfwm4",
    "xorg",
    "Xorg",
    "wayland",
    # Terminal Emulators & Multiplexers
    "gnome-terminal-server",
    "gnome-terminal",
    "konsole",
    "alacritty",
    "kitty",
    "wezterm",
    "xterm",
    "urxvt",
    "tilix",
    "tmux",
    "tmux: server",
    "screen",
    # Interactive Shells
    "bash",
    "zsh",
    "fish",
    "sh",
    "csh",
    "tcsh",
    "dash",
}


def is_boundary(pdata):
    """Check if a process is a shell, terminal emulator, desktop host, or system supervisor."""
    cmd = pdata.get("cmdline", "").lower()
    name = pdata.get("name", "").lower()

    if "systemd --user" in cmd:
        return True

    binary = os.path.basename(cmd.split()[0]) if cmd else ""

    for b in BOUNDARIES:
        if binary == b or name == b or binary.startswith(b) or name.startswith(b):
            return True

    return False


def get_meminfo():
    info = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    info[parts[0].strip()] = int(parts[1].strip().split()[0])
    except Exception:
        pass
    return info


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
    print(f"\n{BOLD_CYAN}{title}{RESET}")

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
    tot_rss_all = 0
    tot_swap_all = 0

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

            tot_rss_all += rss
            tot_swap_all += swap
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
    print_table(
        "TOP PROCESS TREES (AGGREGATED)",
        ancestor_procs,
        TOP_N_ANCESTORS,
        cmd_max_len=CMDLINE_MAX_LEN,
        kb_to_gb=KB_TO_GB,
    )

    # 5. System Reconciliation Summary
    m = get_meminfo()
    mem_total = m.get("MemTotal", 0) / KB_TO_GB
    mem_free = m.get("MemFree", 0) / KB_TO_GB
    mem_avail = m.get("MemAvailable", 0) / KB_TO_GB
    ram_used = mem_total - mem_avail if "MemAvailable" in m else (mem_total - mem_free)

    swap_total = m.get("SwapTotal", 0) / KB_TO_GB
    swap_free = m.get("SwapFree", 0) / KB_TO_GB
    swap_used = swap_total - swap_free

    proc_ram_gb = tot_rss_all / KB_TO_GB
    proc_swap_gb = tot_swap_all / KB_TO_GB

    anon_pages = m.get("AnonPages", 0) / KB_TO_GB
    cached_gb = (m.get("Cached", 0) + m.get("Buffers", 0)) / KB_TO_GB
    kernel_driver_gb = max(0.0, mem_total - anon_pages - cached_gb - mem_free)
    other_swap_gb = max(0.0, swap_used - proc_swap_gb)

    # Plain strings for length measurement
    ram1 = f"RAM Used: {ram_used:.2f}/{mem_total:.2f} GB (Avail: {mem_avail:.2f} GB, Free: {mem_free:.2f} GB)"
    ram2 = f"RAM (RSS): {proc_ram_gb:.2f} GB (incl. shared)"
    ram_w = len(ram1)

    p_line1 = f"  System Totals : {ram1:<{ram_w}} | Swap Used: {swap_used:.2f}/{swap_total:.2f} GB (Free: {swap_free:.2f} GB)"
    p_line2 = (
        f"  Process Sums  : {ram2:<{ram_w}} | Swap (VmSwap): {proc_swap_gb:.2f} GB"
    )

    lbl_ram = f"RAM  ({mem_total:.2f} GB)"
    lbl_swap = f"SWAP ({swap_total:.2f} GB)"
    lbl_w = max(len(lbl_ram), len(lbl_swap))

    p_line_ram = f"  {lbl_ram:<{lbl_w}} : Heap (Anon): {anon_pages:.2f} GB | Cache: {cached_gb:.2f} GB | Kernel/Drivers: {kernel_driver_gb:.2f} GB | Free: {mem_free:.2f} GB"
    p_line_swap = f"  {lbl_swap:<{lbl_w}} : Process Swap: {proc_swap_gb:.2f} GB | Other/IPC Swap: {other_swap_gb:.2f} GB | Free: {swap_free:.2f} GB"

    summary_width = max(len(s) for s in [p_line1, p_line2, p_line_ram, p_line_swap])

    # Colored strings for output
    sep = f"{DIM}|{RESET}"
    c_ram1 = f"RAM Used: {YELLOW}{ram_used:.2f}{RESET}/{mem_total:.2f} GB (Avail: {GREEN}{mem_avail:.2f}{RESET} GB, Free: {GREEN}{mem_free:.2f}{RESET} GB)"
    c_ram2 = f"RAM (RSS): {GREEN}{proc_ram_gb:.2f}{RESET} GB (incl. shared)"
    spaces_ram2 = " " * (ram_w - len(ram2))

    c_line1 = f"  {BOLD}System Totals{RESET} : {c_ram1} {sep} Swap Used: {MAGENTA}{swap_used:.2f}{RESET}/{swap_total:.2f} GB (Free: {GREEN}{swap_free:.2f}{RESET} GB)"
    c_line2 = f"  {BOLD}Process Sums {RESET} : {c_ram2}{spaces_ram2} {sep} Swap (VmSwap): {MAGENTA}{proc_swap_gb:.2f}{RESET} GB"

    lbl_ram_pad = f"{lbl_ram:<{lbl_w}}"
    lbl_swap_pad = f"{lbl_swap:<{lbl_w}}"

    c_line_ram = (
        f"  {BOLD_CYAN}{lbl_ram_pad}{RESET} : "
        f"Heap (Anon): {YELLOW}{anon_pages:.2f}{RESET} GB {sep} "
        f"Cache: {CYAN}{cached_gb:.2f}{RESET} GB {sep} "
        f"Kernel/Drivers: {BLUE}{kernel_driver_gb:.2f}{RESET} GB {sep} "
        f"Free: {GREEN}{mem_free:.2f}{RESET} GB"
    )

    c_line_swap = (
        f"  {BOLD_CYAN}{lbl_swap_pad}{RESET} : "
        f"Process Swap: {MAGENTA}{proc_swap_gb:.2f}{RESET} GB {sep} "
        f"Other/IPC Swap: {YELLOW}{other_swap_gb:.2f}{RESET} GB {sep} "
        f"Free: {GREEN}{swap_free:.2f}{RESET} GB"
    )

    print("\n" + f"{DIM}{'=' * summary_width}{RESET}")
    print(f"{BOLD_CYAN}SYSTEM RECONCILIATION SUMMARY{RESET}")
    print(f"{DIM}{'-' * summary_width}{RESET}")
    print(c_line1)
    print(c_line2)
    print(f"{DIM}{'-' * summary_width}{RESET}")
    print(c_line_ram)
    print(c_line_swap)


if __name__ == "__main__":
    main()
