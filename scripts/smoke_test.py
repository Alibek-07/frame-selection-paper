#!/usr/bin/env python3
"""
Smoke test for the frame-selection harness.

Not a science test - a plumbing test. Catches the bugs that produce
plausible-looking but wrong results tables.

ADAPT THE CONFIG BLOCK, then:  python scripts/smoke_test.py
"""

import sys
import numpy as np

# ============ ADAPT THIS BLOCK TO YOUR CODE ============
# from selection import STRATEGIES          # {'uniform': fn, 'random': fn, ...}
# from selection import load_capture, run_pipeline, room_error

STRATEGIES = {}          # name -> fn(capture, K, seed) -> list[int]
CAPTURE_PATH = "PATH/TO/ONE/TENSOR/CAPTURE"
K_NORMAL = 30
# =======================================================

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main():
    if not STRATEGIES:
        sys.exit("Fill in the CONFIG BLOCK first (STRATEGIES, CAPTURE_PATH).")

    cap = load_capture(CAPTURE_PATH)                      # noqa: F821
    print(f"capture loaded, {len(cap)} frames\n")

    # ---- 1. strategies must produce DIFFERENT selections ----------------
    print("1. strategies are distinct")
    sel = {n: sorted(fn(cap, K_NORMAL, seed=0)) for n, fn in STRATEGIES.items()}
    for n, idx in sel.items():
        check(f"{n}: returns K distinct indices",
              len(idx) == K_NORMAL and len(set(idx)) == K_NORMAL,
              f"got {len(set(idx))}/{K_NORMAL}")
        check(f"{n}: indices in range",
              min(idx) >= 0 and max(idx) < len(cap))

    names = list(sel)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = set(sel[names[i]]), set(sel[names[j]])
            ov = len(a & b) / K_NORMAL
            check(f"{names[i]} vs {names[j]} differ",
                  ov < 0.95, f"overlap {ov:.0%}")

    # ---- 2. determinism --------------------------------------------------
    print("\n2. determinism")
    for n, fn in STRATEGIES.items():
        r1 = sorted(fn(cap, K_NORMAL, seed=0))
        r2 = sorted(fn(cap, K_NORMAL, seed=0))
        check(f"{n}: same seed -> same result", r1 == r2)
    if "random" in STRATEGIES:
        d1 = sorted(STRATEGIES["random"](cap, K_NORMAL, seed=0))
        d2 = sorted(STRATEGIES["random"](cap, K_NORMAL, seed=1))
        check("random: different seed -> different result", d1 != d2)

    # ---- 3. the failure branch must actually fire ------------------------
    print("\n3. degenerate budgets record failure, not crash")
    for K in (1, 2, 3):
        for n, fn in STRATEGIES.items():
            try:
                idx = fn(cap, K, seed=0)
                res = run_pipeline(cap, idx)               # noqa: F821
                ok = (res is None) or (not res.get("success", True)) \
                     or ("dimensions" in res)
                check(f"K={K} {n}: handled", ok,
                      f"success={None if res is None else res.get('success')}")
            except Exception as e:
                check(f"K={K} {n}: handled", False, f"raised {type(e).__name__}")

    # ---- 4. error magnitude is sane --------------------------------------
    print("\n4. metric magnitude")
    idx = STRATEGIES[names[0]](cap, K_NORMAL, seed=0)
    res = run_pipeline(cap, idx)                           # noqa: F821
    if res and res.get("success", True):
        err = room_error(res, cap)                         # noqa: F821
        check("relative error in 0.5%-40%", 0.005 < err < 0.40, f"got {err:.1%}")
        print("   <0.5% is suspiciously good (leaking GT?);"
              " >40% means something is broken")
    else:
        check("baseline capture succeeds at K=30", False)

    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()