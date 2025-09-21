#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_FILE = os.path.join(ROOT, "pools_result.json")
OUT_FILE = os.path.join(ROOT, "pools_summary.json")


def main() -> int:
    with open(IN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    by_chain = Counter()
    by_exchange = Counter()
    by_pair = Counter()  # (chain, exchange)
    assets = 0

    for asset, chains in data.items():
        assets += 1
        if not isinstance(chains, dict):
            continue
        for chain, entries in chains.items():
            if not isinstance(entries, list):
                continue
            n = len(entries)
            total += n
            by_chain[chain] += n
            for e in entries:
                exch = (e.get("биржа") or e.get("exchange") or "").strip() or "unknown"
                by_exchange[exch] += 1
                by_pair[(chain, exch)] += 1

    summary = {
        "assets": assets,
        "total_entries": total,
        "by_chain": dict(by_chain.most_common()),
        "by_exchange": dict(by_exchange.most_common()),
        "by_chain_exchange": {
            f"{k[0]}::{k[1]}": v
            for k, v in sorted(by_pair.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        },
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Краткий вывод
    print(f"assets: {assets}")
    print(f"total_entries: {total}")
    print("by_chain (top 10):")
    for k, v in by_chain.most_common(10):
        print(f"  {k}: {v}")
    print("by_exchange (top 10):")
    for k, v in by_exchange.most_common(10):
        print(f"  {k}: {v}")
    print("by_chain_exchange (top 10):")
    for (ch, ex), v in by_pair.most_common(10):
        print(f"  {ch}::{ex}: {v}")

    print(f"WROTE -> {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
