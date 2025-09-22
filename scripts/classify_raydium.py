#!/usr/bin/env python3
"""Classify Raydium pools by program owner via Solana RPC."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    requests = None

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POOLS_FILE = os.path.join(ROOT_DIR, "pools_result.json")
OUT_FILE = os.path.join(ROOT_DIR, "classified_pools_raydium.json")
LOG_FILE = os.path.join(ROOT_DIR, "term.txt")

PROGRAM_OWNERS: Dict[str, Tuple[str, str]] = {
    "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK": (
        "clmm",
        "Raydium CLMM (concentrated liquidity)",
    ),
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": (
        "amm_v4",
        "Raydium AMM V4 / Hybrid pool",
    ),
    "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C": (
        "cpmm",
        "Raydium CPMM pool",
    ),
}


_log_fh = None  # type: ignore

def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {msg}"
    try:
        print(line, flush=True)
    except BrokenPipeError:
        pass
    if _log_fh is not None:
        try:
            _log_fh.write(line + "\n")
            _log_fh.flush()
        except Exception:
            pass


def ensure_requests() -> None:
    if requests is None:
        raise RuntimeError("Требуется пакет 'requests'. Установите зависимости из requirements.txt")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify Raydium pools by Solana program owner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pools", default=POOLS_FILE, help="Path to pools_result.json")
    parser.add_argument("--out", default=OUT_FILE, help="Destination JSON for classification")
    parser.add_argument(
        "--rpc",
        default=os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
        help="Solana RPC endpoint",
    )
    return parser.parse_args(argv)


def iter_raydium_addresses(pools: dict):
    for asset, chains in pools.items():
        if not isinstance(chains, dict):
            continue
        for chain, entries in chains.items():
            if chain.lower() != "solana":
                continue
            if not isinstance(entries, list):
                continue
            for entry in entries:
                exch = (entry.get("биржа") or entry.get("exchange") or "").lower()
                if exch != "raydium":
                    continue
                address = (entry.get("контракт") or entry.get("contract") or "").strip()
                if not address:
                    continue
                pair = entry.get("пара") or entry.get("pair")
                url = entry.get("url")
                yield chain, asset, pair, address, url


def fetch_owner(address: str, rpc_url: str, cache: Dict[str, Tuple[Optional[str], Optional[str]]]) -> Tuple[Optional[str], Optional[str]]:
    if address in cache:
        return cache[address]
    ensure_requests()
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAccountInfo",
        "params": [address, {"encoding": "jsonParsed"}],
    }
    try:
        _log(f"rpc getAccountInfo -> {rpc_url} address={address}")
        resp = requests.post(rpc_url, json=payload, timeout=20)
    except Exception as exc:  # pragma: no cover - network failure
        err = str(exc)
        cache[address] = (None, err)
        return None, err
    if resp.status_code != 200:
        err = f"HTTP {resp.status_code}"
        cache[address] = (None, err)
        return None, err
    try:
        body = resp.json()
    except Exception as exc:
        err = f"invalid json: {exc}"
        cache[address] = (None, err)
        return None, err
    if "error" in body and body["error"]:
        err = str(body["error"])
        cache[address] = (None, err)
        return None, err
    value = body.get("result", {}).get("value")
    if value is None:
        cache[address] = (None, "account_not_found")
        return None, "account_not_found"
    owner = value.get("owner")
    cache[address] = (owner, None)
    return owner, None


def classify_owner(owner: Optional[str]) -> Tuple[str, Optional[str]]:
    if not owner:
        return "unknown", "owner missing"
    info = PROGRAM_OWNERS.get(owner)
    if info:
        version, note = info
        return version, note
    return "unknown", f"unrecognized owner {owner}"


def merge_results(out_path: str, new_items: List[dict], scanned: int) -> dict:
    combined: Dict[str, dict] = {}
    prev_total = 0
    if os.path.exists(out_path):
        try:
            prev = load_json(out_path)
            prev_items = prev.get("items") or []
            if isinstance(prev_items, list):
                for item in prev_items:
                    key = f"{item.get('chain')}::{item.get('address','').lower()}"
                    combined[key] = item
                prev_total = int(prev.get("total_scanned") or 0)
        except Exception as exc:
            _log(f"merge warning: failed to read previous file: {exc}")
    for item in new_items:
        key = f"{item.get('chain')}::{item.get('address','').lower()}"
        combined[key] = item
    out = {
        "updated_at": int(time.time()),
        "total_scanned": prev_total + scanned,
        "items": list(combined.values()),
    }
    save_json(out_path, out)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    global _log_fh
    try:
        _log_fh = open(LOG_FILE, "a", encoding="utf-8")
    except Exception:
        _log_fh = None

    pools = load_json(args.pools)
    entries = list(iter_raydium_addresses(pools))
    total = len(entries)
    _log(f"total raydium candidates: {total}; rpc={args.rpc}")

    cache: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    results: List[dict] = []
    start = time.time()

    for idx, (chain, asset, pair, address, url) in enumerate(entries, start=1):
        owner, err = fetch_owner(address, args.rpc, cache)
        base = {
            "chain": chain,
            "asset": asset,
            "pair": pair,
            "address": address,
            "url": url,
        }
        if err:
            base.update({"version": "error", "error": err})
            _log(f"{idx}/{total} error {address}: {err}")
        else:
            version, note = classify_owner(owner)
            item = {"version": version, "checks": {"owner": owner}}
            if note:
                item["note"] = note
            base.update(item)
            _log(f"{idx}/{total} [{version}] {address}")
        results.append(base)

    merge_results(args.out, results, total)
    elapsed = time.time() - start
    _log(f"Saved -> {args.out} ({len(results)} items) elapsed={elapsed:.1f}s")

    if _log_fh is not None:
        try:
            _log_fh.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
