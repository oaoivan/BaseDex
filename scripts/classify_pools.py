#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Классификация пулов Uniswap (V2/V3/unknown) по доступности методов через Etherscan-подобные API.

Логика:
- slot0() -> V3
- getReserves() -> V2
- иначе -> unknown (V4 по адресу пула определить нельзя; см. README)

Читает адреса из pools_result.json, фильтрует EVM + биржа == 'uniswap' + валидные 0x-адреса.
Результат: classified_pools.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:
    requests = None


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POOLS_FILE = os.path.join(ROOT_DIR, "pools_result.json")
OUT_FILE = os.path.join(ROOT_DIR, "classified_pools.json")
CACHE_FILE = os.path.join(ROOT_DIR, "tmp", "etherscan_cache.json")
LOG_FILE = os.path.join(ROOT_DIR, "term.txt")

# Глобальные настройки
try:
    RATE_DELAY = float(os.getenv("ETHERSCAN_RATE_DELAY", "0.5"))
except Exception:
    RATE_DELAY = 0.5

_log_fh = None  # type: ignore

def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {msg}"
    print(line, flush=True)
    try:
        global _log_fh
        if _log_fh is not None:
            _log_fh.write(line + "\n")
            _log_fh.flush()
    except Exception:
        pass

ADDRESS20_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
HEX32_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")  # возможный poolId (например, Uniswap v4)


@dataclass
class ChainCfg:
    name: str
    api_base: str
    api_key_env: str  # конкретная переменная окружения, fallback на ETHERSCAN_API_KEY


# Поддерживаемые EVM-сканы (формат Etherscan)
CHAINS: Dict[str, ChainCfg] = {
    # L1
    "ethereum": ChainCfg("ethereum", "https://api.etherscan.io", "ETHERSCAN_API_KEY"),
    # L2
    "arbitrum": ChainCfg("arbitrum", "https://api.arbiscan.io", "ARBISCAN_API_KEY"),
    "optimism": ChainCfg("optimism", "https://api-optimistic.etherscan.io", "OPTIMISTIC_API_KEY"),
    "base": ChainCfg("base", "https://api.basescan.org", "BASESCAN_API_KEY"),
    # Sidechains
    "polygon": ChainCfg("polygon", "https://api.polygonscan.com", "POLYGONSCAN_API_KEY"),
    "bsc": ChainCfg("bsc", "https://api.bscscan.com", "BSCSCAN_API_KEY"),
    "avalanche": ChainCfg("avalanche", "https://api.snowtrace.io", "SNOWTRACE_API_KEY"),
    "cronos": ChainCfg("cronos", "https://api.cronoscan.com", "CRONOSCAN_API_KEY"),
    # Добавляйте по мере необходимости
}


SLOT0_SELECTOR = "0x3850c7bd"  # slot0()
GET_RESERVES_SELECTOR = "0x0902f1ac"  # getReserves()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_api_key(chain: ChainCfg) -> str:
    # Публичный режим по умолчанию (низкий лимит): 'YourApiKeyToken'
    return os.getenv(chain.api_key_env) or os.getenv("ETHERSCAN_API_KEY") or "YourApiKeyToken"


def ensure_requests():
    if requests is None:
        raise RuntimeError("Требуется пакет 'requests'. Установите зависимости из requirements.txt")


class Cache:
    def __init__(self, path: str) -> None:
        self.path = path
        self._data: Dict[str, dict] = {}
        if os.path.exists(path):
            try:
                self._data = load_json(path)
            except Exception:
                self._data = {}

    def get(self, key: str) -> Optional[dict]:
        return self._data.get(key)

    def set(self, key: str, val: dict) -> None:
        self._data[key] = val

    def flush(self) -> None:
        save_json(self.path, self._data)


def etherscan_call(chain: ChainCfg, to: str, data: str, api_key: str, cache: Cache, rate_delay: float = RATE_DELAY) -> Tuple[Optional[str], Optional[str]]:
    """Возвращает (result_hex, error_str). Результат может быть '0x' при реентрации/реверте."""
    ensure_requests()
    url = f"{chain.api_base}/api"
    params = {
        "module": "proxy",
        "action": "eth_call",
        "to": to,
        "data": data,
        "tag": "latest",
        "apikey": api_key,
    }
    cache_key = json.dumps({"u": url, "p": params}, sort_keys=True)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached.get("result"), cached.get("error")

    for attempt in range(4):
        try:
            safe_params = {k: ("***" if k == "apikey" else v) for k, v in params.items()}
            _log(f"eth_call -> {url} params={safe_params}")
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code != 200:
                err = f"HTTP {resp.status_code}"
            else:
                j = resp.json()
                # Etherscan proxy: { status?, message?, result }
                if "result" in j:
                    res = j.get("result")
                    if isinstance(res, str):
                        _log(f"eth_call <- 200 len={len(res)}")
                    else:
                        _log("eth_call <- 200 (non-str result)")
                    cache.set(cache_key, {"result": res, "error": None})
                    time.sleep(rate_delay)
                    return res, None
                err = j.get("message") or j.get("error", {}).get("message") or str(j)
                _log(f"eth_call error: {err}")
        except Exception as e:
            err = str(e)
            _log(f"eth_call exception: {err}")

        # Бэкофф
        backoff = rate_delay * (attempt + 1)
        _log(f"eth_call retry in {backoff:.2f}s (attempt {attempt+1}/4)")
        time.sleep(backoff)

    cache.set(cache_key, {"result": None, "error": err})
    return None, err


def etherscan_get_code(chain: ChainCfg, address: str, api_key: str, cache: Cache, rate_delay: float = RATE_DELAY) -> Tuple[Optional[str], Optional[str]]:
    ensure_requests()
    url = f"{chain.api_base}/api"
    params = {
        "module": "proxy",
        "action": "eth_getCode",
        "address": address,
        "tag": "latest",
        "apikey": api_key,
    }
    cache_key = json.dumps({"u": url, "p": params}, sort_keys=True)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached.get("result"), cached.get("error")

    for attempt in range(4):
        try:
            safe_params = {k: ("***" if k == "apikey" else v) for k, v in params.items()}
            _log(f"eth_getCode -> {url} params={safe_params}")
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code != 200:
                err = f"HTTP {resp.status_code}"
            else:
                j = resp.json()
                if "result" in j:
                    res = j.get("result")
                    if isinstance(res, str):
                        _log(f"eth_getCode <- 200 len={len(res)}")
                    else:
                        _log("eth_getCode <- 200 (non-str result)")
                    cache.set(cache_key, {"result": res, "error": None})
                    time.sleep(rate_delay)
                    return res, None
                err = j.get("message") or j.get("error", {}).get("message") or str(j)
                _log(f"eth_getCode error: {err}")
        except Exception as e:
            err = str(e)
            _log(f"eth_getCode exception: {err}")
        backoff = rate_delay * (attempt + 1)
        _log(f"eth_getCode retry in {backoff:.2f}s (attempt {attempt+1}/4)")
        time.sleep(backoff)

    cache.set(cache_key, {"result": None, "error": err})
    return None, err


def looks_like_v3_slot0(result_hex: Optional[str]) -> bool:
    if not result_hex or not result_hex.startswith("0x"):
        return False
    # Ожидаем 7 слов по 32 байта => 224 байта => 448 hex + '0x' = 450
    return len(result_hex) >= 450


def looks_like_v2_reserves(result_hex: Optional[str]) -> bool:
    if not result_hex or not result_hex.startswith("0x"):
        return False
    # Ожидаем 3 слова по 32 байта => 96 байт => 192 hex + '0x' = 194
    return len(result_hex) >= 194


def classify_pool(chain: str, address: str, api_key: str, cache: Cache) -> Dict[str, object]:
    cfg = CHAINS[chain]
    checks = {}
    is_public = api_key == "YourApiKeyToken"
    _log(f"classify start {chain} {address} via {cfg.api_base} (key={'public' if is_public else 'custom'})")
    code_hex, code_err = etherscan_get_code(cfg, address, api_key, cache)
    if code_err:
        _log(f"getCode error: {chain} {address} -> {code_err}")
    else:
        clen = len(code_hex) if isinstance(code_hex, str) else 0
        preview = (code_hex[:10] + '...') if isinstance(code_hex, str) else str(code_hex)
        _log(f"getCode ok: {chain} {address} len={clen} {preview}")
    if code_hex == "0x":
        return {"version": "unknown", "checks": {"code": code_hex}, "note": "no bytecode (EOA?)"}

    # V3?
    _log(f"call slot0 {chain} {address}")
    slot0_res, slot0_err = etherscan_call(cfg, address, SLOT0_SELECTOR, api_key, cache)
    if slot0_err:
        _log(f"slot0 error: {chain} {address} -> {slot0_err}")
    else:
        slen = len(slot0_res) if isinstance(slot0_res, str) else 0
        _log(f"slot0 ok: {chain} {address} len={slen}")
    checks["slot0"] = slot0_res if slot0_err is None else f"error: {slot0_err}"
    if looks_like_v3_slot0(slot0_res):
        return {"version": "v3", "checks": checks}

    # V2?
    _log(f"call getReserves {chain} {address}")
    reserves_res, reserves_err = etherscan_call(cfg, address, GET_RESERVES_SELECTOR, api_key, cache)
    if reserves_err:
        _log(f"getReserves error: {chain} {address} -> {reserves_err}")
    else:
        rlen = len(reserves_res) if isinstance(reserves_res, str) else 0
        _log(f"getReserves ok: {chain} {address} len={rlen}")
    checks["getReserves"] = reserves_res if reserves_err is None else f"error: {reserves_err}"
    if looks_like_v2_reserves(reserves_res):
        return {"version": "v2", "checks": checks}

    return {"version": "unknown", "checks": checks}


def iter_uniswap_evm_addresses(pools_json: dict):
    for asset, chains in pools_json.items():
        if not isinstance(chains, dict):
            continue
        for chain, entries in chains.items():
            if chain not in CHAINS:
                continue
            for e in entries:
                try:
                    exch = e.get("биржа") or e.get("exchange") or ""
                    addr = (e.get("контракт") or e.get("contract") or "").strip()
                    if exch.lower() != "uniswap":
                        continue
                    # Пропускаем non-EVM адреса, но отмечаем возможные v4 poolId (64 hex)
                    if not (ADDRESS20_RE.match(addr) or HEX32_RE.match(addr)):
                        continue
                    pair = e.get("пара") or e.get("pair")
                    url = e.get("url")
                    yield chain, asset, pair, addr, url
                except Exception:
                    continue


def main() -> int:
    # Открываем файл логов
    global _log_fh
    try:
        _log_fh = open(LOG_FILE, "a", encoding="utf-8")
    except Exception:
        _log_fh = None

    pools = load_json(POOLS_FILE)
    cache = Cache(CACHE_FILE)
    results: List[dict] = []

    items = list(iter_uniswap_evm_addresses(pools))
    total = len(items)
    _log(f"total candidates: {total}; rate_delay={RATE_DELAY}s")
    start_ts = time.time()
    processed = 0
    for chain, asset, pair, addr, url in items:
        processed += 1
        cfg = CHAINS[chain]
        # Если адрес выглядит как 32-байтный идентификатор (poolId), отметим как потенциальный v4
        if HEX32_RE.match(addr):
            info = {"version": "v4_pool_id", "checks": {}, "note": "64-hex pool identifier; для v4 проверка идёт по PoolManager, а не по адресу пула"}
            results.append({
                "chain": chain,
                "asset": asset,
                "pair": pair,
                "address": addr,
                "url": url,
                **info,
            })
            _log(f"{processed}/{total} v4_pool_id {chain} {pair} {addr}")
            continue

        api_key = get_api_key(cfg)
        try:
            info = classify_pool(chain, addr, api_key, cache)
            results.append({
                "chain": chain,
                "asset": asset,
                "pair": pair,
                "address": addr,
                "url": url,
                **info,
            })
            elapsed = time.time() - start_ts
            per = elapsed / processed if processed else 0
            eta = per * (total - processed)
            _log(f"{processed}/{total} [{info['version']}] {chain} {pair} {addr} | avg={per:.2f}s ETA~{eta:.1f}s")
        except Exception as e:
            results.append({
                "chain": chain,
                "asset": asset,
                "pair": pair,
                "address": addr,
                "url": url,
                "version": "error",
                "error": str(e),
            })
            _log(f"error {chain} {addr}: {e}")

    # Сохранение
    out = {"updated_at": int(time.time()), "total_scanned": total, "items": results}
    save_json(OUT_FILE, out)
    cache.flush()
    _log(f"Saved -> {OUT_FILE} ({len(results)} items)")
    try:
        if _log_fh is not None:
            _log_fh.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
