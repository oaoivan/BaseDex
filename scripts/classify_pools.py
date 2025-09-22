#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Классификация пулов DEX (V2/V3/unknown) по доступности методов через RPC.

Логика:
- slot0() -> V3
- getReserves() -> V2
- иначе -> unknown (например, V4 по адресу пула определить нельзя; см. README)

По умолчанию обрабатывает Uniswap, но через аргумент --exchange можно указать другую биржу
(например, PancakeSwap). Читает адреса из pools_result.json, фильтрует EVM + целевая биржа +
валидные 0x-адреса.

Результат: файл с классификацией (по умолчанию classified_pools.json).
"""

from __future__ import annotations

import argparse
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
    try:
        print(line, flush=True)
    except BrokenPipeError:
        # Игнорируем разрыв пайпа (например, при | head)
        pass
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


# Поддерживаемые сети (для метаданных; RPC берём отдельно)
CHAINS: Dict[str, ChainCfg] = {
    # L1 / L2 EVM совместимые сети, где встречается Uniswap
    "ethereum": ChainCfg("ethereum", "https://api.etherscan.io", "ETHERSCAN_API_KEY"),
    "arbitrum": ChainCfg("arbitrum", "https://api.arbiscan.io", "ARBISCAN_API_KEY"),
    "optimism": ChainCfg("optimism", "https://api-optimistic.etherscan.io", "OPTIMISTIC_API_KEY"),
    "base": ChainCfg("base", "https://api.basescan.org", "BASESCAN_API_KEY"),
    "polygon": ChainCfg("polygon", "https://api.polygonscan.com", "POLYGONSCAN_API_KEY"),
    "zksync": ChainCfg("zksync", "", "ZKSYNC_API_KEY"),  # RPC только через URL
    # ниже сети, где Uniswap, как правило, не развёрнут; оставим на случай редких записей
    "avalanche": ChainCfg("avalanche", "https://api.snowtrace.io", "SNOWTRACE_API_KEY"),
    "bsc": ChainCfg("bsc", "https://api.bscscan.com", "BSCSCAN_API_KEY"),
    "boba": ChainCfg("boba", "https://api.bobascan.com", "BOBASCAN_API_KEY"),
    "cronos": ChainCfg("cronos", "https://api.cronoscan.com", "CRONOSCAN_API_KEY"),
}


SLOT0_SELECTOR = "0x3850c7bd"  # slot0()
GET_RESERVES_SELECTOR = "0x0902f1ac"  # getReserves()
MASTER_DEPLOYER_SELECTOR = "0xee327147"  # masterDeployer()
BALANCER_WEIGHTS_SELECTOR = "0xf89f27ed"  # getNormalizedWeights()
BALANCER_AMP_SELECTOR = "0x6daccffa"  # getAmplificationParameter()
BALANCER_SCALING_SELECTOR = "0x1dd746ea"  # getScalingFactors()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_api_key(chain: ChainCfg) -> str:
    # Для Ethereum будем использовать RPC; ключ скана не требуется
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


def _extract_alchemy_key(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    # ожидаем .../v2/<KEY>[...]
    try:
        parts = url.split("/v2/")
        if len(parts) < 2:
            return None
        tail = parts[1]
        key = tail.split("/")[0].split("?")[0]
        return key or None
    except Exception:
        return None


ALCHEMY_HOSTS = {
    "ethereum": "eth-mainnet.g.alchemy.com",
    "arbitrum": "arb-mainnet.g.alchemy.com",
    "optimism": "opt-mainnet.g.alchemy.com",
    "polygon": "polygon-mainnet.g.alchemy.com",
    "base": "base-mainnet.g.alchemy.com",
}


def get_rpc_url(chain: str) -> Optional[str]:
    # 1) Явные ENV переменные для каждой сети
    env_var = f"{chain.upper()}_RPC_URL"
    val = os.getenv(env_var)
    if val:
        return val
    # 2) Универсальные ETHEREUM_RPC_URL/ALCHEMY_URL -> извлечь ключ и построить URL для поддерживаемых сетей Alchemy
    eth_rpc = os.getenv("ETHEREUM_RPC_URL") or os.getenv("ALCHEMY_URL")
    key = os.getenv("ALCHEMY_API_KEY") or _extract_alchemy_key(eth_rpc)
    if key and chain in ALCHEMY_HOSTS:
        return f"https://{ALCHEMY_HOSTS[chain]}/v2/{key}"
    # 3) иначе нет RPC
    return None


def rpc_call(chain: str, to: str, data: str, cache: Cache, rate_delay: float = RATE_DELAY) -> Tuple[Optional[str], Optional[str]]:
    """Возвращает (result_hex, error_str). Результат может быть '0x' при реентрации/реверте."""
    ensure_requests()
    rpc_url = get_rpc_url(chain)
    if not rpc_url:
        return None, f"RPC URL missing for {chain}"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": to, "data": data}, "latest"],
    }
    cache_key = json.dumps({"c": chain, "u": rpc_url, "b": payload}, sort_keys=True)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached.get("result"), cached.get("error")

    for attempt in range(4):
        try:
            _log(f"rpc eth_call[{chain}] -> {rpc_url} to={to} data={data[:10]}...")
            resp = requests.post(rpc_url, json=payload, timeout=20)
            if resp.status_code != 200:
                err = f"HTTP {resp.status_code}"
            else:
                j = resp.json()
                if "result" in j:
                    res = j.get("result")
                    _log(f"rpc eth_call[{chain}] <- len={len(res) if isinstance(res,str) else 'n/a'}")
                    cache.set(cache_key, {"result": res, "error": None})
                    time.sleep(rate_delay)
                    return res, None
                err = j.get("error", {}).get("message") or str(j)
                _log(f"rpc eth_call[{chain}] error: {err}")
        except Exception as e:
            err = str(e)
            _log(f"rpc eth_call[{chain}] exception: {err}")

        # Бэкофф
        backoff = rate_delay * (attempt + 1)
        _log(f"rpc eth_call[{chain}] retry in {backoff:.2f}s (attempt {attempt+1}/4)")
        time.sleep(backoff)

    cache.set(cache_key, {"result": None, "error": err})
    return None, err


def rpc_get_code(chain: str, address: str, cache: Cache, rate_delay: float = RATE_DELAY) -> Tuple[Optional[str], Optional[str]]:
    ensure_requests()
    rpc_url = get_rpc_url(chain)
    if not rpc_url:
        return None, f"RPC URL missing for {chain}"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_getCode",
        "params": [address, "latest"],
    }
    cache_key = json.dumps({"c": chain, "u": rpc_url, "b": payload}, sort_keys=True)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached.get("result"), cached.get("error")

    for attempt in range(4):
        try:
            _log(f"rpc eth_getCode[{chain}] -> {rpc_url} address={address}")
            resp = requests.post(rpc_url, json=payload, timeout=20)
            if resp.status_code != 200:
                err = f"HTTP {resp.status_code}"
            else:
                j = resp.json()
                if "result" in j:
                    res = j.get("result")
                    _log(f"rpc eth_getCode[{chain}] <- len={len(res) if isinstance(res,str) else 'n/a'}")
                    cache.set(cache_key, {"result": res, "error": None})
                    time.sleep(rate_delay)
                    return res, None
                err = j.get("error", {}).get("message") or str(j)
                _log(f"rpc eth_getCode[{chain}] error: {err}")
        except Exception as e:
            err = str(e)
            _log(f"rpc eth_getCode[{chain}] exception: {err}")
        backoff = rate_delay * (attempt + 1)
        _log(f"rpc eth_getCode[{chain}] retry in {backoff:.2f}s (attempt {attempt+1}/4)")
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


def looks_like_address_slot(result_hex: Optional[str]) -> bool:
    if not result_hex or not result_hex.startswith("0x"):
        return False
    return len(result_hex) >= 66


def has_payload(result_hex: Optional[str]) -> bool:
    return bool(result_hex and isinstance(result_hex, str) and result_hex.startswith("0x") and len(result_hex) > 2)


def classify_pool(
    chain: str,
    address: str,
    api_key: str,
    cache: Cache,
    exchange: str,
) -> Dict[str, object]:
    cfg = CHAINS[chain]
    checks = {}
    is_public = api_key == "YourApiKeyToken"
    _log(f"classify start {chain} {address} via RPC")
    # Получаем код
    code_hex, code_err = rpc_get_code(chain, address, cache)
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
    slot0_res, slot0_err = rpc_call(chain, address, SLOT0_SELECTOR, cache)
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
    reserves_res, reserves_err = rpc_call(chain, address, GET_RESERVES_SELECTOR, cache)
    if reserves_err:
        _log(f"getReserves error: {chain} {address} -> {reserves_err}")
    else:
        rlen = len(reserves_res) if isinstance(reserves_res, str) else 0
        _log(f"getReserves ok: {chain} {address} len={rlen}")
    checks["getReserves"] = reserves_res if reserves_err is None else f"error: {reserves_err}"
    if looks_like_v2_reserves(reserves_res):
        version = "v2"
        note: Optional[str] = None
        if exchange == "sushiswap":
            version = "classic"
            note = "SushiSwap Classic AMM pool"
        result = {"version": version, "checks": checks}
        if note:
            result["note"] = note
        return result

    if exchange == "sushiswap":
        _log(f"call masterDeployer {chain} {address}")
        master_res, master_err = rpc_call(chain, address, MASTER_DEPLOYER_SELECTOR, cache)
        if master_err:
            _log(f"masterDeployer error: {chain} {address} -> {master_err}")
            checks["masterDeployer"] = f"error: {master_err}"
        else:
            mlen = len(master_res) if isinstance(master_res, str) else 0
            _log(f"masterDeployer ok: {chain} {address} len={mlen}")
            checks["masterDeployer"] = master_res
            if looks_like_address_slot(master_res):
                return {
                    "version": "trident",
                    "checks": checks,
                    "note": "SushiSwap Trident pool",
                }

    if exchange == "balancer":
        _log(f"call getNormalizedWeights {chain} {address}")
        weights_res, weights_err = rpc_call(chain, address, BALANCER_WEIGHTS_SELECTOR, cache)
        if weights_err:
            _log(f"getNormalizedWeights error: {chain} {address} -> {weights_err}")
            checks["getNormalizedWeights"] = f"error: {weights_err}"
        else:
            wlen = len(weights_res) if isinstance(weights_res, str) else 0
            _log(f"getNormalizedWeights ok: {chain} {address} len={wlen}")
            checks["getNormalizedWeights"] = weights_res
            if has_payload(weights_res):
                return {
                    "version": "weighted",
                    "checks": checks,
                    "note": "Balancer Weighted pool",
                }

        _log(f"call getAmplificationParameter {chain} {address}")
        amp_res, amp_err = rpc_call(chain, address, BALANCER_AMP_SELECTOR, cache)
        if amp_err:
            _log(f"getAmplificationParameter error: {chain} {address} -> {amp_err}")
            checks["getAmplificationParameter"] = f"error: {amp_err}"
        else:
            alen = len(amp_res) if isinstance(amp_res, str) else 0
            _log(f"getAmplificationParameter ok: {chain} {address} len={alen}")
            checks["getAmplificationParameter"] = amp_res
            if has_payload(amp_res):
                return {
                    "version": "stable",
                    "checks": checks,
                    "note": "Balancer Stable/Composable pool",
                }

        _log(f"call getScalingFactors {chain} {address}")
        scaling_res, scaling_err = rpc_call(chain, address, BALANCER_SCALING_SELECTOR, cache)
        if scaling_err:
            _log(f"getScalingFactors error: {chain} {address} -> {scaling_err}")
            checks["getScalingFactors"] = f"error: {scaling_err}"
        else:
            slen = len(scaling_res) if isinstance(scaling_res, str) else 0
            _log(f"getScalingFactors ok: {chain} {address} len={slen}")
            checks["getScalingFactors"] = scaling_res
            if has_payload(scaling_res):
                return {
                    "version": "linear",
                    "checks": checks,
                    "note": "Balancer Linear pool",
                }

    return {"version": "unknown", "checks": checks}


def iter_exchange_evm_addresses(
    pools_json: dict,
    exchange: str,
    skip_chains: Optional[set] = None,
    include_chains: Optional[set] = None,
):
    skip_chains = skip_chains or set()
    include_chains = include_chains or set()
    exchange = exchange.lower()
    for asset, chains in pools_json.items():
        if not isinstance(chains, dict):
            continue
        for chain, entries in chains.items():
            # Берём только известные EVM сети
            if chain not in CHAINS:
                continue
            if include_chains and chain not in include_chains:
                continue
            if chain in skip_chains:
                continue
            for e in entries:
                try:
                    exch = e.get("биржа") or e.get("exchange") or ""
                    addr = (e.get("контракт") or e.get("contract") or "").strip()
                    if exch.lower() != exchange:
                        continue
                    # Для Balancer допускаем составные идентификаторы вида addr1-addr2-...
                    if exchange == "balancer" and "-" in addr:
                        yield chain, asset, pair, addr, url
                        continue
                    # Пропускаем non-EVM адреса, но отмечаем возможные v4 poolId (64 hex)
                    if not (ADDRESS20_RE.match(addr) or HEX32_RE.match(addr)):
                        continue
                    pair = e.get("пара") or e.get("pair")
                    url = e.get("url")
                    yield chain, asset, pair, addr, url
                except Exception:
                    continue


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Классификация пулов DEX по интерфейсу (V2/V3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exchange",
        default=os.getenv("TARGET_EXCHANGE", "uniswap"),
        help="Название биржи в pools_result.json (регистронезависимо)",
    )
    parser.add_argument(
        "--pools",
        default=POOLS_FILE,
        help="Путь до pools_result.json",
    )
    parser.add_argument(
        "--out",
        default=OUT_FILE,
        help="Файл для сохранения результатов",
    )
    return parser.parse_args(argv)


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    exchange = (args.exchange or "").strip().lower()
    if not exchange:
        raise ValueError("exchange must be non-empty")

    # Открываем файл логов
    global _log_fh
    try:
        _log_fh = open(LOG_FILE, "a", encoding="utf-8")
    except Exception:
        _log_fh = None

    pools_path = resolve_path(args.pools)
    out_path = resolve_path(args.out)
    pools = load_json(pools_path)
    cache = Cache(CACHE_FILE)
    results: List[dict] = []

    # Фильтр по сети через env ONLY_CHAIN (comma-separated). Если не задан — все, кроме Ethereum
    only = os.getenv("ONLY_CHAIN", "").strip()
    include_set: Optional[set] = None
    if only:
        include_set = {c.strip().lower() for c in only.split(",") if c.strip()}
    default_skip = set()
    if not include_set and exchange == "uniswap":
        default_skip = {"ethereum"}
    items = list(
        iter_exchange_evm_addresses(
            pools,
            exchange=exchange,
            skip_chains=default_skip,
            include_chains=include_set,
        )
    )
    total = len(items)
    _log(f"total candidates: {total}; exchange={exchange}; rate_delay={RATE_DELAY}s")
    start_ts = time.time()
    processed = 0

    pool_id_version = "v4_pool_id" if exchange == "uniswap" else "hex_64_id"
    pool_id_note = (
        "64-hex pool identifier; для v4 проверка идёт по PoolManager, а не по адресу пула"
        if exchange == "uniswap"
        else "64-hex identifier (non-address); требуется ручная проверка"
    )
    for chain, asset, pair, addr, url in items:
        processed += 1
        cfg = CHAINS[chain]
        # Проверка наличия RPC для сети заранее
        rpc_url = get_rpc_url(chain)
        if not rpc_url:
            info = {"version": "skipped_no_rpc", "checks": {}, "note": f"no RPC URL for {chain}"}
            results.append({
                "chain": chain,
                "asset": asset,
                "pair": pair,
                "address": addr,
                "url": url,
                **info,
            })
            _log(f"{processed}/{total} skipped_no_rpc {chain} {pair} {addr}")
            continue
        if exchange == "balancer" and "-" in addr:
            info = {
                "version": "balancer_pool_id",
                "checks": {},
                "note": "Balancer multi-address pool identifier; requires manual mapping",
            }
            results.append({
                "chain": chain,
                "asset": asset,
                "pair": pair,
                "address": addr,
                "url": url,
                **info,
            })
            _log(f"{processed}/{total} balancer_pool_id {chain} {pair} {addr}")
            continue
        # Если адрес выглядит как 32-байтный идентификатор (poolId), отметим как потенциальный v4
        if HEX32_RE.match(addr):
            info = {"version": pool_id_version, "checks": {}, "note": pool_id_note}
            results.append({
                "chain": chain,
                "asset": asset,
                "pair": pair,
                "address": addr,
                "url": url,
                **info,
            })
            _log(f"{processed}/{total} {pool_id_version} {chain} {pair} {addr}")
            continue

        api_key = get_api_key(cfg)
        try:
            info = classify_pool(chain, addr, api_key, cache, exchange)
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
    # Слияние с существующим файлом (сохранить результаты Ethereum) c дедупликацией по (chain,address)
    combined_map: Dict[str, dict] = {}
    prev_total = 0
    if os.path.exists(out_path):
        try:
            prev = load_json(out_path)
            prev_items = prev.get("items") or []
            if isinstance(prev_items, list):
                for it in prev_items:
                    addr_key = (it.get('address') or '').lower()
                    key = f"{it.get('chain')}::{addr_key}"
                    combined_map[key] = it
                prev_total = int(prev.get("total_scanned") or 0)
        except Exception as e:
            _log(f"merge warning: failed to read previous OUT_FILE: {e}")
    for it in results:
        addr_key = (it.get('address') or '').lower()
        key = f"{it.get('chain')}::{addr_key}"
        combined_map[key] = it

    out = {"updated_at": int(time.time()), "total_scanned": prev_total + total, "items": list(combined_map.values())}
    save_json(out_path, out)
    cache.flush()
    _log(f"Saved -> {out_path} ({len(results)} items)")
    try:
        if _log_fh is not None:
            _log_fh.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
