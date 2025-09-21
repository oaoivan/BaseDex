#!/usr/bin/env python3
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import logging


WORKDIR = Path(__file__).resolve().parents[1]
SRC_JSON = WORKDIR / "all_contracts_merged_reformatted.json"
OUT_JSON = WORKDIR / "pools_result.json"
logger = logging.getLogger("dex_pools")


# Ранее использовалась карта соответствия сетей, теперь пытаемся работать с любым chain из входных данных напрямую.
CHAIN_MAP = None  # оставлено для совместимости, но не используется


def http_get(url: str, timeout: float = 20.0, retries: int = 3, backoff: float = 1.5) -> bytes:
    # Simple global rate limiter (min interval between requests)
    global _LAST_REQ_TIME
    MIN_INTERVAL = 0.25  # ~4 req/s, below 300 rpm limit
    last_err = None
    for attempt in range(retries):
        try:
            # enforce min interval between requests
            now = time.time()
            if _LAST_REQ_TIME is not None:
                delta = now - _LAST_REQ_TIME
                if delta < MIN_INTERVAL:
                    sleep_for = MIN_INTERVAL - delta
                    logger.debug(f"Rate limit: sleep {sleep_for:.3f}s")
                    time.sleep(sleep_for)
            logger.debug(f"HTTP GET {url}")
            req = urllib.request.Request(url, headers={"Accept": "*/*", "User-Agent": "DexPoolsFetcher/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                _LAST_REQ_TIME = time.time()
                logger.debug(f"HTTP OK {len(data)} bytes")
                return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # rate limit: backoff
                logger.warning(f"429 Too Many Requests, retry {attempt+1}/{retries}")
                time.sleep(backoff * (attempt + 1))
                last_err = e
                continue
            last_err = e
        except Exception as e:
            logger.warning(f"HTTP error: {e} (attempt {attempt+1}/{retries})")
            last_err = e
        time.sleep(backoff * (attempt + 1))
    raise last_err


def chunked(seq: List[str], n: int) -> List[List[str]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def parse_tokens(src: Dict[str, Any], tokens_filter: Set[str] = None) -> Dict[str, Dict[str, Set[str]]]:
    """
    Return mapping: tokenSymbol -> chainId(repo) -> set(addresses)
    It merges addresses from all CEX keys (binance/bitget/...). Skips meta keys starting with '_'.
    """
    out: Dict[str, Dict[str, Set[str]]] = {}
    for symbol, body in src.items():
        if tokens_filter and symbol not in tokens_filter:
            continue
        chains: Dict[str, Set[str]] = {}
        if isinstance(body, dict):
            for provider_key, provider_val in body.items():
                if provider_key.startswith("_"):
                    continue
                if not isinstance(provider_val, dict):
                    continue
                for chain_key, addr in provider_val.items():
                    if not isinstance(addr, str):
                        continue
                    chains.setdefault(chain_key, set()).add(addr)
        if chains:
            out[symbol] = chains
            logger.info(f"Token {symbol}: chains={list(chains.keys())}")
    return out


def fetch_pairs_for_chain(chain: str, token_addresses: List[str]) -> List[Dict[str, Any]]:
    """Use tokens/v1 batch endpoint (up to 30 addresses). Returns list of pair objects."""
    # Пытаемся запрашивать по переданному chain как есть
    ds_chain = chain
    pairs: List[Dict[str, Any]] = []
    for batch in chunked(token_addresses, 30):
        joined = ",".join(batch)
        url = f"https://api.dexscreener.com/tokens/v1/{ds_chain}/{joined}"
        logger.info(f"Fetch tokens batch: chain={chain} addrs={len(batch)} url={url}")
        try:
            data = http_get(url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.info(f"404 Not Found tokens batch, skip: {url}")
                continue
            logger.warning(f"HTTP {e.code} on tokens batch, skip: {url}")
            continue
        except Exception as e:
            logger.warning(f"Error fetching tokens batch {url}: {e}")
            continue
        try:
            arr = json.loads(data.decode("utf-8"))
        except Exception:
            logger.warning("JSON parse error on tokens batch")
            continue
        if isinstance(arr, list):
            pairs.extend(arr)
            logger.info(f"Pairs fetched: {len(arr)} (total {len(pairs)})")
    return pairs


def filter_and_shape_pairs(symbol: str, chain: str, pairs: List[Dict[str, Any]], min_liq_usd: float) -> List[Dict[str, Any]]:
    out = []
    for p in pairs:
        try:
            liq = float(p.get("liquidity", {}).get("usd", 0) or 0)
        except Exception:
            liq = 0.0
        if liq < min_liq_usd:
            continue
        base = p.get("baseToken", {})
        quote = p.get("quoteToken", {})
        base_sym = base.get("symbol") or symbol
        quote_sym = quote.get("symbol") or "?"
        pair_str = f"{base_sym}/{quote_sym}"
        out.append({
            "биржа": p.get("dexId"),
            "пара": pair_str,
            "контракт": p.get("pairAddress"),
            "url": p.get("url"),
            "ликвидность_usd": liq,
        })
    return out


def integrate_native_pools(symbol: str, body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse _quoted_to_native_pools entries and return minimal info with parsed chain/addr."""
    res: List[Dict[str, Any]] = []
    pools = body.get("_quoted_to_native_pools")
    if not isinstance(pools, list):
        return res
    for item in pools:
        if not isinstance(item, dict):
            continue
        url = item.get("pool_url")
        pair_addr = None
        chain_from_url = None
        if isinstance(url, str) and url.startswith("https://dexscreener.com/"):
            try:
                parts = url.split("/")
                # https://dexscreener.com/{chain}/{pairAddress}
                chain_from_url = parts[3]
                pair_addr = parts[4]
            except Exception:
                pass
        res.append({
            "url": url,
            "контракт": pair_addr,
            "цепочка": chain_from_url,
            "из_исходного_списка": True,
        })
    logger.info(f"Native pools for {symbol}: {len(res)}")
    return res


def fetch_pair_details(chain: str, pair_address: str) -> Dict[str, Any] | None:
    # Пытаемся запрашивать по переданному chain как есть
    ds_chain = chain
    if not pair_address:
        return None
    url = f"https://api.dexscreener.com/latest/dex/pairs/{ds_chain}/{pair_address}"
    try:
        data = http_get(url)
        obj = json.loads(data.decode("utf-8"))
        pairs = obj.get("pairs") if isinstance(obj, dict) else None
        if isinstance(pairs, list) and pairs:
            logger.info(f"Pair details ok: {chain}/{pair_address}")
            return pairs[0]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.info(f"404 pair details, skip: {chain}/{pair_address}")
            return None
        logger.warning(f"HTTP {e.code} pair details failed: {chain}/{pair_address}")
        return None
    except Exception as e:
        logger.warning(f"Pair details failed: {chain}/{pair_address}: {e}")
        return None
    return None


def build(symbols_filter: Set[str] = None, min_liq_usd: float = 100000.0) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    with open(SRC_JSON, "r", encoding="utf-8") as f:
        src = json.load(f)

    token_map = parse_tokens(src, tokens_filter=symbols_filter)

    result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for symbol, chains in token_map.items():
        body = src.get(symbol, {}) if isinstance(src, dict) else {}
        # Pre-attach native pools (will be distributed to their chain if detected)
        native_pools = integrate_native_pools(symbol, body)

        for chain, addrs_set in chains.items():
            addrs = sorted(addrs_set)
            logger.info(f"Process {symbol} on {chain}: addrs={len(addrs)}")
            pairs = fetch_pairs_for_chain(chain, addrs)
            shaped = filter_and_shape_pairs(symbol, chain, pairs, min_liq_usd)
            if shaped:
                logger.info(f"Kept pools (≥{min_liq_usd:.0f}) for {symbol}/{chain}: {len(shaped)}")

            # Enrich native pools for this chain via pair-details and apply liquidity filter
            to_merge: List[Dict[str, Any]] = []
            for np in native_pools:
                if np.get("цепочка") != chain:
                    continue
                pair_addr = np.get("контракт")
                details = fetch_pair_details(chain, pair_addr) if pair_addr else None
                if details:
                    try:
                        liq = float(details.get("liquidity", {}).get("usd", 0) or 0)
                    except Exception:
                        liq = 0.0
                    if liq < min_liq_usd:
                        continue
                    base = details.get("baseToken", {})
                    quote = details.get("quoteToken", {})
                    pair_str = f"{base.get('symbol')}/{quote.get('symbol')}"
                    to_merge.append({
                        "биржа": details.get("dexId"),
                        "пара": pair_str,
                        "контракт": pair_addr,
                        "url": np.get("url"),
                        "ликвидность_usd": liq,
                        "из_исходного_списка": True,
                    })
                else:
                    # If no details, skip (can't evaluate liquidity and dex)
                    continue
            if to_merge:
                logger.info(f"Added native pools merged for {symbol}/{chain}: {len(to_merge)}")

            if shaped or to_merge:
                result.setdefault(symbol, {}).setdefault(chain, [])
                result[symbol][chain].extend(shaped)
                result[symbol][chain].extend(to_merge)

        # Примечание: теперь не фильтруем по списку поддерживаемых сетей — пробуем все, что есть во входном JSON.

    return result


def main(argv: List[str]) -> int:
    # Optional: pass symbols as CLI args to limit scope, e.g., python build_pools.py RLC TRX
    # Инициализация логирования: в консоль и файл одновременно
    logger.setLevel(logging.INFO)
    # Избежать дублирования хендлеров при повторном запуске
    if not logger.handlers:
        fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        # Консоль
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        # Файл
        fh_path = WORKDIR / 'run_all.log'
        fh = logging.FileHandler(fh_path, encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    symbols_filter: Set[str] = set(argv[1:]) if len(argv) > 1 else None
    out = build(symbols_filter=symbols_filter, min_liq_usd=100000.0)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote {OUT_JSON} with {len(out)} token entries")
    return 0


if __name__ == "__main__":
    _LAST_REQ_TIME = None
    raise SystemExit(main(sys.argv))
