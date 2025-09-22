# Классификация пулов DEX (V2/V3)

Скрипт `scripts/classify_pools.py` читает `pools_result.json`, выбирает EVM-пулы заданной биржи (по умолчанию Uniswap) и через публичные RPC делает `eth_call`:

- slot0() → V3
- getReserves() → V2
- иначе → unknown (или non-Uniswap)

V4: у Uniswap v4 ликвидность лежит у менеджера (PoolManager), адрес пула может быть 32-байтным идентификатором. В этом случае помечаем записи как `v4_pool_id`. Для других DEX 64-символьные идентификаторы получают маркер `hex_64_id` и требуют ручной проверки.

Поддерживаемые сети: ethereum, arbitrum, optimism, base, polygon, bsc, avalanche, cronos.

Как запустить

1) Зависимости

```bash
pip install -r requirements.txt
```

2) Ключи API (достаточно ETHERSCAN_API_KEY, но можно сетевые):

```bash
export ETHERSCAN_API_KEY=...
export ARBISCAN_API_KEY=...
export OPTIMISTIC_API_KEY=...
export BASESCAN_API_KEY=...
export POLYGONSCAN_API_KEY=...
export BSCSCAN_API_KEY=...
export SNOWTRACE_API_KEY=...
export CRONOSCAN_API_KEY=...
```

3) Запуск

```bash
# Uniswap (как раньше)
python scripts/classify_pools.py

# PancakeSwap → отдельный файл
python scripts/classify_pools.py --exchange pancakeswap --out classified_pools_pancake.json
```

Результаты

- Выход: `classified_pools.json`
- Кеш ответов: `tmp/etherscan_cache.json`

Примечания

- Есть троттлинг и экспоненциальный бэкофф, но всё равно учитывайте лимиты публичных API.
- Фильтрация: берём только выбранную биржу и валидные 0x-адреса (20 байт) либо 32-байтные идентификаторы (для Uniswap — `v4_pool_id`, для остальных — `hex_64_id`).
- Можно расширить проверками token0()/token1() и сигнатурами байткода.
