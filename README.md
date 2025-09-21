# Классификация пулов Uniswap (V2/V3)

Скрипт `scripts/classify_pools.py` читает `pools_result.json`, выбирает EVM-пулы Uniswap и через публичные Etherscan-подобные API делает `eth_call`:

- slot0() → V3
- getReserves() → V2
- иначе → unknown (или non-Uniswap)

V4: у Uniswap v4 ликвидность лежит у менеджера (PoolManager), адрес пула может быть 32-байтным идентификатором. В этом случае помечаем записи как `v4_pool_id`. Для строгой проверки нужно знать адрес соответствующего PoolManager и формат хука.

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
python scripts/classify_pools.py
```

Результаты

- Выход: `classified_pools.json`
- Кеш ответов: `tmp/etherscan_cache.json`

Примечания

- Есть троттлинг и экспоненциальный бэкофф, но всё равно учитывайте лимиты публичных API.
- Фильтрация: берём только биржа == "uniswap" и валидные 0x-адреса (20 байт) либо 32-байтные идентификаторы (помечаются как v4_pool_id).
- Можно расширить проверками token0()/token1() и сигнатурами байткода.
