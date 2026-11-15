# FreeModelAggregator

[![free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/yuujiitadori1/FreeModelAggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/yuujiitadori1/FreeModelAggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

The best free AI model aggregator: one server that speaks the OpenAI Chat Completions API and the Anthropic Messages API (JSON and SSE) and answers every request with the highest-rated free model that meets it, across ten free providers on three wires. It reads each provider's own catalog, routes by the sealed quality table, fails over when a provider refuses, and reserves the providers a later constrained request will need — so a vision or long-context request later in the trial still gets its best model. Zero dependencies, standard library only, fully deterministic: no wall-clock time, no unseeded randomness.

## Run it

```sh
node main.ts --list
```

Set `MAKEFUN_PROMPT_FILE` to the trial brief and `MAKEFUN_ENV_URL` to the trial env, then run `node main.ts` to register and serve the trial; it exits 0 when the env says the workload is done.

Entered on [make.fun](https://make.fun/?ref=08hb5yc1&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards
