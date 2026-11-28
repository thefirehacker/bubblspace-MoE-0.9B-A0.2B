# Phase 3 SFT vs Phase 4 RL v1: chat benchmarks (vLLM backend, greedy, A10, 28 Nov)

| Model | Harmless refused | Harmful refused | Live data honest | HumanEval % | MBPP % | GSM8K % | IFEval prompt % | IFEval instr. % | BFCL simple % | BFCL multiple % | BFCL parallel % | BFCL no-tool % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT run 01 | 9/45 | 8/8 | 6/8 | 15.2 | 22.6 | 10.0 | 26.1 | 38.1 | 57.0 | 40.0 | 31.0 | 75.4 |
| SFT run 02b | 1/45 | 8/8 | 7/8 | 15.2 | 21.8 | 9.6 | 23.1 | 36.0 | 63.0 | 36.0 | 24.0 | 56.2 |
| RL v1 step 100 | 1/45 | 7/8 | 8/8 | 15.9 | 22.4 | 10.5 | 28.1 | 40.3 | 68.0 | 42.0 | 37.0 | 71.7 |
| RL v1 step 300 | 0/45 | 7/8 | 8/8 | 15.9 | 22.8 | 12.3 | 27.9 | 41.8 | 70.0 | 37.0 | 34.0 | 75.4 |
| RL v1 step 500 | 0/45 | 7/8 | 8/8 | 16.5 | 22.2 | 12.4 | 29.6 | 43.5 | 71.0 | 43.0 | 40.0 | 69.6 |

English 6-task (forgetting check):

| Model | HellaSwag | PIQA | ARC-e | ARC-c | WinoGrande | LAMBADA | Average |
|---|---|---|---|---|---|---|---|
| SFT run 02b | 41.4 | 66.2 | 58.5 | 30.6 | 54.6 | 36.1 | **47.91** |
| RL v1 step 500 | 41.5 | 66.6 | 59.6 | 30.5 | 54.3 | 35.5 | **48.00** |
