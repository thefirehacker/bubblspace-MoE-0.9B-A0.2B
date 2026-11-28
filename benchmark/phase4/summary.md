# Phase 3 SFT vs Phase 4 RL v1: chat benchmarks (vLLM backend, greedy, A10, 28 Nov)

| Model | Harmless refused | Harmful refused | Live data honest | HumanEval % | MBPP % | GSM8K % | IFEval prompt % | IFEval instr. % | BFCL simple % | BFCL multiple % | BFCL parallel % | BFCL no-tool % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT run 01 | 9/45 | 8/8 | 6/8 | 15.2 | 22.6 | 10.0 | 26.1 | 38.1 | 57.0 | 40.0 | 31.0 | 75.4 |
| SFT run 02b | 1/45 | 8/8 | 7/8 | 15.2 | 21.8 | 9.6 | 23.1 | 36.0 | 63.0 | 36.0 | 24.0 | 56.2 |
| RL v1 step 100 | 1/45 | 7/8 | 8/8 | 15.9 | 22.4 | 10.5 | 28.1 | 40.3 | 68.0 | 42.0 | 37.0 | 71.7 |
| RL v1 step 300 | 0/45 | 7/8 | 8/8 | 15.9 | 22.8 | 12.3 | 27.9 | 41.8 | 70.0 | 37.0 | 34.0 | 75.4 |
| RL v1 step 500 | 0/45 | 7/8 | 8/8 | 16.5 | 22.2 | 12.4 | 29.6 | 43.5 | 71.0 | 43.0 | 40.0 | 69.6 |

English 6-task (forgetting check):
- SFT run 02b: {"hellaswag": {"name": "hellaswag", "alias": "hellaswag", "sample_len": 10042, "acc,none": 0.3461461860187214, "acc_stderr,none": 0.004747682003491476, "acc_norm,none": 0.4135630352519418, "acc_norm_stderr,none": 0.004914655063329509}, "piqa": {"name": "piqa", "alias": "piqa", "sample_len": 1838, "acc,none": 0.6626768226332971, "acc_stderr,none": 0.0110311147850597, "acc_norm,none": 0.6621327529923831, "acc_norm_stderr,none": 0.011035474307853845}, "arc_easy": {"name": "arc_easy", "alias": "arc_easy", "sample_len": 2376, "acc,none": 0.6220538720538721, "acc_stderr,none": 0.009949405744045459, 
- RL v1 step 500: {"hellaswag": {"name": "hellaswag", "alias": "hellaswag", "sample_len": 10042, "acc,none": 0.3466440948018323, "acc_stderr,none": 0.004749286071559571, "acc_norm,none": 0.4148575980880303, "acc_norm_stderr,none": 0.004916905095810846}, "piqa": {"name": "piqa", "alias": "piqa", "sample_len": 1838, "acc,none": 0.6637649619151251, "acc_stderr,none": 0.01102234670897024, "acc_norm,none": 0.6664853101196954, "acc_norm_stderr,none": 0.011000139592184575}, "arc_easy": {"name": "arc_easy", "alias": "arc_easy", "sample_len": 2376, "acc,none": 0.6287878787878788, "acc_stderr,none": 0.009913599001845743,
