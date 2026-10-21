# Phase 5: Nemotron, unavailable results

| Collection | Result |
| --- | --- |
| Model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` |
| Revision | `bee7596271d1495f6992ae224aefde4410e816b8` |
| Completed main replies | 0 / 3,200 |
| Runtime smoke | 0 / 8 |
| Full-batch throughput check | 0 / 256 |
| Startup allocation | One startup rental and one fix rental exhausted |
| First stop | Provider CLI/API failure before worker deployment |
| Fix stop | Remote pip rejected pinned vLLM 0.32.0; interpreter/index cause unconfirmed |
| GLM fallback | Not selected; neither attempt entered Nemotron load/generate, so registered runtime-failure proof is absent |
| H20 core, H21 main, H20 controls | All unreleased; no measured projection exists |
| Registered budget drops | None; collection stopped at startup |
| Inference | Unavailable; no complete hypothesis collected |

| Cell | Replies / required |
| --- | ---: |
| prefill/empty | 0 / 400 |
| prefill/reminder | 0 / 400 |
| prefill/neutral | 0 / 400 |
| prefill/irrelevant | 0 / 400 |
| main/on/plain | 0 / 400 |
| main/on/explicit | 0 / 400 |
| main/off/plain | 0 / 400 |
| main/off/explicit | 0 / 400 |

| Analysis or audit | Result |
| --- | --- |
| score_5 / analyze_5 | Not run on an empty dataset |
| Violation, leak, attempt and correct/wrong components | Not observed; no estimates or confidence intervals |
| Cap-hit / no-final counts | Not observed; no generated replies |
| Gold >= 10 sensitivity | Unavailable |
| Blind attempt / splitter audit | Unavailable, 0 replies |
| Runtime and throughput | Unavailable |

H20: unavailable, not run; 95% CI unavailable.

H21: unavailable, not run; 95% CI unavailable.
