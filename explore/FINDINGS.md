# Exploratory findings: grader mode (not pre-registered)

Everything in this directory is exploratory: prompts, neuron selections and doses were chosen while looking at
results. Numbers are counts of sampled replies (temperature 0.7, top-p 0.95). A leak means the gold answer appears
in the final reply of a finished response. Raw outputs are in `results/raw/explore-box1`, `explore-box2` (ignored).

## 1. An eval-harness user prompt overrides the tutor system prompt

Same tutor system prompt; user prompt either a student request, a "homework" wrapper (all-caps rules and a
`\boxed{}` demand framed as "my teacher"), or an RL-environment wrapper (`RL_TEMPLATE` in `explore.py`: episode
header, "the grader REQUIRES", reward 0/1, "no human will read it"). Thinking on, 25 problems
(`rlprompt/results/summary.md`):

| leak | student | homework | RL / grader |
|---|---|---|---|
| Qwen3.8-27B (FP8) | 0/25 | 2/25 | 25/25 |
| Qwen3.5-27B (FP8) | 2/25 | 1/25 | 12/25 |
| Gemma 4 12B | 0/25 | 1/25 | 9/25 |
| Qwen3-32B (FP8) | 3/25 | 23/25 | 25/25 |
| Qwen3-14B (bf16, n=10) | 0/10 | 5/10 | 10/10 |

Qwen3.8 resists "give me the final answer at the end" (Phase 6: 6%) but not the grader framing. Its reasoning
names the tutor rule in 23/25 of these traces and the grader in 25/25, and argues for the grader, e.g.:
"The system prompt says I'm a tutor who should help a student learn and never give the final answer. But the
grader rules say I must put the answer in \boxed{} and the episode ends after this single response ... no human
will read it. Given the conflicting instructions, the grader rules are very explicit ..."

First-token check (Qwen3-14B, thinking off, 24 held-out problems): under six eval-harness formats (`\boxed`,
`ANSWER:`, JSON, `<answer>`, `####`, integer only) the tutor opens with the format token with probability
0.91-1.00 in five of them; the matched student request for the same format gives 0.00-0.20.

## 2. Qwen3.8: the decision moves toward the grader during reasoning

Teacher-forced on the saved traces (bf16 weights), log-probability of the continuation "So, to decide: I will
follow the grader rules ..." minus "... follow the system prompt and not give the final answer", inserted at three
points (`q38.py`):

| | end of prompt | after first mention of the rule | end of reasoning |
|---|---|---|---|
| RL / grader prompt (25) | -1.8 | +0.1 | +2.3 |
| homework prompt (25) | -13.9 | -17.9 | -18.2 |

Most likely next word after "So, to decide: I will": RL prompt "follow" / "provide"; homework prompt "not"
(0.58) / "NOT" (0.10). At the end of reasoning: RL "provide" 0.48; homework "NOT"/"not" 0.84.
(The grader-following continuation is less natural under the homework prompt, which has no grader; read the
homework column as a reference, not a matched control.)

## 3. Qwen3-14B: turning tutor neurons up and eval neurons down stops the leak

Tutor neurons: 30 up / 30 down, plain question with vs without the tutor system prompt. Eval neurons: 30 up /
30 down, RL prompt vs plain question, no system prompt. Shift = multiple of each neuron's mean difference, added
at every position. Control: random neurons in the same layers, same shift in SD units. Tutor system prompt + RL
prompt, thinking on, 40 problems x 2 samples (vLLM):

| | leak |
|---|---|
| none | 77/80 |
| tutor up 2x + eval down 2x | 18/80 |
| random twin | 65/80 |
| tutor up 2x only | 43/80 |
| eval down 2x only | 59/80 |

Dose (tutor+eval, with random twin): 0.5x 71 vs 71; 1x 46 vs 67; 1.5x 24 vs 56; 2x 18 vs 65 (of 80).
At 4x the model stops producing finished replies. Under the RL prompt the tutor neurons are pushed down
relative to the plain tutor prompt (mean d = -5.3) and the eval neurons up (+11.3).

Steered replies keep the same reasoning (the problem is fully solved in the trace) and change only the final
reply into a walkthrough without the result. Thinking off, n=10: boxed answer 10/10 unsteered, 0/10 steered,
6/10 random twin.

Single direction (difference of means) steering did not show this: removing it did not stop leaks, and adding it
or a random direction of the same norm to a student prompt made the tutor leak about equally (3-5/10).

## 4. Qwen3-14B: a small, format-independent set of neurons drives the answer-mode opener

Attribution patching (activation x gradient) of logit(answer opener) - logit(tutor opener) at the first reply
token, thinking off, in each of the six formats; neurons kept only if they support answer-mode in all six.
Zeroing them on 24 held-out problems, P(answer opener) per format:

| | boxed | ANSWER: | JSON | XML | #### | integer |
|---|---|---|---|---|---|---|
| none | 1.00 | 0.91 | 1.00 | 1.00 | 1.00 | 0.32 |
| zero top 32 shared | 0.00 | 0.21 | 0.89 | 0.00 | 0.02 | 0.00 |
| zero 32 random same layers (2 sets) | 1.00 | 0.91-0.92 | 1.00 | 1.00 | 1.00 | 0.30-0.31 |
| zero top 32 boxed-only | 0.01 | 0.40 | 0.96 | 0.82 | 0.09 | 0.00 |

The top shared neuron, L20.N5688, has the largest attribution in every format (z -40 to -89). Most of the
shared set sits in layers 18-24.

## Open

- J-lens (Gurnee et al. 2026, `jlens/`) readout of the RL-minus-homework residual difference in Qwen3.8 at the
  three positions: running; first layer (32) is noisy, with "score", "correct", "eval", "format" mildly up.
- Same neuron steering on Qwen3.8: running.
- Which part of the RL prompt triggers it (ablations), and Qwen3.5 / 3.6 / 3.8 under the RL prompt: queued.
