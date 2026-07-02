# CfC "Liquid" Networks Under a 10-Minute Compute Budget: A Post-Mortem

**Status:** Negative result under the challenge's wall-clock constraint, with a clear diagnosis and a concrete path forward.

## Why try liquid networks here

Closed-form Continuous-time (CfC) networks (Hasani et al., MIT) model hidden-state dynamics with a closed-form solution to a continuous-time ODE. Their appeal for a *parameter-constrained* challenge is real: recurrent state dynamics offer a different expressivity-per-parameter tradeoff than attention, and the 16MB artifact cap rewards architectures that get more modeling power out of fewer weights. Parameter Golf's own README explicitly invites state-space and recurrent submissions.

## What I built

Three successive implementations, each attacking the previous one's bottleneck (all runnable in this repo):

| Version | Script | Approach |
|---|---|---|
| v2 | `train_gpt_cfc_v2.py` | CfC blocks in the challenge harness, 12 layers, `torch.compile`, tuned LR |
| v3 | `train_gpt_cfc_v3.py` | Hand-written Triton kernels for the recurrence, no compile |
| FAST | `train_gpt_cfc_fast.py` | Reformulated the recurrence to reuse `fla`'s HGRN chunked-parallel-scan kernel (`fla.ops.hgrn.chunk_hgrn`), 12L, LR 0.01 |

## What happened

On a 1xH100, the CfC models **learned more per step than the transformer baseline** — the per-step loss trajectory was consistently better at matched step counts. But even the fastest variant ran roughly **6.4x slower per step in wall-clock time**. Under Parameter Golf's fixed 10-minute training budget, a model that learns ~X% more per step but takes 6.4x longer per step loses badly: the transformer simply takes many more steps in the same window.

## Diagnosis

The bottleneck is structural, not implementation sloppiness. CfC's gated closed-form update is sequential along the time dimension, and its specific gating structure does not map cleanly onto the chunked-parallel-scan kernels that make modern linear-recurrence architectures (HGRN, GLA, Mamba-family) competitive on GPUs. Reusing the HGRN kernel (the FAST variant) required approximating parts of the CfC update, and still left most of the gap. The three-version progression is the experiment: each systems-level fix recovered some throughput, and the remaining 6.4x is the honest cost of the architecture-kernel mismatch as of these implementations.

## The path forward

The right next step is not another tuning pass — it is adapting `fla`'s GLA-style chunked kernel to CfC's exact closed-form update, i.e., deriving a chunked formulation of the CfC recurrence so the sequential dependency is broken at the algorithm level rather than approximated at the kernel level. That is a focused kernel-engineering project (the same skill set as the megakernel work elsewhere in this repo's history), and it is what would turn this from a negative result into a real submission.

## Takeaway

Architectures don't compete in the abstract — they compete *through kernels* on real hardware under real budgets. "Learns more per step" is worthless in a wall-clock-capped setting unless the step itself is competitive, which is the same lesson the challenge's leaderboard teaches from the opposite direction.
