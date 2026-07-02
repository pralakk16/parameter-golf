# Hybrid Cross-Entropy + Wasserstein-1 Loss: A Large Effect With the Wrong Explanation

**Status:** Smoke-scale result with controls; unreplicated at competition scale. Honest open question, not a claim.

## The idea

Replace pure cross-entropy with a hybrid loss for language model pretraining:

```
loss = alpha * cross_entropy + (1 - alpha) * wasserstein_1
```

where the Wasserstein-1 component penalizes probability mass by its distance from the correct token:

```
W = sum_i P(token_i) * distance(token_i, correct_token)
```

The intuition: cross-entropy gives a binary right/wrong signal, while Wasserstein grades predictions by *how* wrong they are. With a 1024-token vocabulary, the full 1024×1024 cost matrix (cosine distances between the model's own embedding vectors, refreshed every N steps) is tractable to compute exactly.

Implementation: `train_gpt_mlx_wasserstein.py` — everything configurable by environment variable (alpha, cost metric, squared/normalized costs, focal weighting, warmup delay, softmax temperature, random-cost ablation).

## Results (Mac M5, 500 steps, batch 8192 tokens, seed 1337)

> **Compute-regime caveat:** these runs use ~4M training tokens — 125x less data exposure than the 8xH100 leaderboard setting. Numbers are internally comparable, not leaderboard-comparable.

| Alpha | Wasserstein share | val_bpb | Gap vs CE baseline |
|---|---|---|---|
| 1.0 (pure CE) | 0% | 2.1452 | — |
| 0.9 | 10% | 1.9758 | 0.169 |
| 0.7 | 30% | 1.6288 | 0.516 |
| 0.5 | 50% | 1.2871 | 0.858 |

The trend is strikingly linear: each additional 10% of Wasserstein weight buys ~0.17 BPB at this scale. Squared+normalized costs and focal weighting each add small further gains at alpha=0.9.

## The controls — where it gets interesting

I ran four validation tests, and the third one killed my own favorite explanation:

1. **Code sanity (alpha=1.0):** my modified script matches the untouched baseline (2.3913 vs 2.3901). The machinery itself is a no-op when disabled.
2. **Learning-rate effect?** Down-weighting CE could act like a smaller LR. Halving the matrix LR on the pure-CE baseline improves only 0.05 BPB — about 6% of the alpha=0.5 effect. Not the mechanism.
3. **Semantic distances?** I replaced the embedding-derived cost matrix with a **random** one: 1.2712 vs 1.2871 with real distances. **Random costs work just as well.** Whatever this effect is, it is *not* optimal-transport semantics — the "Wasserstein" story I started with is wrong.
4. **Fancy label smoothing?** Label smoothing at 0.1 and 0.5 makes things *worse* (2.4716 / 3.4293). The two techniques move in opposite directions, so this isn't smoothing in disguise.

## Honest interpretation

Something about per-token differentiated gradient weighting produces a large improvement at this scale, and it survives the obvious alternative explanations I could test locally — but I do not know the mechanism, and the random-cost result means the elegant theory I began with is dead. Remaining candidates: an implicit change in effective step size that the halved-LR control doesn't capture, an interaction with the small vocabulary (1024 tokens), or a genuine regularization effect from non-uniform target weighting. It also has not been replicated at competition scale (20K steps, 8xH100, full batch), where smoke-scale effects routinely evaporate.

I consider the negative result of Test 3 the most valuable output of this thread: it is exactly the control that prevents an exciting-but-wrong claim from being published.

## What would settle it

1. Complete the alpha sweep (0.3, 0.0) — is there an optimum or does it keep improving into pure-W collapse?
2. Replicate at 1000+ steps and then at full competition scale.
3. Learned or optimal weight matrices: if random beats semantic, does *anything* beat random?
4. A theoretical account of why non-uniform per-token weighting helps while uniform smoothing hurts.
