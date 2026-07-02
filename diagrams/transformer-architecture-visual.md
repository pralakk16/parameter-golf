# Transformer Architecture — Complete Visual Guide
## TRAINING FLOW

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                           TRAINING DATA                                     ║
║                                                                             ║
║   "The cat sat on the mat in the warm sunny afternoon"                      ║
║                                                                             ║
║   Input:  "The cat sat on the mat in the warm sunny"                        ║
║   Target: "cat sat on the mat in the warm sunny afternoon"                  ║
║           (shifted by 1 — each position predicts the NEXT token)            ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 1: TOKENIZER                                                          ║
║  ┌─────────────────────────────────────────────────────────┐                ║
║  │ Splits text into token IDs using a fixed vocabulary     │                ║
║  │                                                         │                ║
║  │ "The" → 42    "cat" → 891   "sat" → 203                │                ║
║  │ "on"  → 156   "the" → 42    "mat" → 742   ...          │                ║
║  │                                                         │                ║
║  │ Output: [42, 891, 203, 156, 42, 742, ...]               │                ║
║  │                                                         │                ║
║  │ PURPOSE: convert text to numbers the model can process  │                ║
║  │ INTELLIGENCE: none — just a lookup dictionary           │                ║
║  │ PARAMETERS: none — fixed before training                │                ║
║  └─────────────────────────────────────────────────────────┘                ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 2: EMBEDDING TABLE                                                    ║
║  ┌─────────────────────────────────────────────────────────┐                ║
║  │ Looks up each token ID in a table of learned vectors    │                ║
║  │                                                         │                ║
║  │ Token 42  → [0.2, 0.5, -0.1, ..., 0.3]  (512 numbers)  │                ║
║  │ Token 891 → [0.8, -0.3, 0.6, ..., 0.1]  (512 numbers)  │                ║
║  │ Token 203 → [0.1, 0.7, 0.4, ..., -0.2]  (512 numbers)  │                ║
║  │                                                         │                ║
║  │ PURPOSE: convert token IDs to rich vector representations│               ║
║  │ INTELLIGENCE: learned — similar tokens get similar vectors│              ║
║  │ PARAMETERS: 1024 × 512 = 524,288 (3% of model)         │                ║
║  │ LEARNED VIA: backpropagation (starts random)            │                ║
║  └─────────────────────────────────────────────────────────┘                ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 3: POSITIONAL ENCODING (RoPE)                                         ║
║  ┌─────────────────────────────────────────────────────────┐                ║
║  │ Adds position information to each token vector          │                ║
║  │                                                         │                ║
║  │ "The" at position 1: embedding rotated by angle_1       │                ║
║  │ "the" at position 5: embedding rotated by angle_5       │                ║
║  │                                                         │                ║
║  │ Same word at different positions → different vectors     │                ║
║  │                                                         │                ║
║  │ PURPOSE: tell the model WHERE each token is             │                ║
║  │ INTELLIGENCE: none — fixed mathematical formula         │                ║
║  │ PARAMETERS: none — computed, not learned                │                ║
║  └─────────────────────────────────────────────────────────┘                ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  STEP 4: TRANSFORMER BLOCKS (× 9)                                          ║
║  ════════════════════════════════                                           ║
║                                                                             ║
║  Each block: Attention → Add & Norm → MLP → Add & Norm                     ║
║  Repeat 9 times with DIFFERENT weights per block                            ║
║                                                                             ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                                                                     │    ║
║  │   BLOCK 1 (basic word-level understanding)                          │    ║
║  │                                                                     │    ║
║  │   ┌───────────────────────────────────────────────────────────┐     │    ║
║  │   │  4A: MULTI-HEAD ATTENTION                                  │     │    ║
║  │   │  ═══════════════════════                                   │     │    ║
║  │   │                                                            │     │    ║
║  │   │  PURPOSE: let each token look at other tokens              │     │    ║
║  │   │           to understand CONTEXT                            │     │    ║
║  │   │                                                            │     │    ║
║  │   │  ★ INTELLIGENCE: RELATIONSHIPS                             │     │    ║
║  │   │    This is where tokens INTERACT                           │     │    ║
║  │   │    "sat" discovers "cat" is the subject                    │     │    ║
║  │   │    "it" finds what it refers to                            │     │    ║
║  │   │                                                            │     │    ║
║  │   │  HOW IT WORKS:                                             │     │    ║
║  │   │  ┌──────────────────────────────────────────────────┐      │     │    ║
║  │   │  │                                                  │      │     │    ║
║  │   │  │  For each token:                                 │      │     │    ║
║  │   │  │    Q = token × W_query  "what am I looking for?" │      │     │    ║
║  │   │  │    K = token × W_key    "what do I contain?"     │      │     │    ║
║  │   │  │    V = token × W_value  "what info do I share?"  │      │     │    ║
║  │   │  │                                                  │      │     │    ║
║  │   │  │  Score = Q · K (dot product)                     │      │     │    ║
║  │   │  │    → one number per token pair                   │      │     │    ║
║  │   │  │    → "how relevant is this token to me?"         │      │     │    ║
║  │   │  │                                                  │      │     │    ║
║  │   │  │  Mask future tokens (can't peek ahead)           │      │     │    ║
║  │   │  │                                                  │      │     │    ║
║  │   │  │  Softmax → attention weights (sum to 1)          │      │     │    ║
║  │   │  │                                                  │      │     │    ║
║  │   │  │  Output = weighted blend of V vectors            │      │     │    ║
║  │   │  │    → each token now contains info from           │      │     │    ║
║  │   │  │      the tokens it attended to                   │      │     │    ║
║  │   │  │                                                  │      │     │    ║
║  │   │  │  × 8 heads in parallel (each finds different     │      │     │    ║
║  │   │  │    type of relationship)                         │      │     │    ║
║  │   │  │                                                  │      │     │    ║
║  │   │  │  Concatenate all heads → project back to 512     │      │     │    ║
║  │   │  │                                                  │      │     │    ║
║  │   │  │  LIMITATION: all relationships forced through    │      │     │    ║
║  │   │  │  dot product (similarity). Reference, causation, │      │     │    ║
║  │   │  │  negation → all encoded as "similarity" via      │      │     │    ║
║  │   │  │  learned Q/K projections. Works but wasteful.    │      │     │    ║
║  │   │  │                                                  │      │     │    ║
║  │   │  │  PARAMETERS: Q(512×512) + K(512×256) +           │      │     │    ║
║  │   │  │    V(512×256) + O(512×512) = 786,432 per block   │      │     │    ║
║  │   │  └──────────────────────────────────────────────────┘      │     │    ║
║  │   └───────────────────────────────────────────────────────────┘     │    ║
║  │                              ↓                                      │    ║
║  │   ┌───────────────────────────────────────────────────────────┐     │    ║
║  │   │  4B: ADD & NORM                                            │     │    ║
║  │   │                                                            │     │    ║
║  │   │  output = RMSNorm(input + attention_output)                │     │    ║
║  │   │                                                            │     │    ║
║  │   │  ADD: preserves original signal (residual connection)      │     │    ║
║  │   │       if attention learned something bad, input survives   │     │    ║
║  │   │  NORM: keeps numbers in reasonable range                   │     │    ║
║  │   │                                                            │     │    ║
║  │   │  INTELLIGENCE: none — just signal preservation             │     │    ║
║  │   │  PARAMETERS: negligible                                    │     │    ║
║  │   └───────────────────────────────────────────────────────────┘     │    ║
║  │                              ↓                                      │    ║
║  │   ┌───────────────────────────────────────────────────────────┐     │    ║
║  │   │  4C: MLP (FEED-FORWARD NETWORK)                            │     │    ║
║  │   │  ═════════════════════════════                              │     │    ║
║  │   │                                                            │     │    ║
║  │   │  PURPOSE: process and transform each token's               │     │    ║
║  │   │           representation using stored knowledge            │     │    ║
║  │   │                                                            │     │    ║
║  │   │  ★★★ INTELLIGENCE: KNOWLEDGE STORAGE & PROCESSING          │     │    ║
║  │   │  This is where the model's KNOWLEDGE lives.                │     │    ║
║  │   │  This is the BRAIN of the transformer.                     │     │    ║
║  │   │  This is the ONLY nonlinear transformation.                │     │    ║
║  │   │  Removing this DEVASTATES the model.                       │     │    ║
║  │   │                                                            │     │    ║
║  │   │  HOW IT WORKS:                                             │     │    ║
║  │   │  ┌──────────────────────────────────────────────────┐      │     │    ║
║  │   │  │                                                  │      │     │    ║
║  │   │  │  EXPAND:  token (512) × W_expand (512×1024)      │      │     │    ║
║  │   │  │           → 1024 pattern-match scores            │      │     │    ║
║  │   │  │           Each row of W_expand is a "pattern"    │      │     │    ║
║  │   │  │           Dot product = "does input match this   │      │     │    ║
║  │   │  │           pattern?"                              │      │     │    ║
║  │   │  │                                                  │      │     │    ║
║  │   │  │  ACTIVATE: ReLU² (square of max(0, x))           │      │     │    ║
║  │   │  │           ~950 neurons → 0 (pattern didn't match)│      │     │    ║
║  │   │  │           ~50 neurons survive (pattern matched!) │      │     │    ║
║  │   │  │           THIS is the only nonlinearity          │      │     │    ║
║  │   │  │           THIS enables complex pattern learning  │      │     │    ║
║  │   │  │                                                  │      │     │    ║
║  │   │  │  SHRINK:  activated (1024) × W_shrink (1024×512) │      │     │    ║
║  │   │  │           → 512-dim output                       │      │     │    ║
║  │   │  │           Each column of W_shrink is "what to    │      │     │    ║
║  │   │  │           output when this pattern matches"      │      │     │    ║
║  │   │  │                                                  │      │     │    ║
║  │   │  │  IT'S A KEY-VALUE MEMORY:                        │      │     │    ║
║  │   │  │    W_expand rows = KEYS (what patterns to find)  │      │     │    ║
║  │   │  │    ReLU²          = GATE (which ones matched)    │      │     │    ║
║  │   │  │    W_shrink cols  = VALUES (what to output)      │      │     │    ║
║  │   │  │                                                  │      │     │    ║
║  │   │  │  SAME ARCHITECTURE SINCE 1986. 40 years.         │      │     │    ║
║  │   │  │  95% of computation wasted (neurons output 0).   │      │     │    ║
║  │   │  │  Static — same weights for every input.          │      │     │    ║
║  │   │  │                                                  │      │     │    ║
║  │   │  │  PARAMETERS: 512×1024 + 1024×512 = 1,048,576     │      │     │    ║
║  │   │  │  (57% of each block's parameters)                │      │     │    ║
║  │   │  └──────────────────────────────────────────────────┘      │     │    ║
║  │   └───────────────────────────────────────────────────────────┘     │    ║
║  │                              ↓                                      │    ║
║  │   ┌───────────────────────────────────────────────────────────┐     │    ║
║  │   │  4D: ADD & NORM (again)                                    │     │    ║
║  │   │                                                            │     │    ║
║  │   │  output = RMSNorm(pre_mlp_input + mlp_output)              │     │    ║
║  │   │  Same as before — preserve signal, normalize.              │     │    ║
║  │   └───────────────────────────────────────────────────────────┘     │    ║
║  │                              ↓                                      │    ║
║  │   Output goes to BLOCK 2                                            │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                             ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  BLOCK 2 (phrase-level understanding)                               │    ║
║  │  Same structure: Attention → Add&Norm → MLP → Add&Norm              │    ║
║  │  DIFFERENT weights (learned separately)                             │    ║
║  │  Builds on Block 1's output — deeper relationships                  │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                              ↓                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  BLOCK 3 ... BLOCK 8 (progressively deeper understanding)           │    ║
║  │  Each block: new attention relationships + new MLP processing       │    ║
║  │  Each has its OWN weights — different from every other block        │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                              ↓                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  BLOCK 9 (prediction-level understanding)                           │    ║
║  │  Final attention gathers prediction-relevant context                │    ║
║  │  Final MLP produces representation ready for prediction             │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                             ║
║  NOTE: U-Net skip connections also pass information from                    ║
║  early blocks (1-4) directly to late blocks (5-9)                          ║
║                                                                             ║
║  TOTAL BLOCK PARAMETERS:                                                    ║
║    9 × (786K attention + 1,048K MLP + small norms/scales)                  ║
║    = ~16.5M parameters (97% of model)                                      ║
║                                                                             ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 5: OUTPUT PROJECTION                                                  ║
║  ┌─────────────────────────────────────────────────────────┐                ║
║  │  Take final token representation (512 dims)             │                ║
║  │  Multiply by embedding table TRANSPOSED                 │                ║
║  │  → 1024 scores (one per token in vocabulary)            │                ║
║  │                                                         │                ║
║  │  Softcap: 30 × tanh(score/30) — prevents extreme values│                ║
║  │  Softmax: convert scores to probabilities (sum to 1)    │                ║
║  │                                                         │                ║
║  │  Result: P("mat")=0.15, P("rug")=0.08, P("the")=0.05  │                ║
║  │                                                         │                ║
║  │  PURPOSE: convert internal representation to prediction │                ║
║  │  INTELLIGENCE: none — just linear projection            │                ║
║  │  PARAMETERS: reuses embedding table (tied embeddings)   │                ║
║  └─────────────────────────────────────────────────────────┘                ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 6: LOSS FUNCTION                                                      ║
║  ┌─────────────────────────────────────────────────────────┐                ║
║  │  Compare prediction to correct answer                   │                ║
║  │                                                         │                ║
║  │  Predicted: P("mat") = 0.15                             │                ║
║  │  Correct:   "mat"                                       │                ║
║  │  Loss = -log(0.15) = 1.90                               │                ║
║  │                                                         │                ║
║  │  ★ THIS DETERMINES WHAT THE MODEL LEARNS                │                ║
║  │  Currently: binary right/wrong (cross-entropy)          │                ║
║  │  Only uses P(correct token). Ignores everything else.   │                ║
║  │  "rug" gets zero credit even though it shows            │                ║
║  │  understanding. "quantum" gets same zero credit.        │                ║
║  │                                                         │                ║
║  │  PARAMETERS: none                                       │                ║
║  │  BUT: determines the QUALITY of all gradient signals    │                ║
║  │  that flow back to update every parameter in the model  │                ║
║  └─────────────────────────────────────────────────────────┘                ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 7: BACKPROPAGATION (right to left)                                    ║
║  ┌─────────────────────────────────────────────────────────┐                ║
║  │  Loss → gradients flow BACKWARD through entire model    │                ║
║  │                                                         │                ║
║  │  Output projection ← gradients                          │                ║
║  │  Block 9 MLP       ← gradients                          │                ║
║  │  Block 9 Attention  ← gradients                          │                ║
║  │  ...                                                    │                ║
║  │  Block 1 MLP       ← gradients (weakest signal here)    │                ║
║  │  Block 1 Attention  ← gradients (weakest signal here)    │                ║
║  │  Embedding table   ← gradients                          │                ║
║  │                                                         │                ║
║  │  Each weight gets a gradient:                           │                ║
║  │    "increase this weight by 0.003"                      │                ║
║  │    "decrease this weight by 0.001"                      │                ║
║  │                                                         │                ║
║  │  Two formulas (from our basic NN discussion):           │                ║
║  │    Neuron error = next layer error × connecting weight   │                ║
║  │    Weight gradient = neuron error × input through weight│                ║
║  │                                                         │                ║
║  │  PURPOSE: compute how to adjust every weight            │                ║
║  │  INTELLIGENCE: none — just math (chain rule)            │                ║
║  │  PARAMETERS: none created, all existing ones get grads  │                ║
║  └─────────────────────────────────────────────────────────┘                ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 8: WEIGHT UPDATE (optimizer)                                          ║
║  ┌─────────────────────────────────────────────────────────┐                ║
║  │  All ~17M weights updated simultaneously                │                ║
║  │                                                         │                ║
║  │  Matrix weights (attention Q/K/V/O, MLP):               │                ║
║  │    → Muon optimizer (orthogonalized gradient update)    │                ║
║  │    → w = w - lr × orthogonalize(gradient)               │                ║
║  │                                                         │                ║
║  │  Scalar weights (scales, biases, gains):                │                ║
║  │    → Adam optimizer (adaptive per-weight learning rate) │                ║
║  │    → w = w - lr × adam_adjusted(gradient)               │                ║
║  │                                                         │                ║
║  │  Embedding weights:                                     │                ║
║  │    → Adam with separate learning rate                   │                ║
║  │                                                         │                ║
║  │  PURPOSE: make each weight slightly smarter             │                ║
║  │  INTELLIGENCE: none in the optimizer itself —           │                ║
║  │    but this is where intelligence ACCUMULATES           │                ║
║  │    in the weights over millions of steps                │                ║
║  └─────────────────────────────────────────────────────────┘                ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 9: REPEAT                                                             ║
║                                                                             ║
║  Go back to Step 1 with next batch of training data.                        ║
║  Repeat 7,000-20,000 times.                                                ║
║  Each step, every weight gets slightly smarter.                             ║
║  After all steps → intelligence has accumulated in the weights.             ║
╚════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════════════════════


## INFERENCE FLOW (after training — model is frozen)

╔══════════════════════════════════════════════════════════════════════════════╗
║                           USER INPUT                                        ║
║                                                                             ║
║   "The cat sat on the"                                                      ║
║   → predict what comes next                                                 ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 1: TOKENIZE                                                           ║
║  "The cat sat on the" → [42, 891, 203, 156, 42]                            ║
║  Same as training. No change.                                               ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 2: EMBED + POSITION                                                   ║
║  Token IDs → vectors → add position info                                    ║
║  Same as training. No change.                                               ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 3: FORWARD PASS THROUGH 9 BLOCKS                                     ║
║                                                                             ║
║  Same attention + MLP as training.                                          ║
║  Same weights (frozen, not updating).                                       ║
║  NO backward pass. NO gradient. NO weight update.                           ║
║  Just data flowing forward through the blocks.                              ║
║                                                                             ║
║  Block 1: attention gathers context → MLP processes                         ║
║  Block 2-8: deeper and deeper understanding                                ║
║  Block 9: final representation ready for prediction                        ║
║                                                                             ║
║  KV CACHE: stores K and V vectors from each block                          ║
║  so you don't recompute them for previous tokens                            ║
║  when generating the next token.                                            ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 4: OUTPUT PROJECTION                                                  ║
║                                                                             ║
║  Final vector × embedding table → 1024 scores                              ║
║  Softcap → softmax → probabilities                                         ║
║                                                                             ║
║  P("mat")=0.45, P("rug")=0.25, P("floor")=0.15, ...                       ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 5: SAMPLING (where "creativity" comes from)                           ║
║  ┌─────────────────────────────────────────────────────────┐                ║
║  │                                                         │                ║
║  │  Greedy (temperature=0):                                │                ║
║  │    Always pick highest → "mat" → deterministic          │                ║
║  │                                                         │                ║
║  │  Sampling (temperature=1):                              │                ║
║  │    Randomly pick based on probabilities                 │                ║
║  │    Usually "mat" but sometimes "rug" → varied output    │                ║
║  │                                                         │                ║
║  │  High temperature (temperature=2):                      │                ║
║  │    Flatten probabilities → more random → "creative"     │                ║
║  │                                                         │                ║
║  │  Selected token: "mat"                                  │                ║
║  └─────────────────────────────────────────────────────────┘                ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 6: APPEND AND REPEAT                                                  ║
║                                                                             ║
║  Add "mat" to the sequence:                                                 ║
║  "The cat sat on the mat"                                                   ║
║                                                                             ║
║  Go back to Step 1 to predict the NEXT token.                               ║
║  KV cache means you only process the NEW token                              ║
║  through the blocks (previous tokens already cached).                       ║
║                                                                             ║
║  Repeat until: end token, max length, or user stops.                        ║
╚══════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════


## INTELLIGENCE MAP — WHERE DOES EACH TYPE LIVE?

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  KNOWLEDGE (facts, patterns, language rules):                               ║
║  ┌──────────────────────────────────────┐                                   ║
║  │  ★★★ MLP WEIGHTS (W_expand, W_shrink)│  ← 57% of all parameters        ║
║  │  Each row of W_expand = a pattern     │  ← "is this a noun-verb pair?"  ║
║  │  Each col of W_shrink = a response    │  ← "if yes, output this"        ║
║  │  9 blocks × 1M params = 9.4M params   │                                 ║
║  │  THIS IS THE BRAIN                    │                                  ║
║  └──────────────────────────────────────┘                                   ║
║                                                                             ║
║  RELATIONSHIPS (which tokens relate to which):                              ║
║  ┌──────────────────────────────────────┐                                   ║
║  │  ★★ ATTENTION WEIGHTS (W_q, W_k, W_v)│  ← 43% of all parameters        ║
║  │  W_q, W_k learn what to look for     │  ← "find the subject"           ║
║  │  W_v learns what to share             │  ← "share subject info"         ║
║  │  8 heads = 8 relationship types       │                                  ║
║  │  THESE ARE THE EYES                   │                                  ║
║  └──────────────────────────────────────┘                                   ║
║                                                                             ║
║  WORD MEANING (what each token represents):                                 ║
║  ┌──────────────────────────────────────┐                                   ║
║  │  ★ EMBEDDING TABLE                    │  ← 3% of all parameters         ║
║  │  1024 tokens × 512 dims               │                                  ║
║  │  Similar words → similar vectors      │                                  ║
║  │  THIS IS THE VOCABULARY               │                                  ║
║  └──────────────────────────────────────┘                                   ║
║                                                                             ║
║  CREATIVITY (novel, unexpected outputs):                                    ║
║  ┌──────────────────────────────────────┐                                   ║
║  │  ☆ SAMPLING (temperature + randomness)│  ← 0 parameters                 ║
║  │  Not in the weights at all             │                                  ║
║  │  Just a random number generator        │                                  ║
║  │  picking from the distribution         │                                  ║
║  │  THIS IS THE DICE ROLL                │                                  ║
║  └──────────────────────────────────────┘                                   ║
║                                                                             ║
║  LEARNING SIGNAL (what shapes all the above):                               ║
║  ┌──────────────────────────────────────┐                                   ║
║  │  ☆ LOSS FUNCTION                      │  ← 0 parameters                 ║
║  │  Determines the QUALITY of gradients  │                                  ║
║  │  that flow back to update every weight│                                  ║
║  │  Currently: cross-entropy (binary)    │                                  ║
║  │  THIS IS THE TEACHER                  │                                  ║
║  └──────────────────────────────────────┘                                   ║
║                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════


## PARAMETER BUDGET (where 16 MB goes)

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  Total: ~17M parameters → compressed to fit in 16 MB                        ║
║                                                                             ║
║  ┌────────────────────────────────────────────────────────────────────┐      ║
║  │████████████████████████████████████████████████████████│ MLP 57%   │      ║
║  │█████████████████████████████████████████│ Attention 43% │          │      ║
║  │███│ Embeddings 3%                                       │          │      ║
║  └────────────────────────────────────────────────────────────────────┘      ║
║                                                                             ║
║  Per block breakdown:                                                       ║
║  ┌────────────────────────────────────────────────────────────────────┐      ║
║  │  Attention:                                                        │      ║
║  │    W_query:  512 × 512  = 262,144 params                          │      ║
║  │    W_key:    512 × 256  = 131,072 params  (shared via GQA)        │      ║
║  │    W_value:  512 × 256  = 131,072 params  (shared via GQA)        │      ║
║  │    W_output: 512 × 512  = 262,144 params                          │      ║
║  │    Subtotal:              786,432 params                           │      ║
║  │                                                                    │      ║
║  │  MLP:                                                              │      ║
║  │    W_expand: 512 × 1024 = 524,288 params                          │      ║
║  │    W_shrink: 1024 × 512 = 524,288 params                          │      ║
║  │    Subtotal:              1,048,576 params                         │      ║
║  │                                                                    │      ║
║  │  Block total: ~1,835,008 params                                    │      ║
║  │  × 9 blocks = ~16,515,072 params                                  │      ║
║  └────────────────────────────────────────────────────────────────────┘      ║
║                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════


## THE FUNDAMENTAL LIMITATIONS

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  1. MLP IS STATIC (same processing for every input — since 1986)           ║
║     "the" and "antidisestablishmentarianism" → same matrix multiplication   ║
║     95% of neurons output zero → 95% of computation wasted                  ║
║                                                                             ║
║  2. ATTENTION IS BRUTE-FORCE (every token checks every token)              ║
║     All relationships forced through dot product (similarity)               ║
║     8 heads = max 8 relationship types                                      ║
║     Most comparisons produce near-zero scores (irrelevant pairs)            ║
║                                                                             ║
║  3. FIXED DEPTH (every token gets same 9 blocks)                           ║
║     Simple tokens waste computation in later blocks                         ║
║     Complex tokens might need more blocks but can't get them                ║
║                                                                             ║
║  4. LOSS FUNCTION IS BINARY (right or wrong, nothing in between)           ║
║     Only uses P(correct token) — ignores full distribution                  ║
║     "rug" gets zero credit when answer is "mat"                             ║
║     Weak gradient signal — 1 bit of info per position                       ║
║                                                                             ║
║  5. NO MEMORY (processes and forgets within MLP)                            ║
║     Each token processed independently in MLP                               ║
║     No accumulation of knowledge across sequence within MLP                 ║
║     Only attention provides cross-token information                         ║
║                                                                             ║
║  6. LEFT-TO-RIGHT ONLY (can't revise earlier decisions)                    ║
║     Information flows one direction                                         ║
║     Early mistakes propagate — can't go back and fix                        ║
║                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
```
