# Hourglass Transformer + Per-Head Attention Gate — Complete Visual Guide
## OUR CURRENT BEST ARCHITECTURE (12L Hourglass + STP + No Skips + Per-Head Attention Gate)
## val_bpb 1.7396 (Mac M5, batch 8192, 600s wallclock, single seed)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                       WHAT'S DIFFERENT FROM BASELINE                        ║
║                                                                             ║
║   1. 12 layers instead of 9 (we redistribute saved params into depth)      ║
║   2. MLP width VARIES across layers (wide → narrow → wide hourglass)       ║
║   3. NO U-Net skip connections (they sabotage the bottleneck)              ║
║   4. STP smoothness loss (penalize jagged hidden state trajectories)       ║
║   5. Per-head content-dependent ATTENTION GATE  ★ NEWEST ★                  ║
║      (each head's output gets a per-token sigmoid gate)                    ║
║                                                                             ║
║   Result: val_bpb 1.7396 (improvement of 0.226 BPB over 9L baseline)       ║
║   Gate analysis revealed L11 attention is functionally unused (gate=0)     ║
║                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════════════════════


## TRAINING FLOW

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                           TRAINING DATA                                     ║
║                                                                             ║
║   "The cat sat on the mat in the warm sunny afternoon"                      ║
║                                                                             ║
║   Same input/target shifting as baseline.                                   ║
║   Same tokenizer (SP1024), same embeddings, same RoPE.                      ║
║                                                                             ║
║   ★ The novelty is INSIDE the transformer blocks, not around them.          ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEPS 1-3: TOKENIZE → EMBED → POSITIONS                                    ║
║                                                                             ║
║  IDENTICAL to baseline.                                                     ║
║  Tokens → 512-dim vectors → RoPE rotation                                   ║
║                                                                             ║
║  PARAMETERS: 1024 × 512 = 524,288 (embedding table, tied to output)        ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  STEP 4: TRANSFORMER BLOCKS (× 12)  ★ NEW ARCHITECTURE ★                   ║
║  ════════════════════════════════════════                                   ║
║                                                                             ║
║  Each block: Attention → Add & Norm → MLP → Add & Norm                     ║
║                                                                             ║
║  ★★★ KEY DIFFERENCE ★★★                                                     ║
║  Baseline: every block has the SAME MLP (1024 hidden)                       ║
║  Hourglass: MLP width VARIES across blocks                                  ║
║                                                                             ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                                                                     │    ║
║  │  THE HOURGLASS PATTERN (visual)                                    │    ║
║  │                                                                     │    ║
║  │     Block 1   ████████████████  MLP=1024  (2× expansion)           │    ║
║  │     Block 2   ████████████████  MLP=1024  (2× expansion)           │    ║
║  │     Block 3   ████████████████  MLP=1024  (2× expansion)           │    ║
║  │     Block 4   ████████████████  MLP=1024  (2× expansion)           │    ║
║  │                                                                     │    ║
║  │     Block 5   ████████          MLP=512   (1× — NARROW!)           │    ║
║  │     Block 6   ████████          MLP=512   (1× — NARROW!)           │    ║
║  │     Block 7   ████████          MLP=512   (1× — NARROW!)           │    ║
║  │     Block 8   ████████          MLP=512   (1× — NARROW!)           │    ║
║  │                                                                     │    ║
║  │     Block 9   ████████████████  MLP=1024  (2× expansion)           │    ║
║  │     Block 10  ████████████████  MLP=1024  (2× expansion)           │    ║
║  │     Block 11  ████████████████  MLP=1024  (2× expansion)           │    ║
║  │     Block 12  ████████████████  MLP=1024  (2× expansion)           │    ║
║  │                                                                     │    ║
║  │     Stage 1 (1-4):   RECOGNIZE → wide capacity for surface forms   │    ║
║  │     Stage 2 (5-8):   ABSTRACT  → forced compression bottleneck     │    ║
║  │     Stage 3 (9-12):  PREDICT   → wide capacity for output mapping  │    ║
║  │                                                                     │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                             ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  STAGE 1: RECOGNITION (Blocks 1-4) — WIDE                                  ║
║  ════════════════════════════════════                                       ║
║                                                                             ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  BLOCK 1 (surface form recognition)                                 │    ║
║  │                                                                     │    ║
║  │   ┌───────────────────────────────────────────────────────────┐     │    ║
║  │   │  ATTENTION + PER-HEAD CONFIDENCE GATE  ★ NEW ★              │     │    ║
║  │   │  4 KV heads, 8 query heads, 512-dim                         │     │    ║
║  │   │  Standard Q/K/V/proj   ───────────────  786,432 params      │     │    ║
║  │   │  + c_g: dim → num_heads (gate proj) ──   4,096 params       │     │    ║
║  │   │                                                            │     │    ║
║  │   │  After SDPA computes y = Attn(Q,K,V):                       │     │    ║
║  │   │    gate = sigmoid(c_g(x))   (per-token, per-head, [0,1])    │     │    ║
║  │   │    y    = y × gate          (suppress unreliable heads)     │     │    ║
║  │   │    out  = proj(y)           (standard output projection)    │     │    ║
║  │   │                                                            │     │    ║
║  │   │  ★ The gate lets the model say "for THIS token in THIS      │     │    ║
║  │   │    context, head H is unreliable, suppress it" or "head     │     │    ║
║  │   │    H is critical, amplify it". 49,152 extra params total.   │     │    ║
║  │   └───────────────────────────────────────────────────────────┘     │    ║
║  │                              ↓                                      │    ║
║  │   ┌───────────────────────────────────────────────────────────┐     │    ║
║  │   │  RMSNorm → MLP → RMSNorm                                   │     │    ║
║  │   └───────────────────────────────────────────────────────────┘     │    ║
║  │                              ↓                                      │    ║
║  │   ┌───────────────────────────────────────────────────────────┐     │    ║
║  │   │  ★ WIDE MLP (mult=2)                                       │     │    ║
║  │   │                                                            │     │    ║
║  │   │  EXPAND:   token (512) × W_expand (512 × 1024)             │     │    ║
║  │   │            → 1024 pattern-match scores                     │     │    ║
║  │   │            "Is this 'the'? Is this a noun? Is this a       │     │    ║
║  │   │             verb? Is this punctuation? ..."                │     │    ║
║  │   │                                                            │     │    ║
║  │   │  ACTIVATE: ReLU² → ~50 patterns survive                    │     │    ║
║  │   │                                                            │     │    ║
║  │   │  SHRINK:   activated (1024) × W_shrink (1024 × 512)        │     │    ║
║  │   │                                                            │     │    ║
║  │   │  PURPOSE: capture LOW-LEVEL surface features               │     │    ║
║  │   │  Why wide? Many surface patterns to detect.                │     │    ║
║  │   │                                                            │     │    ║
║  │   │  PARAMETERS: 1,048,576                                     │     │    ║
║  │   └───────────────────────────────────────────────────────────┘     │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                              ↓                                              ║
║  Blocks 2, 3, 4: SAME (wide MLP, surface-level processing)                  ║
║  Each builds on previous: word → phrase → clause level patterns             ║
║                                                                             ║
║  Stage 1 total: 4 × (786K attn + 1,048K MLP) = 7.34M params                ║
║                                                                             ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  STAGE 2: ABSTRACTION (Blocks 5-8) — NARROW BOTTLENECK                     ║
║  ═══════════════════════════════════════════════════                        ║
║                                                                             ║
║  ★★★ THIS IS WHERE THE MAGIC HAPPENS ★★★                                    ║
║                                                                             ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                                                                     │    ║
║  │   THE INFORMATION BOTTLENECK                                       │    ║
║  │                                                                     │    ║
║  │   The model carries a 512-dim vector per token through Stage 1.    │    ║
║  │   In Stage 1, MLP expands it to 1024 → many fine-grained features. │    ║
║  │                                                                     │    ║
║  │   In Stage 2, MLP only expands to 512 → HALF the room.             │    ║
║  │                                                                     │    ║
║  │   The model PHYSICALLY CANNOT carry all surface-level details      │    ║
║  │   through 512 hidden dimensions. It is FORCED to:                   │    ║
║  │                                                                     │    ║
║  │     1. Drop irrelevant fine-grained features                       │    ║
║  │     2. Compress related features into shared abstractions         │    ║
║  │     3. Preserve only what matters for downstream prediction       │    ║
║  │                                                                     │    ║
║  │   This is the INFORMATION BOTTLENECK PRINCIPLE (Tishby 2000)       │    ║
║  │   made architectural instead of optimization-discovered.           │    ║
║  │                                                                     │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                             ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  BLOCK 5 (forced abstraction)                                       │    ║
║  │                                                                     │    ║
║  │   ┌───────────────────────────────────────────────────────────┐     │    ║
║  │   │  ATTENTION (same — full 512-dim)                            │     │    ║
║  │   │  PARAMETERS: 786,432                                        │     │    ║
║  │   └───────────────────────────────────────────────────────────┘     │    ║
║  │                              ↓                                      │    ║
║  │   ┌───────────────────────────────────────────────────────────┐     │    ║
║  │   │  ★★★ NARROW MLP (mult=1)                                   │     │    ║
║  │   │                                                            │     │    ║
║  │   │  EXPAND:   token (512) × W_expand (512 × 512)              │     │    ║
║  │   │            → only 512 pattern-match scores                 │     │    ║
║  │   │            HALF as many "questions" can be asked!          │     │    ║
║  │   │                                                            │     │    ║
║  │   │  ACTIVATE: ReLU² → ~25 patterns survive                    │     │    ║
║  │   │                                                            │     │    ║
║  │   │  SHRINK:   activated (512) × W_shrink (512 × 512)          │     │    ║
║  │   │                                                            │     │    ║
║  │   │  PURPOSE: ABSTRACT, COMPRESS, GENERALIZE                   │     │    ║
║  │   │  Why narrow? Force the model to choose what matters.       │     │    ║
║  │   │                                                            │     │    ║
║  │   │  PARAMETERS: 524,288 (HALF of wide block!)                 │     │    ║
║  │   └───────────────────────────────────────────────────────────┘     │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                              ↓                                              ║
║  Blocks 6, 7, 8: SAME (narrow MLP, deep abstraction)                        ║
║                                                                             ║
║  Stage 2 total: 4 × (786K attn + 524K MLP) = 5.24M params                  ║
║                                                                             ║
║  ★ Stage 2 saves ~2.1M params vs uniform 12L. We invest these elsewhere.   ║
║                                                                             ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  STAGE 3: PREDICTION (Blocks 9-12) — WIDE                                  ║
║  ══════════════════════════════════════                                     ║
║                                                                             ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                                                                     │    ║
║  │   The narrow stage produced ABSTRACT representations.              │    ║
║  │   Stage 3 has to UNPACK those abstractions into a concrete         │    ║
║  │   probability distribution over 1024 vocabulary tokens.            │    ║
║  │                                                                     │    ║
║  │   It needs WIDE MLPs again — many "if abstract X, output token Y"  │    ║
║  │   rules to learn.                                                  │    ║
║  │                                                                     │    ║
║  │   Symmetry intuition: encoding (compression) and decoding          │    ║
║  │   (expansion) both need capacity. The middle is where you can      │    ║
║  │   afford to be small.                                              │    ║
║  │                                                                     │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                             ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  BLOCK 9 (begin unpacking abstractions)                             │    ║
║  │                                                                     │    ║
║  │   Attention (786K params)                                           │    ║
║  │   ↓                                                                 │    ║
║  │   ★ WIDE MLP (mult=2) — 1,048,576 params                           │    ║
║  │     EXPAND 512 → 1024 → ReLU² → SHRINK 1024 → 512                  │    ║
║  │     "Which concrete predictions does this abstraction support?"    │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                              ↓                                              ║
║  Blocks 10, 11, 12: SAME (wide, prediction refinement)                      ║
║                                                                             ║
║  Stage 3 total: 4 × (786K attn + 1,048K MLP) = 7.34M params                ║
║                                                                             ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  ★★★ CRITICAL: NO U-NET SKIP CONNECTIONS ★★★                                ║
║                                                                             ║
║  The baseline has skip connections that route information from              ║
║  early blocks (1-4) DIRECTLY to late blocks (9-12), bypassing               ║
║  the middle entirely.                                                       ║
║                                                                             ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                                                                     │    ║
║  │  BASELINE (with skips — what we REMOVED):                          │    ║
║  │                                                                     │    ║
║  │     Block 1 ──────────skip──────────→  Block 12                    │    ║
║  │     Block 2 ──────────skip──────────→  Block 11                    │    ║
║  │     Block 3 ──────────skip──────────→  Block 10                    │    ║
║  │     Block 4 ──────────skip──────────→  Block 9                     │    ║
║  │                                                                     │    ║
║  │     Block 5 (narrow)                                                │    ║
║  │     Block 6 (narrow)                                                │    ║
║  │     Block 7 (narrow)                                                │    ║
║  │     Block 8 (narrow)                                                │    ║
║  │                                                                     │    ║
║  │  PROBLEM: Why would the model bother learning to compress          │    ║
║  │  through the bottleneck if it can just copy raw features            │    ║
║  │  around the side?                                                  │    ║
║  │                                                                     │    ║
║  │  The skip path is a SHORTCUT. The bottleneck becomes decorative.   │    ║
║  │                                                                     │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                             ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                                                                     │    ║
║  │  OUR ARCHITECTURE (skips REMOVED):                                 │    ║
║  │                                                                     │    ║
║  │     Block 1 ↓                                                      │    ║
║  │     Block 2 ↓                                                      │    ║
║  │     Block 3 ↓                                                      │    ║
║  │     Block 4 ↓                                                      │    ║
║  │              ╲                                                      │    ║
║  │               ╲                                                     │    ║
║  │     Block 5  → → →  ALL information must pass                      │    ║
║  │     Block 6  → → →  through the narrow                             │    ║
║  │     Block 7  → → →  bottleneck. No detours.                        │    ║
║  │     Block 8  → → →                                                  │    ║
║  │               ╱                                                     │    ║
║  │              ╱                                                      │    ║
║  │     Block 9  ↓                                                      │    ║
║  │     Block 10 ↓                                                      │    ║
║  │     Block 11 ↓                                                      │    ║
║  │     Block 12 ↓                                                      │    ║
║  │                                                                     │    ║
║  │  NOW the model HAS to compress. The bottleneck is REAL.            │    ║
║  │                                                                     │    ║
║  │  Removing skips alone gave us +0.088 BPB — the single biggest      │    ║
║  │  improvement of any change we made.                                │    ║
║  │                                                                     │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                             ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 5: OUTPUT PROJECTION                                                  ║
║                                                                             ║
║  Same as baseline:                                                          ║
║    final vector × embedding_table^T → 1024 logits → softcap → softmax      ║
║                                                                             ║
║  PARAMETERS: tied to embedding table (no extra cost)                        ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 6: LOSS FUNCTION  ★ STP MODIFIED ★                                    ║
║  ┌─────────────────────────────────────────────────────────┐                ║
║  │                                                         │                ║
║  │  L = CE  +  beta × STP                                 │                ║
║  │                                                         │                ║
║  │  ┌──────────────────────────────────────────────────┐  │                ║
║  │  │  CE = standard cross-entropy on next token       │  │                ║
║  │  │       (same as baseline)                          │  │                ║
║  │  └──────────────────────────────────────────────────┘  │                ║
║  │                                                         │                ║
║  │  ┌──────────────────────────────────────────────────┐  │                ║
║  │  │  STP = Semantic Tube Prediction                  │  │                ║
║  │  │                                                  │  │                ║
║  │  │  Take the hidden states from the LAST block:    │  │                ║
║  │  │     h[1], h[2], h[3], ..., h[T]                  │  │                ║
║  │  │                                                  │  │                ║
║  │  │  Compute "acceleration" at each position:        │  │                ║
║  │  │     a[t] = h[t+1] - 2·h[t] + h[t-1]              │  │                ║
║  │  │                                                  │  │                ║
║  │  │  STP loss = mean( a[t] · a[t] )                  │  │                ║
║  │  │                                                  │  │                ║
║  │  │  WHAT IT DOES:                                   │  │                ║
║  │  │  Penalizes sudden changes in hidden state.      │  │                ║
║  │  │  Forces the trajectory to be locally LINEAR.    │  │                ║
║  │  │                                                  │  │                ║
║  │  │  WHY IT HELPS:                                   │  │                ║
║  │  │  Natural language has smooth semantic flow.     │  │                ║
║  │  │  Adjacent words are usually related.            │  │                ║
║  │  │  Forcing smoothness gives the model a useful    │  │                ║
║  │  │  inductive bias matching natural data.          │  │                ║
║  │  │                                                  │  │                ║
║  │  │  WHY beta = 0.005 (we found this empirically):  │  │                ║
║  │  │     beta = 0.1   → too smooth, hurts (-0.038)   │  │                ║
║  │  │     beta = 0.01  → helps (+0.038)               │  │                ║
║  │  │     beta = 0.005 → BEST (+0.051) ★              │  │                ║
║  │  │     beta = 0.001 → helps less (+0.045)          │  │                ║
║  │  │                                                  │  │                ║
║  │  └──────────────────────────────────────────────────┘  │                ║
║  │                                                         │                ║
║  │  WHY STP + HOURGLASS IS SUPER-ADDITIVE:                │                ║
║  │                                                         │                ║
║  │  STP alone: +0.051 → +0.027 (gap shrinks 47%)          │                ║
║  │  STP + Hourglass: +0.082 → +0.218 (gap GROWS)          │                ║
║  │                                                         │                ║
║  │  The hourglass FORCES abstraction. STP makes those     │                ║
║  │  abstractions COHERENT. Together they reinforce        │                ║
║  │  each other instead of fading.                         │                ║
║  │                                                         │                ║
║  │  PARAMETERS: 0 (it's a loss function modification)     │                ║
║  └─────────────────────────────────────────────────────────┘                ║
╚════════════════════════════════════╦════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEPS 7-9: BACKPROP, OPTIMIZER, REPEAT                                     ║
║                                                                             ║
║  Same as baseline.                                                          ║
║  Muon for matrix weights, Adam for scalars and embeddings.                  ║
║  Gradients now include the STP smoothness signal.                           ║
║                                                                             ║
║  Repeat for 7,000-20,000 steps.                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════════════════════


## ★★★ THE PER-HEAD ATTENTION CONFIDENCE GATE — DETAILED VIEW ★★★

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  THE PROBLEM IT SOLVES                                                      ║
║                                                                             ║
║  In standard attention, every head ALWAYS outputs something. There is no   ║
║  per-token mechanism for the model to say "for THIS token, head 3's        ║
║  output is unreliable, suppress it" or "head 5 is critical, amplify it".   ║
║                                                                             ║
║  The model can learn per-head importance via the OUTPUT projection, but    ║
║  that projection is SHARED across all tokens. So if head H is helpful for  ║
║  nouns but unhelpful for verbs, the output projection has to compromise.   ║
║                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  THE MECHANISM                                                              ║
║                                                                             ║
║      x: input  (B, T, dim=512)                                              ║
║      │                                                                      ║
║      ├──────────────┬──────────────┬─────────────┐                          ║
║      │              │              │             │                          ║
║      ▼              ▼              ▼             ▼                          ║
║    W_q(x)        W_k(x)         W_v(x)       W_g(x)  ★ NEW projection      ║
║      │              │              │             │                          ║
║      ▼              ▼              ▼             ▼                          ║
║      Q              K              V         logits                          ║
║      │              │              │             │                          ║
║      └────────┬─────┴──────────────┘             ▼                          ║
║               │                              sigmoid                         ║
║               ▼                                  │                           ║
║       SDPA(Q, K, V)                              ▼                           ║
║       (causal mask)                            gate                          ║
║               │                       (B, T, num_heads)                      ║
║               │                       values in [0, 1]                       ║
║               ▼                                  │                           ║
║      y_attn (B, H, T, head_dim)                  │                           ║
║               │                                  │                           ║
║               └────────────┬─────────────────────┘                          ║
║                            │                                                ║
║                            ▼                                                ║
║                       y_attn × gate                                          ║
║                       (broadcast over head_dim)                              ║
║                            │                                                ║
║                            ▼                                                ║
║                          W_out                                               ║
║                            │                                                ║
║                            ▼                                                ║
║                         output                                               ║
║                                                                             ║
║  KEY: gate has shape (B, T, num_heads). Each token gets a different gate    ║
║  value for each of the 8 attention heads, each independent of the others.   ║
║  The model decides per-token-per-head how much to trust each head's output. ║
║                                                                             ║
║  COST: dim × num_heads = 512 × 8 = 4,096 params per layer                   ║
║         × 12 layers = 49,152 total = +0.24% over baseline                   ║
║         Step time impact: +0.16% (one tiny matmul per layer)                ║
║                                                                             ║
║  RESULT: val_bpb 1.7473 (Frame) → 1.7396 (Frame+Gate) = -0.0077             ║
║                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  WHAT THE TRAINED GATE LEARNED                                              ║
║                                                                             ║
║  After training, we extracted the learned gate values per (layer, head).    ║
║  Average sigmoid activation on validation data:                             ║
║                                                                             ║
║   Layer  Type      H0     H1     H2     H3     H4     H5     H6     H7     ║
║   ─────  ──────   ──────────────────────────────────────────────────────   ║
║   L0     wide     0.253  0.590  0.665  0.610  0.445  0.160  0.373  0.813   ║
║   L1     wide     0.095  0.058  0.474  0.108  0.575  0.146  0.287  0.081   ║
║   L2     wide     0.011  0.865  0.052  0.386  0.117  0.029  0.812  0.018   ║
║   L3     wide     0.066  0.357  0.253  0.069  0.542  0.008  0.198  0.014   ║
║   L4     narrow   0.000  0.396  0.026  0.036  0.040  0.139  0.015  0.068   ║
║   L5     narrow   0.083  0.337  0.140  0.154  0.929  0.000  0.008  0.130   ║
║   L6     narrow   0.703  0.338  0.052  1.000  0.146  0.388  0.028  0.307   ║
║   L7     narrow   0.100  0.446  0.092  0.053  0.424  0.171  0.106  0.649   ║
║   L8     wide     0.288  0.647  0.059  0.470  0.655  0.051  0.079  0.074   ║
║   L9     wide     0.020  0.038  0.346  0.169  0.000  0.001  0.007  0.007   ║
║   L10    wide     0.024  0.227  0.223  0.144  0.017  0.056  0.003  0.262   ║
║   L11    wide     0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000   ║
║                                                                             ║
║  ★ THREE STRIKING FINDINGS:                                                  ║
║                                                                             ║
║   1. L11 (last layer) attention is COMPLETELY ZEROED OUT.                    ║
║      Every head, every token, every input → gate = 0.000 (std=0.004).        ║
║      The model learned that the FINAL layer's attention is unused.          ║
║                                                                             ║
║   2. The narrow bottleneck (L5, L6) has FULL-TRUST heads (1.000, 0.929).     ║
║      These are the workhorses doing the forced-abstraction job.             ║
║                                                                             ║
║   3. Late layers progressively suppress attention                            ║
║      (L0 0.49 → L9 0.07 → L10 0.12 → L11 0.00).                             ║
║      Consistent with: late transformer layers do "translation"               ║
║      (representation → vocabulary), not "context aggregation".               ║
║                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  WE TRIED TO REMOVE L11 ATTENTION — IT FAILED                               ║
║                                                                             ║
║  HYPOTHESIS:  "L11 gate = 0 means L11 attention is unused. We can            ║
║                surgically remove it and put those params into a wider MLP."  ║
║                                                                             ║
║  EXPERIMENT:  train_gpt_mlx_no_last_attn.py                                  ║
║               - L11 attention REMOVED entirely                               ║
║               - L11 MLP widened from 2x (1024 hidden) to 4x (2048 hidden)    ║
║               - Same total params, just redistributed                        ║
║                                                                             ║
║  RESULT:                                                                    ║
║                                                                             ║
║          Frame + Gate                  val_bpb 1.7396  ← our best            ║
║          Frame + Gate + L11 removed    val_bpb 1.7940  ← significantly worse ║
║          Delta:                              +0.0544 (HURTS)                 ║
║                                                                             ║
║  WHY: "Gate value = 0" ≠ "operation is removable". The gate's suppression    ║
║       was a TRAINED EQUILIBRIUM — given the other 11 layers as configured,   ║
║       L11's attention happens to be set to zero. But removing the option    ║
║       forces a different equilibrium that is worse.                          ║
║                                                                             ║
║  LESSON: Gate-based interpretability tells us what the model isn't using    ║
║          IN ITS CURRENT CONFIGURATION, not what is dispensable.              ║
║                                                                             ║
║          Related: "Lottery Ticket Hypothesis" (Frankle & Carbin 2019) —      ║
║          weights that look "small and removable" depend on initialization    ║
║          and training dynamics, not on inherent importance.                  ║
║                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════════════════════


## INTELLIGENCE MAP — WHERE DOES INTELLIGENCE LIVE NOW?

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  KNOWLEDGE (facts, patterns, language rules):                               ║
║  ┌──────────────────────────────────────┐                                   ║
║  │  ★★★ MLP WEIGHTS — but TIERED        │                                   ║
║  │                                      │                                   ║
║  │  Stage 1 wide MLPs (Blocks 1-4):     │  surface-level patterns          ║
║  │    4 × 1.05M = 4.19M params          │  vocab, morphology, syntax       ║
║  │                                      │                                   ║
║  │  Stage 2 narrow MLPs (Blocks 5-8):   │  ABSTRACT patterns ★★★           ║
║  │    4 × 0.52M = 2.10M params          │  semantics, relationships        ║
║  │                                      │  meaning, intent                 ║
║  │                                      │                                   ║
║  │  Stage 3 wide MLPs (Blocks 9-12):    │  prediction-mapping patterns     ║
║  │    4 × 1.05M = 4.19M params          │  abstract → concrete tokens     ║
║  │                                      │                                   ║
║  │  Total MLP: 10.49M params            │                                   ║
║  │                                      │                                   ║
║  │  ★ The narrow middle is where the    │                                   ║
║  │    REAL THINKING happens. It has     │                                   ║
║  │    LESS capacity but does MORE work. │                                   ║
║  └──────────────────────────────────────┘                                   ║
║                                                                             ║
║  RELATIONSHIPS (which tokens relate to which):                              ║
║  ┌──────────────────────────────────────┐                                   ║
║  │  ★★ ATTENTION + GATE                 │                                   ║
║  │  12 × 786K + 12 × 4K = 9.49M params  │  1.3× the baseline's attention   ║
║  │  Still 4 KV heads, 8 query heads     │  More layers = more relationships║
║  │  + per-token, per-head gate          │  Model can suppress unreliable    ║
║  │    (49K extra params, 0.24% of model)│  heads on a per-token basis       ║
║  └──────────────────────────────────────┘                                   ║
║                                                                             ║
║  WORD MEANING (what each token represents):                                 ║
║  ┌──────────────────────────────────────┐                                   ║
║  │  ★ EMBEDDING TABLE (unchanged)        │                                   ║
║  │  524K params                          │                                   ║
║  └──────────────────────────────────────┘                                   ║
║                                                                             ║
║  ★★★ NEW: COHERENCE OF ABSTRACTIONS                                         ║
║  ┌──────────────────────────────────────┐                                   ║
║  │  STP loss term (no parameters)        │                                   ║
║  │  Shapes the hidden state trajectory   │                                   ║
║  │  through gradient signal              │                                   ║
║  │  Ensures abstract representations     │                                   ║
║  │  evolve smoothly across positions     │                                   ║
║  │  THIS IS THE GLUE                     │                                   ║
║  └──────────────────────────────────────┘                                   ║
║                                                                             ║
║  ★★★ NEW: FORCED ABSTRACTION PRESSURE                                       ║
║  ┌──────────────────────────────────────┐                                   ║
║  │  Architectural bottleneck (no params) │                                   ║
║  │  Encoded in W_expand/W_shrink shapes  │                                   ║
║  │  Cannot be bypassed (skip removed)    │                                   ║
║  │  Forces compression at training time  │                                   ║
║  │  THIS IS THE CHISEL                   │                                   ║
║  └──────────────────────────────────────┘                                   ║
║                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════


## PARAMETER BUDGET (where 16 MB goes — hourglass version)

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  Total: ~20.5M parameters → fit in 16 MB via int8 + zlib                    ║
║  (vs baseline 17M — we packed in MORE params with the same budget)          ║
║                                                                             ║
║  ┌────────────────────────────────────────────────────────────────────┐      ║
║  │█████████████████████████████████████████████│ MLP 51%              │      ║
║  │██████████████████████████████████████████│ Attention 46%           │      ║
║  │██│ Embeddings 3%                                                   │      ║
║  └────────────────────────────────────────────────────────────────────┘      ║
║                                                                             ║
║  Per-stage breakdown:                                                       ║
║  ┌────────────────────────────────────────────────────────────────────┐      ║
║  │                                                                    │      ║
║  │  STAGE 1 (Blocks 1-4) — RECOGNITION                                │      ║
║  │    Attention: 4 × 786,432 = 3,145,728                              │      ║
║  │    MLP wide:  4 × 1,048,576 = 4,194,304                            │      ║
║  │    Subtotal:                  7,340,032                            │      ║
║  │                                                                    │      ║
║  │  STAGE 2 (Blocks 5-8) — ABSTRACTION ★ NARROW ★                     │      ║
║  │    Attention: 4 × 786,432 = 3,145,728                              │      ║
║  │    MLP narrow: 4 × 524,288 = 2,097,152  ← HALF the wide MLP        │      ║
║  │    Subtotal:                  5,242,880                            │      ║
║  │                                                                    │      ║
║  │  STAGE 3 (Blocks 9-12) — PREDICTION                                │      ║
║  │    Attention: 4 × 786,432 = 3,145,728                              │      ║
║  │    MLP wide:  4 × 1,048,576 = 4,194,304                            │      ║
║  │    Subtotal:                  7,340,032                            │      ║
║  │                                                                    │      ║
║  │  TOTAL BLOCKS: 19,922,944 params                                   │      ║
║  │  + Embeddings:    524,288                                          │      ║
║  │  + Norms/scales: ~50,000                                           │      ║
║  │  ≈ 20.5M params                                                    │      ║
║  │                                                                    │      ║
║  └────────────────────────────────────────────────────────────────────┘      ║
║                                                                             ║
║  WHAT THE HOURGLASS SAVED:                                                  ║
║  ┌────────────────────────────────────────────────────────────────────┐      ║
║  │                                                                    │      ║
║  │  Uniform 12L would have:                                           │      ║
║  │    12 × 1,048,576 MLP params = 12,582,912                          │      ║
║  │                                                                    │      ║
║  │  Hourglass 12L has:                                                │      ║
║  │    8 wide × 1,048,576 + 4 narrow × 524,288                         │      ║
║  │    = 8,388,608 + 2,097,152 = 10,485,760                            │      ║
║  │                                                                    │      ║
║  │  SAVED: 2,097,152 MLP params                                       │      ║
║  │                                                                    │      ║
║  │  Those saved params became 3 EXTRA layers (9L → 12L) within        │      ║
║  │  the same 16MB int8-zlib budget. More depth from same budget.      │      ║
║  │                                                                    │      ║
║  └────────────────────────────────────────────────────────────────────┘      ║
║                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════


## WHY THIS WORKS — THE THEORETICAL STORY

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  1. THE INFORMATION BOTTLENECK PRINCIPLE (Tishby 2000)                      ║
║                                                                             ║
║     Optimal representations COMPRESS the input while PRESERVING task-       ║
║     relevant information. Standard transformers rely on training to         ║
║     discover this compression through gradient descent.                     ║
║                                                                             ║
║     Hourglass MAKES IT ARCHITECTURAL. The model can't avoid compression     ║
║     because the hidden dimension is physically smaller. The bottleneck      ║
║     is in the structure, not the optimization.                              ║
║                                                                             ║
║                                                                             ║
║  2. NATURAL SEMANTIC SMOOTHNESS                                             ║
║                                                                             ║
║     "The cat sat on the mat"                                                ║
║       └→ at every step, meaning evolves a LITTLE                            ║
║       └→ "the" → "the cat" → "the cat sat" → ...                            ║
║       └→ adjacent positions are semantically RELATED                        ║
║                                                                             ║
║     STP exploits this. By penalizing acceleration of hidden states,         ║
║     it tells the model: "your understanding should evolve smoothly,         ║
║     not lurch around." Matches the data structure.                          ║
║                                                                             ║
║                                                                             ║
║  3. SKIP CONNECTIONS UNDERMINE BOTTLENECKS                                  ║
║                                                                             ║
║     ResNets and U-Nets use skips to ease gradient flow.                     ║
║     But skips = shortcuts = "you don't have to do the hard thing."          ║
║                                                                             ║
║     In a bottleneck architecture, the "hard thing" IS the point.            ║
║     Removing skips removes the shortcut. The model has no choice but        ║
║     to actually compress through the narrow layers.                         ║
║                                                                             ║
║                                                                             ║
║  4. THE GAP GROWS (super-additivity)                                        ║
║                                                                             ║
║     With more training, the hourglass model:                                ║
║       a) gets BETTER at compressing through the bottleneck                  ║
║       b) develops MORE refined abstract representations                     ║
║       c) USES the extra 3 layers more effectively                           ║
║                                                                             ║
║     Meanwhile the baseline plateaus — it has only 9 layers and no           ║
║     architectural pressure to abstract.                                     ║
║                                                                             ║
║     200 steps:  +0.082                                                      ║
║     500 steps:  +0.068                                                      ║
║     962 steps:  +0.218  ← gap GROWS                                         ║
║                                                                             ║
║     The advantage compounds. Longer training → bigger gap.                  ║
║                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════


## SIDE-BY-SIDE: BASELINE vs HOURGLASS

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║   BASELINE                          OUR HOURGLASS                          ║
║   ─────────                         ──────────────                         ║
║                                                                             ║
║   ┌──────────────┐                  ┌──────────────┐                       ║
║   │  Block 1     │ MLP=1024         │  Block 1     │ MLP=1024              ║
║   │  Block 2     │ MLP=1024         │  Block 2     │ MLP=1024              ║
║   │  Block 3     │ MLP=1024         │  Block 3     │ MLP=1024              ║
║   │  Block 4     │ MLP=1024         │  Block 4     │ MLP=1024              ║
║   │  Block 5     │ MLP=1024  ◄┐     │  Block 5     │ MLP=512  ◄┐           ║
║   │  Block 6     │ MLP=1024  ◄┤     │  Block 6     │ MLP=512  ◄┤ NARROW   ║
║   │  Block 7     │ MLP=1024  ◄┤     │  Block 7     │ MLP=512  ◄┤ STAGE    ║
║   │  Block 8     │ MLP=1024  ◄┘     │  Block 8     │ MLP=512  ◄┘           ║
║   │  Block 9     │ MLP=1024         │  Block 9     │ MLP=1024              ║
║   └──────────────┘                  │  Block 10    │ MLP=1024              ║
║                                     │  Block 11    │ MLP=1024              ║
║   9 layers, uniform                 │  Block 12    │ MLP=1024              ║
║   U-Net skip connections            └──────────────┘                       ║
║   Pure cross-entropy loss                                                   ║
║                                     12 layers, hourglass MLP                ║
║                                     NO skip connections                     ║
║                                     CE + 0.005 × STP loss                   ║
║                                                                             ║
║   ~17M params                       ~20.5M params (same 16MB budget)        ║
║   val_bpb ≈ 1.965                   val_bpb ≈ 1.747  (+0.218)               ║
║                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════


## ABLATION — WHAT EACH PIECE CONTRIBUTES

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║   POSITIVE CONTRIBUTIONS (starting from 9L baseline ~1.965 BPB):            ║
║                                                                             ║
║   + 3 extra layers (12L uniform):       +0.029 BPB                         ║
║   + Hourglass shape (12L hourglass):    +0.101 BPB                         ║
║   + STP smoothness (β=0.005):           +0.027 BPB                         ║
║   + Remove U-Net skip connections:      +0.088 BPB  ← BIGGEST single win   ║
║                                         ───────                             ║
║   = Frame baseline:                     +0.218 BPB → 1.7473                ║
║                                                                             ║
║   + Per-head attention gate:            +0.0077 BPB ← NEW                   ║
║                                         ───────                             ║
║   = Frame + Gate (current best):        +0.226 BPB → 1.7396                ║
║                                                                             ║
║   The Per-Head Gate adds only 49,152 params (+0.24%) for the +0.0077       ║
║   improvement. Best ROI per parameter of any addition we tested.            ║
║                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║   NEGATIVE RESULTS (things we tried that did NOT work):                     ║
║                                                                             ║
║   Crown shape (1,1,2,2,2,2,2,2,2,2,1,1):                                    ║
║      val_bpb 1.7625 (vs Frame 1.7473) = +0.0152 worse                       ║
║      Tests the OPPOSITE shape (narrow-wide-narrow). Same total params.      ║
║      Frame wins at sub-100M scale.                                          ║
║      (NOTE: Baroian & Notebomer 2025 found Crown WINS at 180M scale —       ║
║       this may be a scale-dependent reversal, untested at other scales)     ║
║                                                                             ║
║   L11 attention surgically removed + L11 MLP widened to 4x:                 ║
║      val_bpb 1.7940 (vs Frame+Gate 1.7396) = +0.0544 worse                  ║
║      Hypothesis was: gate=0 means L11 attention is removable.               ║
║      REFUTED. The gate's zero was a trained equilibrium, not architectural. ║
║                                                                             ║
║   Compositional MLP (static-scalar version):                                ║
║      val_bpb 1.7481 (vs Frame+Gate 1.7396) = +0.0085 worse                  ║
║      Each MLP composes with previous same-shape layers' patterns via        ║
║      learnable scalar weights. Showed BETTER train loss but WORSE val_bpb   ║
║      — generalization gap. The static version overfits to training mix.    ║
║      A per-token routed version is untested.                                ║
║                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║   FINAL SCOREBOARD (all measured at Mac M5, batch 8192, 600s, seed 1337)    ║
║                                                                             ║
║   Architecture                              val_bpb     Δ vs Frame          ║
║   ─────────────────────────────────────────────────────────────────────     ║
║   9L baseline (extrapolated)                ~1.965         +0.218           ║
║   12L uniform                                1.9360        +0.189           ║
║   12L hourglass + STP (with skips)           1.8357        +0.088           ║
║   12L hourglass + STP, NO skips (Frame)      1.7473         0.000           ║
║   Crown shape control                        1.7625        +0.0152          ║
║   Frame + Compositional MLP                  1.7481        +0.0008          ║
║ ★ Frame + Per-Head Gate (BEST)               1.7396        −0.0077          ║
║   Frame + Gate + L11 attn removed            1.7940        +0.0467          ║
║                                                                             ║
║   IMPORTANT: All numbers measured on Mac with TRAIN_BATCH_TOKENS=8192        ║
║   (~7.9M total tokens). The competition leaderboard uses 8xH100s with        ║
║   TRAIN_BATCH_TOKENS=524288 (~503M tokens). These val_bpb numbers are        ║
║   NOT directly comparable to the leaderboard. They ARE comparable to        ║
║   each other since all use identical compute settings.                       ║
║                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
```
