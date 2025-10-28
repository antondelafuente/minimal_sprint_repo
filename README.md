# Steered Model for Thought Anchors

Activation steering tools for language models, with support for **prefill-based steering** for Thought Anchors experiments.

## What's in this repo

- `steered_model.py` - Basic steered model (steers all generated tokens)
- `prefill_steered_model.py` - **Prefill-based steering** (the main focus)
- `steered_model_base.py` - Shared base class
- `steering_vectors/` - Pre-computed steering vectors

## The Key Idea: PrefillSteeredModel

**Regular steering**: You give the model a question, it generates a full chain-of-thought (CoT), and steering affects ALL generated tokens.

**Prefill-based steering**: You give the model a question PLUS the first K sentences already written (the "prefill"). The model continues from there, and steering is applied ONLY to the continuation, not the prefilled part.

### Why This Matters - Thought Anchors

This lets you test: *"What if the model reasoned correctly up to sentence K, then we apply steering from there?"*

Example scenario:
1. Baseline CoT: "Sentence 1. Sentence 2. Sentence 3. Sentence 4."
2. You **anchor** sentences 1-2 (lock them in as the prefill)
3. Generate sentences 3+ with steering applied
4. Resample 100 times to see how steering affects completion from that specific point

This is way more precise than regular steering because you're isolating **which part** of the reasoning chain is affected.

## Quick Start

```python
from prefill_steered_model import PrefillSteeredModel

# Initialize
steerer = PrefillSteeredModel(
    model_path="/workspace/gpt-oss-20b",
    steering_vector_path="steering_vectors/steering_vectors_normalized.pt",
    layer=None,  # Multi-layer steering
    alpha=100    # Steering strength
)

# Generate with prefilled reasoning
result = steerer.generate_with_prefill(
    problem="What is 2+2?",
    prefill="Let me think step by step. First,",  # Lock in this start
    reasoning_level="low"
)

print(result['reasoning'])      # Full CoT (prefill + continuation)
print(result['continuation'])   # Just what was generated
print(result['final_answer'])

# Clean up
steerer.close()
```

## Thought Anchors Workflow

Resample from a specific sentence position:

```python
# You have a baseline CoT broken into sentences
problem = "What is 2+2?"
sentences = [
    "Let me break this down.",
    "First, I'll add 2 and 2.",
    "That gives me 4.",
    "So the answer is 4."
]

# Resample from position 2 (after "First, I'll add 2 and 2.")
# This anchors sentences 0-1, then generates from sentence 2 onwards
results = steerer.resample_from_position(
    problem=problem,
    sentences=sentences,
    position=2,
    n=100,  # Generate 100 samples
    temperature=0.8
)

# Analyze: How often does steering affect the final answer?
correct = sum(1 for r in results if "4" in r['final_answer'])
print(f"Accuracy: {correct}/100")
```

## Batch Generation

For statistics, generate many samples from the same prefill point:

```python
# Generate 100 continuations from the same starting point
results = steerer.generate_batch_with_prefill(
    problem="What is 2+2?",
    prefill="Let me calculate.",
    n=100,
    temperature=0.8
)

# Each result has: reasoning, continuation, final_answer, n_tokens, hit_eos, is_truncated
for r in results:
    print(r['continuation'])  # What the model generated
```

## How It Works

1. **Builds Harmony prompt** with your prefill, ending mid-stream (no closing tag)
2. **Calculates token length** of the prefill
3. **Sets steering to start AFTER** that token position
4. **Generates continuation** with steering applied only to new tokens
5. **Returns both** the full reasoning (prefill + continuation) and just the continuation

## Key Methods

### `generate_with_prefill(problem, prefill, reasoning_level="low", ...)`
Generate a single completion with prefilled reasoning.

### `generate_batch_with_prefill(problem, prefill, n=100, ...)`
Generate N completions from the same prefill point (efficient batched version).

### `resample_from_position(problem, sentences, position, n=100, ...)`
Convenience method: prefill up to sentence position K, then generate N continuations.

### `set_alpha(new_alpha)`
Change steering strength without reloading the model.

## Setup

```bash
pip install transformers accelerate
```

Model location: `/workspace/gpt-oss-20b`

## Parameters

- `alpha = 0` - No steering (baseline)
- `alpha > 0` - Steer in positive direction
- `alpha < 0` - Steer in negative direction
- `layer = None` - Multi-layer steering (uses all layers in .pt file)
- `layer = 14` - Single-layer steering (layer 14 only)

## Files

- `prefill_steered_model.py` - Main class for Thought Anchors experiments
- `steered_model.py` - Simpler version (steers after full prompt, no prefill support)
- `steered_model_base.py` - Shared model loading, hooks, and generation logic
- `steering_vectors/steering_vectors_normalized.pt` - Multi-layer steering vectors
- `steering_vectors/layer15_feature_normalized.pt` - Single-layer feature vector
