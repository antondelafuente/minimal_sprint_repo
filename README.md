# Steered Model for Thought Anchors

Activation steering tools for language models, with support for **prefill-based steering** for Thought Anchors experiments.

## What's in this repo

- `steered_model.py` - Unified steered model with prefill support
- `steering_vectors/` - Pre-computed steering vectors

## The Key Idea: Prefill-Based Steering

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
from steered_model import SteeredModel

# Initialize
steerer = SteeredModel(
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

- **`generate_with_prefill(problem, prefill, reasoning_level="low", ...)`** - Generate a single completion with prefilled reasoning.

- **`generate_batch_with_prefill(problem, prefill, n=100, ...)`** - Generate N completions from the same prefill point (efficient batched version).

- **`resample_from_position(problem, sentences, position, n=100, ...)`** - Convenience method: prefill up to sentence position K, then generate N continuations.

- **`set_alpha(new_alpha)`** - Change steering strength without reloading the model.

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

- `steered_model.py` - Unified model class supporting both regular and prefill-based steering
- `steering_vectors/steering_vectors_normalized.pt` - Multi-layer steering vectors
- `steering_vectors/layer15_feature_normalized.pt` - Single-layer feature vector
- `generate_with_prefill_vllm.py` - Standalone API for prefilled generation with vLLM
- `usage_examples.py` - Examples demonstrating the prefill generation API

---

## Prefill Generation API (New!)

### Generate with vLLM and Prefilled Reasoning

The `generate_with_prefill_vllm.py` module provides a minimal, standalone API for generating completions with prefilled reasoning. This is the core technique behind Thought Anchors experiments.

### Installation

```bash
pip install requests  # Only dependency
```

### Quick Start

```python
from generate_with_prefill_vllm import generate_rollout, generate_rollouts

# Generate from scratch
result = generate_rollout(
    problem="What is 2+2?",
    temperature=0.0
)[0]
print(result['full_text'])

# Continue from prefilled reasoning
result = generate_rollout(
    problem="What is 2+2?",
    prefill="Let me calculate: 2+2 = ",  # Model continues from here
    temperature=0.0
)[0]
print(result['continuation'])  # Likely outputs "4"
```

### The Key Insight

The critical detail is **not closing the assistant's turn** after the prefill. This makes the model continue as if it had written the prefill itself:

```python
prompt = (
    "<|start|>system<|message|>...<|end|>"
    "<|start|>user<|message|>{problem}<|end|>"
    "<|start|>assistant<|channel|>analysis<|message|>{prefill}"
    # NO CLOSING TOKEN - model continues seamlessly
)
```

### Batch Generation for Statistics

Generate many rollouts efficiently using vLLM's batched inference:

```python
# Generate 100 samples from the same prefill point
rollouts = generate_rollouts(
    problem="Is 17 prime?",
    prefill="Let me check if 17 is divisible by any number.",
    n=100,
    temperature=0.8
)

# Analyze results
for rollout in rollouts:
    print(rollout['continuation'])  # What the model generated
    # Extract activations here for your experiments
```

### Thought Anchors Style Comparison

Compare completions with/without a critical sentence:

```python
# Baseline: Include critical sentence
baseline = generate_rollouts(
    problem="Calculate: (15 * 3) + (12 / 4)",
    prefill="15 * 3 = 45. 12 / 4 = 3. Now I add them together.",  # Includes key sentence
    n=100
)

# Intervention: Stop before critical sentence
intervention = generate_rollouts(
    problem="Calculate: (15 * 3) + (12 / 4)",
    prefill="15 * 3 = 45. 12 / 4 = 3.",  # Excludes "Now I add them together"
    n=100
)

# Compare how often each gets the right answer
```

### Integration with Activation Extraction

The API is designed to integrate easily with activation extraction tools:

```python
from generate_with_prefill_vllm import generate_rollout

# Define prefills at different positions
prefills = [
    "",  # Position 0: No prefill
    "I need to calculate 2^10.",  # Position 1
    "I need to calculate 2^10. This means 2 multiplied by itself 10 times.",  # Position 2
]

# Generate and extract activations
for prefill in prefills:
    rollout = generate_rollout(problem="What is 2^10?", prefill=prefill)[0]

    # Your activation extraction code here
    # activations = extract_activations(rollout['full_text'])
    # process_activations(activations)
```

### API Reference

#### Core Functions

- **`generate_rollout(problem, prefill="", ...)`** - Generate N completions (batched)
- **`generate_rollouts(problem, prefill="", n=10, ...)`** - Convenience alias for multiple rollouts
- **`build_harmony_prompt(problem, prefill, reasoning_level)`** - Build the prompt (exposed for debugging)

#### Helper Functions

- **`extract_final_answer(text)`** - Extract the final answer from a completion
- **`parse_channels(text)`** - Parse all channels (analysis, final, commentary)

### Run Examples

```bash
# Test if vLLM server is running
python usage_examples.py

# Run specific example (1-6)
python usage_examples.py 4  # Run thought anchors comparison
```

### Requirements

- vLLM server running: `vllm serve /workspace/gpt-oss-20b --host 0.0.0.0 --port 8000`
- Or set custom endpoint: `api_url="http://your-server:8000/v1/completions"`
