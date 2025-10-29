#!/usr/bin/env python3
"""
Usage examples for generate_with_prefill_vllm.

These examples demonstrate common use cases including:
- Basic generation from scratch
- Continuing from prefilled reasoning
- Batch generation for statistics
- Thought anchors style comparison
- Integration point for activation extraction
"""

from generate_with_prefill_vllm import (
    generate_rollout,
    generate_rollouts,
    extract_final_answer,
    parse_channels
)


def example_basic():
    """Example 1: Basic generation from scratch."""
    print("="*80)
    print("EXAMPLE 1: Generate from scratch")
    print("="*80)

    result = generate_rollout(
        problem="What is the capital of France?",
        reasoning_level="minimal",
        temperature=0.0,
        max_tokens=200
    )[0]

    print(f"Full response:\n{result['full_text']}\n")

    # Extract final answer if present
    final = extract_final_answer(result['full_text'])
    if final:
        print(f"Final answer: {final}")


def example_with_prefill():
    """Example 2: Continue from prefilled reasoning."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Continue from prefill")
    print("="*80)

    prefill_text = (
        "I need to solve this step by step. "
        "The capital of France is a major European city. "
        "It's known for the Eiffel Tower. "
        "The city is"
    )

    result = generate_rollout(
        problem="What is the capital of France?",
        prefill=prefill_text,
        temperature=0.0,
        max_tokens=50
    )[0]

    print(f"Prefill: {result['prefill']}")
    print(f"Continuation: {result['continuation']}")
    print(f"\nModel continued with: '{result['continuation'].strip()}'")


def example_batch_generation():
    """Example 3: Generate multiple rollouts for statistics."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Batch generation for statistics")
    print("="*80)

    problem = "Is the number 17 prime? Think step by step."

    # Generate 10 rollouts
    rollouts = generate_rollouts(
        problem=problem,
        prefill="Let me check if 17 is divisible by any number. ",
        n=10,
        temperature=0.8,
        max_tokens=500
    )

    print(f"Generated {len(rollouts)} rollouts")

    # Analyze final answers
    answers = []
    for i, rollout in enumerate(rollouts):
        final = extract_final_answer(rollout['full_text'])
        if final:
            # Check if answer says yes/true/prime
            is_prime = any(word in final.lower() for word in ['yes', 'true', 'prime'])
            answers.append(is_prime)
            print(f"  Rollout {i+1}: {'Prime' if is_prime else 'Not prime'} - {final[:50]}")

    if answers:
        print(f"\nConsensus: {sum(answers)}/{len(answers)} say it's prime")


def example_thought_anchors():
    """Example 4: Thought anchors style comparison."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Thought anchors comparison")
    print("="*80)

    problem = "Calculate: (15 * 3) + (12 / 4)"

    # Baseline: Include the critical sentence
    baseline_prefill = (
        "I need to follow order of operations. "
        "First, I'll compute the multiplication and division. "
        "15 * 3 = 45. "
        "12 / 4 = 3. "
        "Now I add them together."  # Critical sentence
    )

    # Intervention: Stop before critical sentence
    intervention_prefill = (
        "I need to follow order of operations. "
        "First, I'll compute the multiplication and division. "
        "15 * 3 = 45. "
        "12 / 4 = 3."  # Stop here
    )

    print("Comparing completions with/without critical sentence...")

    # Generate with baseline (includes critical sentence)
    baseline = generate_rollout(
        problem=problem,
        prefill=baseline_prefill,
        n=5,
        temperature=0.8
    )

    # Generate with intervention (excludes critical sentence)
    intervention = generate_rollout(
        problem=problem,
        prefill=intervention_prefill,
        n=5,
        temperature=0.8
    )

    print(f"\nBaseline (with 'Now I add them together'):")
    for i, r in enumerate(baseline[:3]):
        continuation_preview = r['continuation'][:50].replace('\n', ' ')
        print(f"  Sample {i+1}: ...{continuation_preview}")

    print(f"\nIntervention (without that sentence):")
    for i, r in enumerate(intervention[:3]):
        continuation_preview = r['continuation'][:50].replace('\n', ' ')
        print(f"  Sample {i+1}: ...{continuation_preview}")

    print("\nThis demonstrates how specific sentences anchor the reasoning trajectory")


def example_for_activation_extraction():
    """Example 5: Setup for activation extraction."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Integration point for activation extraction")
    print("="*80)

    problem = "What is 2^10?"

    # Define specific prefills to test
    prefills = [
        "",  # No prefill
        "I need to calculate 2^10.",
        "I need to calculate 2^10. This means 2 multiplied by itself 10 times.",
        "I need to calculate 2^10. This means 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 = ",
    ]

    print("Generating rollouts at different prefill positions...\n")

    all_rollouts = []
    for prefill in prefills:
        rollout = generate_rollout(
            problem=problem,
            prefill=prefill,
            temperature=0.0,
            max_tokens=200
        )[0]

        all_rollouts.append(rollout)

        # This is where you would extract activations
        # activations = your_extraction_function(rollout['full_text'])

        preview = prefill[:30] + "..." if len(prefill) > 30 else prefill or "(empty)"
        print(f"Prefill: {preview}")
        print(f"→ Continuation starts: {rollout['continuation'][:40]}...")
        print()

    print("Your partner can insert activation extraction here:")
    print("  for rollout in all_rollouts:")
    print("      activations = extract_activations(rollout['full_text'])")
    print("      # Process activations...")


def example_parse_channels():
    """Example 6: Parse different channels from output."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Parse channels")
    print("="*80)

    result = generate_rollout(
        problem="What is 5 + 3?",
        reasoning_level="low",
        max_tokens=200
    )[0]

    channels = parse_channels(result['full_text'])

    print("Parsed channels:")
    for channel, content in channels.items():
        preview = content[:100] + "..." if len(content) > 100 else content
        print(f"\n{channel.upper()}:")
        print(f"  {preview}")


if __name__ == "__main__":
    import sys

    # Check if vLLM server is running
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=1)
        print("✅ vLLM server is running\n")
    except:
        print("⚠️  Warning: vLLM server not accessible at localhost:8000")
        print("   Start it with: vllm serve /workspace/gpt-oss-20b --host 0.0.0.0 --port 8000\n")
        sys.exit(1)

    # Run examples
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_basic()
        elif example_num == "2":
            example_with_prefill()
        elif example_num == "3":
            example_batch_generation()
        elif example_num == "4":
            example_thought_anchors()
        elif example_num == "5":
            example_for_activation_extraction()
        elif example_num == "6":
            example_parse_channels()
        else:
            print(f"Unknown example number: {example_num}")
    else:
        print("Usage: python usage_examples.py [1-6]")
        print("\nAvailable examples:")
        print("  1. Basic generation from scratch")
        print("  2. Continue from prefilled reasoning")
        print("  3. Batch generation for statistics")
        print("  4. Thought anchors style comparison")
        print("  5. Setup for activation extraction")
        print("  6. Parse different channels")
        print("\nRunning example 1 and 2 as demo...\n")

        example_basic()
        example_with_prefill()