#!/usr/bin/env python3
"""
Minimal API for generating rollouts with prefilled reasoning using vLLM.

Key insight: Don't close the assistant turn - model continues seamlessly from prefill.

Usage:
    from generate_with_prefill_vllm import generate_rollout, generate_rollouts

    # Single rollout from scratch
    result = generate_rollout(problem="What is 2+2?")[0]

    # Multiple rollouts with prefill
    results = generate_rollouts(
        problem="What is 2+2?",
        prefill="Let me think carefully. First, I need to add these numbers.",
        n=100
    )
"""
import os
import requests
from typing import List, Dict, Optional, Union


def build_harmony_prompt(
    problem: str,
    prefill: str = "",
    reasoning_level: str = "low"
) -> str:
    """
    Build Harmony prompt with optional prefilled reasoning.

    The key insight is to NOT close the assistant's turn after the prefill,
    which forces the model to continue generating as if it had written the
    prefill text itself, rather than treating it as separate context.

    Args:
        problem: The task/problem description
        prefill: Reasoning text to continue from (empty = generate from scratch)
        reasoning_level: One of: minimal, low, medium, high

    Returns:
        Complete prompt string (no closing token on assistant turn)
    """
    return (
        f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
        f"Reasoning: {reasoning_level}\n"
        "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        f"<|start|>user<|message|>{problem}<|end|>"
        f"<|start|>assistant<|channel|>analysis<|message|>{prefill}"
        # NO CLOSING TOKEN - this is the critical detail that makes continuation work
    )


def generate_rollout(
    problem: str,
    prefill: str = "",
    reasoning_level: str = "low",
    temperature: float = 0.8,
    max_tokens: int = 4000,
    n: int = 1,
    model_path: Optional[str] = None,
    api_url: str = "http://localhost:8000/v1/completions",
    stop_tokens: Optional[List[str]] = None,
    return_raw: bool = False
) -> Union[List[Dict[str, str]], requests.Response]:
    """
    Generate rollouts continuing from prefilled reasoning.

    The model will seamlessly continue from the prefill as if it had generated
    that text itself. This is useful for:
    - Thought anchors analysis (measuring importance of specific reasoning steps)
    - Controlled generation experiments
    - Extracting activations at specific reasoning points

    Args:
        problem: Task description or problem to solve
        prefill: Reasoning to continue from (empty = start fresh)
        reasoning_level: One of: minimal, low, medium, high
        temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        max_tokens: Maximum tokens to generate
        n: Number of rollouts to generate (uses vLLM's batched generation)
        model_path: Path to model (defaults to MODEL_PATH env var, then gpt-oss-20b)
        api_url: vLLM API endpoint URL
        stop_tokens: Custom stop tokens (defaults to ["<|return|>", "<|call|>"])
        return_raw: If True, return raw API response instead of parsed results

    Returns:
        If return_raw=False (default):
            List of dicts with keys:
            - 'continuation': Text generated after prefill
            - 'full_text': Prefill + continuation
            - 'prefill': The prefill text used
        If return_raw=True:
            Raw requests.Response object

    Raises:
        requests.HTTPError: If API request fails

    Example:
        >>> # Generate from scratch
        >>> result = generate_rollout("What is 2+2?")[0]
        >>> print(result['full_text'])

        >>> # Continue from prefill
        >>> result = generate_rollout(
        ...     "What is 2+2?",
        ...     prefill="Let me calculate: 2+2 = ",
        ...     temperature=0.0
        ... )[0]
        >>> print(result['continuation'])  # Will likely be "4"
    """
    if model_path is None:
        model_path = os.getenv("MODEL_PATH", "/workspace/gpt-oss-20b")

    if stop_tokens is None:
        stop_tokens = ["<|return|>", "<|call|>"]

    prompt = build_harmony_prompt(problem, prefill, reasoning_level)

    response = requests.post(api_url, json={
        "model": model_path,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n,
        "stop": stop_tokens
    })

    response.raise_for_status()

    if return_raw:
        return response

    results = []
    for choice in response.json()['choices']:
        continuation = choice['text']
        results.append({
            'continuation': continuation,
            'full_text': prefill + continuation,
            'prefill': prefill
        })

    return results


def generate_rollouts(
    problem: str,
    prefill: str = "",
    n: int = 10,
    **kwargs
) -> List[Dict[str, str]]:
    """
    Convenience function for generating multiple rollouts.

    This is an alias for generate_rollout with n>1, providing a more
    intuitive name when generating multiple samples.

    Args:
        problem: Task description
        prefill: Reasoning to continue from
        n: Number of rollouts to generate (default 10)
        **kwargs: Additional arguments passed to generate_rollout

    Returns:
        List of rollout dictionaries

    Example:
        >>> # Generate 100 rollouts with temperature variation
        >>> rollouts = generate_rollouts(
        ...     "Solve: x^2 - 5x + 6 = 0",
        ...     prefill="To solve this quadratic equation,",
        ...     n=100,
        ...     temperature=0.9
        ... )
        >>> print(f"Generated {len(rollouts)} rollouts")
    """
    return generate_rollout(problem, prefill, n=n, **kwargs)


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from a completion (if present).

    Looks for the final channel marker and extracts the content.

    Args:
        text: The full completion text

    Returns:
        The final answer text if found, None otherwise
    """
    import re

    # Look for final channel content
    pattern = r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()
    return None


def parse_channels(text: str) -> Dict[str, str]:
    """
    Parse all channels from a completion.

    Args:
        text: The full completion text

    Returns:
        Dict mapping channel names to their content
    """
    import re

    channels = {}

    # Extract analysis channel (everything before final)
    final_split = text.split('<|start|>assistant<|channel|>final<|message|>')
    if len(final_split) > 0:
        channels['analysis'] = final_split[0].strip()

    # Extract final channel if present
    if len(final_split) > 1:
        # Get text until stop token
        final_text = final_split[1].split('<|')[0].strip()
        channels['final'] = final_text

    # Look for commentary channel (rare but possible)
    commentary_pattern = r'<\|start\|>assistant<\|channel\|>commentary<\|message\|>(.*?)(?:<\|start\|>|<\|end\|>|$)'
    commentary_match = re.search(commentary_pattern, text, re.DOTALL)
    if commentary_match:
        channels['commentary'] = commentary_match.group(1).strip()

    return channels


if __name__ == "__main__":
    # Simple test
    print("Testing generate_with_prefill_vllm...")

    # Test 1: Generate from scratch
    result = generate_rollout(
        problem="What is 2+2?",
        n=1,
        max_tokens=100,
        temperature=0.0
    )[0]

    print("\nTest 1 - Generate from scratch:")
    print(f"Full text: {result['full_text'][:200]}...")

    # Test 2: Generate with prefill
    result = generate_rollout(
        problem="What is 2+2?",
        prefill="Let me calculate this step by step. I need to add 2 and 2.",
        n=1,
        max_tokens=100,
        temperature=0.0
    )[0]

    print("\nTest 2 - Generate with prefill:")
    print(f"Prefill: {result['prefill']}")
    print(f"Continuation: {result['continuation'][:200]}...")

    print("\n✅ Basic tests complete!")