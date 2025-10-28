#!/usr/bin/env python3
"""
Prefill-based steered model for Thought Anchors experiments.
Applies steering only AFTER a specified prefill point in the CoT.

This is based on the targeted steering approach from steering_sweep.py.

Usage:
    steerer = PrefillSteeredModel(
        model_path="/workspace/gpt-oss-20b",
        steering_vector_path="steering_vector.json",
        layer=14,
        alpha=100
    )

    # Generate with prefilled CoT
    result = steerer.generate_with_prefill(
        problem="What is 2+2?",
        prefill="Let me think step by step. First, ",
        reasoning_level="low"
    )

    # Batch generation for Thought Anchors resampling
    results = steerer.generate_batch_with_prefill(
        problem="What is 2+2?",
        prefill="Let me calculate. ",
        n=100,  # 100 resamples
        temperature=0.8
    )

    steerer.close()
"""

from steered_model_base import BaseSteeredModel
from typing import Dict, List, Any


class PrefillSteeredModel(BaseSteeredModel):
    """
    A steered model that applies steering only AFTER a prefilled portion of CoT.

    This is designed for Thought Anchors experiments where you want to:
    1. Prefill the CoT up to sentence K
    2. Apply steering only to the continuation (not the prefill)
    3. Measure how steering affects completion from that point

    Based on the targeted steering approach from steering_sweep.py.
    """

    def _build_prefill_prompt(
        self,
        problem: str,
        prefill: str,
        reasoning_level: str = "low"
    ) -> str:
        """
        Build Harmony prompt with prefill for targeted steering.

        CRITICAL: This follows the exact format from steering_sweep.py lines 24-32.
        The prompt ends mid-analysis channel with NO closing tag.

        Args:
            problem: The question or task
            prefill: The CoT text to prefill (sentences 1 through K)
            reasoning_level: One of "minimal", "low", "medium", "high"

        Returns:
            Harmony-formatted prompt ending with prefill
        """
        prompt = (
            f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
            f"Reasoning: {reasoning_level}\n"
            "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
            f"<|start|>user<|message|>{problem}<|end|>"
            f"<|start|>assistant<|channel|>analysis<|message|>{prefill}"
        )
        # NO CLOSING TAG - forces continuation from this point
        return prompt

    def generate_with_prefill(
        self,
        problem: str,
        prefill: str,
        reasoning_level: str = "low",
        max_tokens: int = 3000,
        temperature: float = 0.8,
        do_sample: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate with prefilled CoT, steering only the continuation.

        This is the key method for Thought Anchors experiments.
        It prefills the CoT up to position K and steers only the continuation.

        CRITICAL: Follows steering_sweep.py lines 175-178 for computing prefill length.

        Args:
            problem: The question or task
            prefill: CoT text to prefill (e.g., sentences 1 through K)
            reasoning_level: One of "minimal", "low", "medium", "high"
            max_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample (False = greedy)
            **kwargs: Additional arguments passed to model.generate()

        Returns:
            Dict with keys:
                - reasoning: Full reasoning (prefill + continuation)
                - continuation: Just the generated continuation
                - final_answer: Parsed final answer (if present)
                - n_tokens: Number of tokens generated (not counting prefill)
                - hit_eos: Whether generation ended with EOS
                - is_truncated: Whether hit max_tokens limit
                - prefill_length_tokens: Number of tokens in prefill

        Example:
            >>> steerer = PrefillSteeredModel(..., alpha=100)
            >>> sentences = ["First sentence.", "Second sentence."]
            >>> prefill = " ".join(sentences)  # Prefill up to sentence 2
            >>> result = steerer.generate_with_prefill(
            ...     problem="What is 2+2?",
            ...     prefill=prefill,
            ...     reasoning_level="low"
            ... )
            >>> print(result['continuation'])  # What the model generated after sentence 2
        """
        # Build prompt with prefill
        prompt = self._build_prefill_prompt(problem, prefill, reasoning_level)

        # CRITICAL: Compute prefill length in TOKENS (not characters!)
        # This follows steering_sweep.py lines 175-178
        prefill_tokens = self.tokenizer(prompt, return_tensors="pt")['input_ids']
        prefill_length = prefill_tokens.shape[1]

        # Set steering to start AFTER the prefill
        self._set_steering_position(prefill_length)

        # Generate
        result = super().generate(prompt, max_tokens, temperature, do_sample, **kwargs)

        # Add prefill-specific information
        result['continuation'] = result['reasoning']  # The generated part
        result['reasoning'] = prefill + " " + result['reasoning']  # Full CoT including prefill
        result['prefill_length_tokens'] = prefill_length

        return result

    def generate_batch_with_prefill(
        self,
        problem: str,
        prefill: str,
        n: int = 100,
        reasoning_level: str = "low",
        max_tokens: int = 3000,
        temperature: float = 0.8,
        do_sample: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate N completions with prefilled CoT (batch version).

        This is optimized for Thought Anchors resampling where you need
        100+ samples from the same prefill point.

        Args:
            problem: The question or task
            prefill: CoT text to prefill (e.g., sentences 1 through K)
            n: Number of completions to generate (default 100 for Thought Anchors)
            reasoning_level: One of "minimal", "low", "medium", "high"
            max_tokens: Maximum new tokens per completion
            temperature: Sampling temperature
            do_sample: Whether to sample (False = greedy)
            **kwargs: Additional arguments passed to model.generate()

        Returns:
            List of dicts, each with keys:
                - reasoning: Full reasoning (prefill + continuation)
                - continuation: Just the generated continuation
                - final_answer: Parsed final answer (if present)
                - n_tokens: Number of tokens generated (not counting prefill)
                - hit_eos: Whether generation ended with EOS
                - is_truncated: Whether hit max_tokens limit
                - prefill_length_tokens: Number of tokens in prefill

        Example for Thought Anchors:
            >>> # Resample from sentence K
            >>> sentences_up_to_k = ["Sent 1.", "Sent 2.", "Sent 3."]
            >>> prefill = " ".join(sentences_up_to_k)
            >>>
            >>> # Generate 100 resamples
            >>> results = steerer.generate_batch_with_prefill(
            ...     problem="What is 2+2?",
            ...     prefill=prefill,
            ...     n=100,
            ...     temperature=0.8
            ... )
            >>>
            >>> # Analyze how often answer is correct
            >>> correct = sum(1 for r in results if "4" in r['final_answer'])
            >>> accuracy = correct / len(results)
        """
        # Build prompt with prefill
        prompt = self._build_prefill_prompt(problem, prefill, reasoning_level)

        # Create batch of identical prompts
        prompts = [prompt] * n

        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False
        )
        attention_mask = inputs['attention_mask'].to(self.device)

        # CRITICAL: Compute prefill length based on actual prompt (not padded)
        # For batch, we use the actual prompt length without padding
        prefill_length = int(attention_mask[0].sum().item())

        # Set steering to start AFTER the prefill
        self._set_steering_position(prefill_length)

        # Generate batch
        results = super().generate_batch(prompt, n, max_tokens, temperature, do_sample, **kwargs)

        # Add prefill-specific information to each result
        for result in results:
            result['continuation'] = result['reasoning']  # The generated part
            result['reasoning'] = prefill + " " + result['reasoning']  # Full CoT including prefill
            result['prefill_length_tokens'] = prefill_length

        return results

    def resample_from_position(
        self,
        problem: str,
        sentences: List[str],
        position: int,
        n: int = 100,
        reasoning_level: str = "low",
        temperature: float = 0.8,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Convenience method for Thought Anchors: resample from sentence position K.

        This handles the common Thought Anchors workflow:
        1. Take sentences 0 through K-1 as prefill
        2. Generate N continuations
        3. Return structured results

        Args:
            problem: The question or task
            sentences: List of sentences from the baseline CoT
            position: Which sentence position to resample from (0-indexed)
            n: Number of resamples (default 100)
            reasoning_level: Reasoning level for Harmony prompt
            temperature: Sampling temperature
            **kwargs: Additional generation arguments

        Returns:
            List of resampling results with continuation analysis

        Example:
            >>> baseline_cot = "First. Second. Third. Fourth."
            >>> sentences = ["First.", "Second.", "Third.", "Fourth."]
            >>>
            >>> # Resample from position 2 (after "Second.")
            >>> results = steerer.resample_from_position(
            ...     problem="What is 2+2?",
            ...     sentences=sentences,
            ...     position=2,
            ...     n=100
            ... )
        """
        # Prefill up to (but not including) the specified position
        if position > 0:
            prefill = " ".join(sentences[:position])
        else:
            prefill = ""

        # Generate batch with this prefill
        return self.generate_batch_with_prefill(
            problem=problem,
            prefill=prefill,
            n=n,
            reasoning_level=reasoning_level,
            temperature=temperature,
            **kwargs
        )
