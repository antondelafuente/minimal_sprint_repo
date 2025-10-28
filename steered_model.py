#!/usr/bin/env python3
"""
Reusable steered model wrapper for activation steering experiments.
This is the original SteeredModel that steers after the full prompt.

Usage:
    steerer = SteeredModel(
        model_path="/workspace/gpt-oss-20b",
        steering_vector_path="path/to/vector.json",
        layer=14,
        alpha=100
    )

    # Generate N times with same prompt
    results = steerer.generate_batch(
        prompt="Your prompt here",
        n=50,
        max_tokens=3000,
        temperature=0.8
    )

    # Change steering strength
    steerer.set_alpha(150)

    # Single generation
    result = steerer.generate(prompt="Another prompt")

    # Clean up when done
    steerer.close()
"""

from steered_model_base import BaseSteeredModel
from typing import Dict, List, Any


class SteeredModel(BaseSteeredModel):
    """
    A language model wrapper that applies activation steering during generation.

    The steering is applied only to generated tokens (not the prompt encoding),
    allowing you to influence the model's reasoning without affecting how it
    interprets the input.

    This is the original behavior: steering starts after the entire prompt.
    """

    def _build_harmony_prompt(
        self,
        problem: str,
        prefill: str = "",
        reasoning_level: str = "low"
    ) -> str:
        """
        Build Harmony format prompt.

        Args:
            problem: The question or task
            prefill: Optional prefill text to start the assistant's response
            reasoning_level: One of "minimal", "low", "medium", "high"

        Returns:
            Harmony-formatted prompt string
        """
        prompt = (
            f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
            f"Reasoning: {reasoning_level}\n"
            "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
            f"<|start|>user<|message|>{problem}<|end|>"
            f"<|start|>assistant<|channel|>analysis<|message|>{prefill}"
        )
        return prompt

    def generate(
        self,
        prompt: str,
        max_tokens: int = 3000,
        temperature: float = 0.8,
        do_sample: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a single completion with steering.

        Args:
            prompt: Input text
            max_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample (False = greedy)
            **kwargs: Additional arguments passed to model.generate()

        Returns:
            Dict with keys:
                - reasoning: Parsed reasoning from analysis channel
                - final_answer: Parsed final answer from final channel (empty if not present)
                - n_tokens: Number of tokens generated
                - hit_eos: Whether generation ended with EOS token
                - is_truncated: Whether generation hit max_tokens limit
        """
        # Tokenize to get prompt length and set steering position
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)

        # Set steering to start after the prompt (original behavior)
        self._set_steering_position(input_ids.shape[1])

        # Call parent generate
        return super().generate(prompt, max_tokens, temperature, do_sample, **kwargs)

    def generate_batch(
        self,
        prompt: str,
        n: int = 50,
        max_tokens: int = 3000,
        temperature: float = 0.8,
        do_sample: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate N completions with steering (efficient batched version).

        Args:
            prompt: Input text (same for all N generations)
            n: Number of completions to generate
            max_tokens: Maximum new tokens per completion
            temperature: Sampling temperature
            do_sample: Whether to sample (False = greedy)
            **kwargs: Additional arguments passed to model.generate()

        Returns:
            List of dicts, each with keys:
                - reasoning: Parsed reasoning from analysis channel
                - final_answer: Parsed final answer from final channel (empty if not present)
                - n_tokens: Number of tokens generated
                - hit_eos: Whether generation ended with EOS token
                - is_truncated: Whether generation hit max_tokens limit
        """
        # Create batch of identical prompts
        prompts = [prompt] * n

        # Tokenize to get actual prompt length (without padding)
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False
        )
        attention_mask = inputs['attention_mask'].to(self.device)

        # Set steering position based on actual prompt length (not padded)
        # For batch, we use the actual prompt length (without padding)
        prompt_length = int(attention_mask[0].sum().item())
        self._set_steering_position(prompt_length)

        # Call parent generate_batch
        return super().generate_batch(prompt, n, max_tokens, temperature, do_sample, **kwargs)

    def ask(
        self,
        question: str,
        reasoning_level: str = "low",
        prefill: str = "",
        max_tokens: int = 3000,
        temperature: float = 0.8,
        **kwargs
    ) -> Dict[str, Any]:
        """
        High-level interface: Ask a question and get a response.

        This abstracts away the Harmony format - you just provide a question
        and get back reasoning and answer.

        Args:
            question: The question or task to ask
            reasoning_level: One of "minimal", "low", "medium", "high" (default: "low")
            prefill: Optional text to prefill the assistant's response (default: "")
            max_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments passed to model.generate()

        Returns:
            Dict with keys:
                - reasoning: The model's reasoning process
                - final_answer: The final answer
                - n_tokens: Number of tokens generated
                - hit_eos: Whether generation ended with EOS
                - is_truncated: Whether generation hit max_tokens limit

        Example:
            >>> steerer = SteeredModel(...)
            >>> result = steerer.ask("What is 2+2?", reasoning_level="low")
            >>> print(result['reasoning'])
            >>> print(result['final_answer'])
        """
        prompt = self._build_harmony_prompt(question, prefill, reasoning_level)
        return self.generate(prompt, max_tokens, temperature, **kwargs)

    def ask_batch(
        self,
        question: str,
        n: int = 50,
        reasoning_level: str = "low",
        prefill: str = "",
        max_tokens: int = 3000,
        temperature: float = 0.8,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        High-level interface: Ask a question N times and get N responses.

        This is the batch version of ask() - useful for collecting statistics
        across multiple samples.

        Args:
            question: The question or task to ask
            n: Number of times to generate (default: 50)
            reasoning_level: One of "minimal", "low", "medium", "high" (default: "low")
            prefill: Optional text to prefill the assistant's response (default: "")
            max_tokens: Maximum new tokens per generation
            temperature: Sampling temperature
            **kwargs: Additional arguments passed to model.generate()

        Returns:
            List of dicts, each with keys:
                - reasoning: The model's reasoning process
                - final_answer: The final answer
                - n_tokens: Number of tokens generated
                - hit_eos: Whether generation ended with EOS
                - is_truncated: Whether generation hit max_tokens limit

        Example:
            >>> steerer = SteeredModel(..., alpha=100)
            >>> results = steerer.ask_batch("What is 2+2?", n=50)
            >>> cheat_rate = sum(1 for r in results if "cheat" in r['reasoning']) / len(results)
        """
        prompt = self._build_harmony_prompt(question, prefill, reasoning_level)
        return self.generate_batch(prompt, n, max_tokens, temperature, **kwargs)