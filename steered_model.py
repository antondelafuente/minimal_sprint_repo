#!/usr/bin/env python3
"""
Unified steered model for activation steering with prefill support.
Supports both regular steering and prefill-based steering for Thought Anchors experiments.

Usage:
    # Initialize
    steerer = SteeredModel(
        model_path="/workspace/gpt-oss-20b",
        steering_vector_path="steering_vectors/steering_vectors_normalized.pt",
        layer=None,  # Multi-layer steering
        alpha=100
    )

    # Regular generation (steering after full prompt)
    result = steerer.generate_with_prefill(
        problem="What is 2+2?",
        prefill="",  # Empty prefill = regular steering
        reasoning_level="low"
    )

    # Prefill-based generation (for Thought Anchors)
    result = steerer.generate_with_prefill(
        problem="What is 2+2?",
        prefill="Let me think step by step. First,",
        reasoning_level="low"
    )

    # High-level interface
    result = steerer.ask("What is 2+2?")  # Simple ask
    results = steerer.ask_batch("What is 2+2?", n=100)  # Batch

    steerer.close()
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union, List, Dict, Any


class SteeredModel:
    """
    Language model with activation steering support.

    Supports both regular steering (applied after the full prompt) and
    prefill-based steering (applied after a specific prefill point) for
    Thought Anchors experiments.
    """

    def __init__(
        self,
        model_path: str,
        steering_vector_path: str,
        layer: int = None,
        alpha: float = 0.0,
        normalize_vectors: bool = True
    ):
        """
        Initialize steered model.

        Args:
            model_path: Path to the model directory
            steering_vector_path: Path to .json (single vector) or .pt (multi-layer) file
            layer: Which layer to apply steering to. If None and using .pt file, applies to ALL layers
            alpha: Steering strength (can be changed later with set_alpha())
            normalize_vectors: If True, normalize vectors to unit length (recommended for consistent alpha scaling)
        """
        print("Initializing SteeredModel...")

        self.model_path = model_path
        self.layer_idx = layer
        self.alpha = alpha
        self.steering_position = 0  # Where steering starts (in tokens)
        self.normalize_vectors = normalize_vectors
        self.multi_layer_mode = False
        self.steering_vectors = {}  # Dict mapping layer_idx -> vector

        # Load steering vector(s)
        print(f"Loading steering vector from {steering_vector_path}...")

        if steering_vector_path.endswith('.pt'):
            # Load multi-layer PyTorch tensor file
            steering_data = torch.load(steering_vector_path, map_location='cpu')

            if isinstance(steering_data, torch.Tensor):
                if len(steering_data.shape) == 2:
                    # Shape: [num_layers, hidden_dim]
                    num_layers, hidden_dim = steering_data.shape
                    print(f"  Loaded multi-layer steering vectors: {num_layers} layers × {hidden_dim} dims")

                    if layer is None:
                        # Use all layers
                        self.multi_layer_mode = True

                        # Normalize entire tensor as one vector to preserve relative layer magnitudes
                        if normalize_vectors:
                            total_norm = steering_data.norm()  # Frobenius norm
                            steering_data = steering_data / total_norm
                            print(f"  Normalized entire tensor (Frobenius norm = 1.0)")
                            print(f"  This preserves relative magnitudes between layers")

                        for layer_idx in range(num_layers):
                            vec = steering_data[layer_idx].float()
                            self.steering_vectors[layer_idx] = vec.tolist()
                        print(f"  Multi-layer mode: steering on ALL {num_layers} layers")
                    else:
                        # Use single specified layer
                        if layer >= num_layers:
                            raise ValueError(f"Requested layer {layer} but only {num_layers} layers available")
                        vec = steering_data[layer].float()
                        if normalize_vectors:
                            vec = vec / vec.norm()
                        self.steering_vectors[layer] = vec.tolist()
                        print(f"  Single-layer mode: steering on layer {layer}")
                elif len(steering_data.shape) == 1:
                    # Single vector
                    if layer is None:
                        raise ValueError("Single vector provided but no layer specified")
                    vec = steering_data.float()
                    if normalize_vectors:
                        vec = vec / vec.norm()
                    self.steering_vectors[layer] = vec.tolist()
                    print(f"  Single vector: {len(vec)} dimensions, applied to layer {layer}")
            else:
                raise ValueError(f"Expected tensor in .pt file, got {type(steering_data)}")
        else:
            # Load JSON file (original format)
            with open(steering_vector_path) as f:
                steering_data = json.load(f)

            # Handle different JSON formats
            if 'universal_delta_direction' in steering_data:
                steering_vector = steering_data['universal_delta_direction']
            elif 'steering_vector' in steering_data:
                steering_vector = steering_data['steering_vector']
            else:
                steering_vector = steering_data

            if layer is None:
                raise ValueError("JSON format requires specifying a layer")

            if normalize_vectors:
                import numpy as np
                vec_array = np.array(steering_vector)
                vec_array = vec_array / np.linalg.norm(vec_array)
                steering_vector = vec_array.tolist()

            self.steering_vectors[layer] = steering_vector
            print(f"  Steering vector: {len(steering_vector)} dimensions, applied to layer {layer}")
            if normalize_vectors:
                print(f"  Vector normalized to unit length")

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model.eval()

        self.device = next(self.model.parameters()).device
        print(f"  Model loaded on {self.device}")
        print(f"  GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

        # EOS tokens for this model
        self.eos_tokens = [199999, 200002, 200012]

        # Register steering hooks
        self._steering_vector_tensors = {}  # Lazy init on first forward pass
        self.hook_handles = []

        if self.multi_layer_mode:
            # Register hooks on all layers
            for layer_idx in self.steering_vectors.keys():
                target_layer = self.model.model.layers[layer_idx]
                hook = self._make_steering_hook(layer_idx)
                handle = target_layer.register_forward_hook(hook)
                self.hook_handles.append(handle)
            print(f"  Registered steering hooks on {len(self.hook_handles)} layers")
        else:
            # Single layer mode
            layer_idx = list(self.steering_vectors.keys())[0]
            self.layer_idx = layer_idx
            target_layer = self.model.model.layers[layer_idx]
            hook = self._make_steering_hook(layer_idx)
            handle = target_layer.register_forward_hook(hook)
            self.hook_handles.append(handle)
            print(f"  Registered steering hook on layer {layer_idx}")

        print(f"  Initial alpha: {self.alpha}")

    def _set_steering_position(self, position_in_tokens: int):
        """
        Set where steering starts in the token sequence.

        Args:
            position_in_tokens: Token position where steering should begin
        """
        self.steering_position = position_in_tokens

    def _make_steering_hook(self, layer_idx):
        """
        Create the steering hook function for a specific layer.

        This hook only steers tokens AFTER steering_position.
        Handles both prefill (full sequence) and incremental decode (single token).

        Args:
            layer_idx: Which layer this hook is for (to get the right steering vector)
        """
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # hidden_states shape: [batch, seq_len, hidden_dim]
            B, T, D = hidden_states.shape

            # Initialize steering vector tensor once per layer
            if layer_idx not in self._steering_vector_tensors:
                self._steering_vector_tensors[layer_idx] = torch.tensor(
                    self.steering_vectors[layer_idx],
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                ).view(1, 1, D)

            sv = self._steering_vector_tensors[layer_idx]

            # Critical gating logic:
            # If we're encoding the full sequence (T >= steering_position), gate to position+
            if T >= self.steering_position:
                # Encoding full sequence: steer positions >= steering_position
                hidden_states[:, self.steering_position:, :] = (
                    hidden_states[:, self.steering_position:, :] + self.alpha * sv
                )
            else:
                # Incremental decoding (T==1): this is a post-steering token, steer it
                hidden_states = hidden_states + self.alpha * sv

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states

        return hook

    def set_alpha(self, new_alpha: float):
        """Change the steering strength."""
        self.alpha = new_alpha
        print(f"Alpha updated to {self.alpha}")

    def _parse_harmony_output(self, text: str) -> Dict[str, str]:
        """
        Parse Harmony format output into reasoning and final answer.

        Args:
            text: Full generated text in Harmony format

        Returns:
            Dict with keys:
                - reasoning: Text from analysis channel
                - final_answer: Text from final channel (if present)
        """
        # Look for final channel separator
        final_marker = '<|channel|>final<|message|>'

        if final_marker in text:
            idx = text.find(final_marker)
            reasoning = text[:idx]

            # Clean up reasoning: remove channel switching tags
            for tag in ['<|end|><|start|>assistant', '<|end|>', '<|start|>assistant']:
                reasoning = reasoning.replace(tag, '')
            reasoning = reasoning.strip()

            # Extract text after marker, up to end tags
            final_part = text[idx + len(final_marker):]
            # Remove trailing tags like <|return|> or <|end|>
            for end_tag in ['<|return|>', '<|end|>']:
                if end_tag in final_part:
                    final_part = final_part[:final_part.find(end_tag)]
            final_answer = final_part.strip()
        else:
            # No final channel found - everything is reasoning
            reasoning = text.strip()
            final_answer = ""

        return {
            'reasoning': reasoning,
            'final_answer': final_answer
        }

    def _build_harmony_prompt(
        self,
        problem: str,
        prefill: str = "",
        reasoning_level: str = "low"
    ) -> str:
        """
        Build Harmony prompt with optional prefill.

        Args:
            problem: The question or task
            prefill: Optional CoT text to prefill (empty string for no prefill)
            reasoning_level: One of "minimal", "low", "medium", "high"

        Returns:
            Harmony-formatted prompt ending with prefill (or empty)
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

    def _generate_core(
        self,
        prompt: str,
        max_tokens: int = 3000,
        temperature: float = 0.8,
        do_sample: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Core generation logic (single completion).

        Args:
            prompt: Formatted prompt
            max_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample (False = greedy)
            **kwargs: Additional arguments passed to model.generate()

        Returns:
            Dict with generation results
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_tokens,
                **kwargs
            )

        # Decode and analyze
        output_ids = outputs[0]
        prompt_len = input_ids.shape[1]
        generated_ids = output_ids[prompt_len:].cpu().tolist()

        # Check for EOS
        hit_eos = False
        for idx, tok_id in enumerate(generated_ids):
            if tok_id in self.eos_tokens:
                generated_ids = generated_ids[:idx+1]
                hit_eos = True
                break

        is_truncated = not hit_eos and len(generated_ids) >= max_tokens

        # Decode
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Parse Harmony format
        parsed = self._parse_harmony_output(text)

        return {
            'reasoning': parsed['reasoning'],
            'final_answer': parsed['final_answer'],
            'n_tokens': len(generated_ids),
            'hit_eos': hit_eos,
            'is_truncated': is_truncated
        }

    def _generate_batch_core(
        self,
        prompt: str,
        n: int = 50,
        max_tokens: int = 3000,
        temperature: float = 0.8,
        do_sample: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Core batch generation logic.

        Args:
            prompt: Formatted prompt (same for all N generations)
            n: Number of completions to generate
            max_tokens: Maximum new tokens per completion
            temperature: Sampling temperature
            do_sample: Whether to sample (False = greedy)
            **kwargs: Additional arguments passed to model.generate()

        Returns:
            List of dicts with generation results
        """
        # Create batch of identical prompts
        prompts = [prompt] * n

        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_tokens,
                **kwargs
            )

        # Decode and analyze each completion
        results = []
        for i, output_ids in enumerate(outputs):
            # Find actual prompt length for this example (excluding padding)
            prompt_len = int(attention_mask[i].sum().item())
            generated_ids = output_ids[prompt_len:].cpu().tolist()

            # Check for EOS
            hit_eos = False
            for idx, tok_id in enumerate(generated_ids):
                if tok_id in self.eos_tokens:
                    generated_ids = generated_ids[:idx+1]
                    hit_eos = True
                    break

            is_truncated = not hit_eos and len(generated_ids) >= max_tokens

            # Decode
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

            # Parse Harmony format
            parsed = self._parse_harmony_output(text)

            results.append({
                'reasoning': parsed['reasoning'],
                'final_answer': parsed['final_answer'],
                'n_tokens': len(generated_ids),
                'hit_eos': hit_eos,
                'is_truncated': is_truncated
            })

        return results

    def generate_with_prefill(
        self,
        problem: str,
        prefill: str = "",
        reasoning_level: str = "low",
        max_tokens: int = 3000,
        temperature: float = 0.8,
        do_sample: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate with optional prefilled CoT.

        When prefill is empty, this acts like regular steering (applied after the full prompt).
        When prefill is provided, steering is applied only after the prefill point (Thought Anchors).

        Args:
            problem: The question or task
            prefill: CoT text to prefill (empty string for regular steering)
            reasoning_level: One of "minimal", "low", "medium", "high"
            max_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample (False = greedy)
            **kwargs: Additional arguments passed to model.generate()

        Returns:
            Dict with keys:
                - reasoning: Full reasoning (prefill + continuation if prefilled)
                - continuation: Just the generated continuation
                - final_answer: Parsed final answer (if present)
                - n_tokens: Number of tokens generated (not counting prefill)
                - hit_eos: Whether generation ended with EOS
                - is_truncated: Whether hit max_tokens limit
                - prefill_length_tokens: Number of tokens in prefill (if applicable)
        """
        # Build prompt with prefill
        prompt = self._build_harmony_prompt(problem, prefill, reasoning_level)

        # Compute where steering should start
        prefill_tokens = self.tokenizer(prompt, return_tensors="pt")['input_ids']
        prefill_length = prefill_tokens.shape[1]

        # Set steering to start after the full prompt (including any prefill)
        self._set_steering_position(prefill_length)

        # Generate
        result = self._generate_core(prompt, max_tokens, temperature, do_sample, **kwargs)

        # Add prefill-specific information
        result['continuation'] = result['reasoning']  # The generated part

        # Only add prefill to reasoning if there is a prefill
        if prefill:
            result['reasoning'] = prefill + " " + result['reasoning']
            result['prefill_length_tokens'] = prefill_length

        return result

    def generate_batch_with_prefill(
        self,
        problem: str,
        prefill: str = "",
        n: int = 100,
        reasoning_level: str = "low",
        max_tokens: int = 3000,
        temperature: float = 0.8,
        do_sample: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate N completions with optional prefilled CoT (batch version).

        Optimized for Thought Anchors resampling where you need many samples from the same prefill point.

        Args:
            problem: The question or task
            prefill: CoT text to prefill (empty string for regular steering)
            n: Number of completions to generate
            reasoning_level: One of "minimal", "low", "medium", "high"
            max_tokens: Maximum new tokens per completion
            temperature: Sampling temperature
            do_sample: Whether to sample (False = greedy)
            **kwargs: Additional arguments passed to model.generate()

        Returns:
            List of dicts, each with keys:
                - reasoning: Full reasoning (prefill + continuation if prefilled)
                - continuation: Just the generated continuation
                - final_answer: Parsed final answer (if present)
                - n_tokens: Number of tokens generated (not counting prefill)
                - hit_eos: Whether generation ended with EOS
                - is_truncated: Whether hit max_tokens limit
                - prefill_length_tokens: Number of tokens in prefill (if applicable)
        """
        # Build prompt with prefill
        prompt = self._build_harmony_prompt(problem, prefill, reasoning_level)

        # Create batch of identical prompts
        prompts = [prompt] * n

        # Tokenize batch to get accurate prefill length
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False
        )
        attention_mask = inputs['attention_mask'].to(self.device)

        # Compute prefill length based on actual prompt (not padded)
        prefill_length = int(attention_mask[0].sum().item())

        # Set steering to start after the prefill
        self._set_steering_position(prefill_length)

        # Generate batch
        results = self._generate_batch_core(prompt, n, max_tokens, temperature, do_sample, **kwargs)

        # Add prefill-specific information to each result
        for result in results:
            result['continuation'] = result['reasoning']  # The generated part

            # Only add prefill to reasoning if there is a prefill
            if prefill:
                result['reasoning'] = prefill + " " + result['reasoning']
                result['prefill_length_tokens'] = prefill_length

        return results

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
            Dict with generation results

        Example:
            >>> steerer = SteeredModel(...)
            >>> result = steerer.ask("What is 2+2?")
            >>> print(result['reasoning'])
            >>> print(result['final_answer'])
        """
        return self.generate_with_prefill(
            problem=question,
            prefill=prefill,
            reasoning_level=reasoning_level,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

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
            List of dicts with generation results

        Example:
            >>> steerer = SteeredModel(..., alpha=100)
            >>> results = steerer.ask_batch("What is 2+2?", n=100)
            >>> accuracy = sum(1 for r in results if "4" in r['final_answer']) / len(results)
        """
        return self.generate_batch_with_prefill(
            problem=question,
            prefill=prefill,
            n=n,
            reasoning_level=reasoning_level,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

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
            >>> sentences = ["First.", "Second.", "Third.", "Fourth."]
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

    def close(self):
        """Remove steering hooks and clean up."""
        if hasattr(self, 'hook_handles'):
            for handle in self.hook_handles:
                handle.remove()
            print(f"Removed {len(self.hook_handles)} steering hook(s)")

    def __del__(self):
        """Cleanup on deletion."""
        self.close()