#!/usr/bin/env python3
"""
Base class for steered model wrappers with activation steering.
Contains all shared logic for model loading, vector management, and generation.
Subclasses only need to implement where steering starts and prompt construction.
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union, List, Dict, Any


class BaseSteeredModel:
    """
    Base class for language models with activation steering.

    Subclasses should:
    1. Call _set_steering_position() before generation to specify where steering starts
    2. Implement their own prompt construction methods
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
        Initialize steered model base.

        Args:
            model_path: Path to the model directory
            steering_vector_path: Path to .json (single vector) or .pt (multi-layer) file
            layer: Which layer to apply steering to. If None and using .pt file, applies to ALL layers
            alpha: Steering strength (can be changed later with set_alpha())
            normalize_vectors: If True, normalize vectors to unit length (recommended for consistent alpha scaling)
        """
        print("Initializing BaseSteeredModel...")

        self.model_path = model_path
        self.layer_idx = layer
        self.alpha = alpha
        self.steering_position = 0  # Where steering starts (in tokens) - set by subclass
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
        Called by subclasses before generation.

        Args:
            position_in_tokens: Token position where steering should begin
        """
        self.steering_position = position_in_tokens

    def _make_steering_hook(self, layer_idx):
        """
        Create the steering hook function for a specific layer.

        This hook only steers tokens AFTER steering_position.
        Handles both prefill (full sequence) and incremental decode (single token).

        Based on steering_sweep.py lines 55-85 for precision.

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

            # Critical gating logic from steering_sweep.py:
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

        NOTE: Subclasses should set steering_position before calling this!

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

        NOTE: Subclasses should set steering_position before calling this!

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

    def close(self):
        """Remove steering hooks and clean up."""
        if hasattr(self, 'hook_handles'):
            for handle in self.hook_handles:
                handle.remove()
            print(f"Removed {len(self.hook_handles)} steering hook(s)")

    def __del__(self):
        """Cleanup on deletion."""
        self.close()