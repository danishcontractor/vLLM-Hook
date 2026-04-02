"""
SpotlightLLM: HookLLM subclass for Spotlight attention steering.

Extends HookLLM with Spotlight-specific generate() that handles span tokenization,
direct parameter passing to worker hooks, and automatic cleanup.
"""
import os
import json
from typing import List, Optional, Union
from vllm import SamplingParams

from vllm_hook_plugins.hook_llm import HookLLM
from vllm_hook_plugins.workers.spotlight_utils import get_span_ranges


class SpotlightLLM(HookLLM):
    """
    HookLLM subclass for Spotlight inference-time attention steering.

    Adds generate_with_spotlight() method that handles:
    - Text span tokenization
    - Direct parameter passing to worker (via instance state)
    - Automatic cleanup

    Key design: Parameters are stored on the SpotlightLLM instance,
    not in globals or env vars. The worker accesses them directly.

    Usage:
        llm = SpotlightLLM(
            model="Qwen/Qwen2-1.5B-Instruct",
            worker_name="probe_spotlight",
            enforce_eager=True,
        )

        outputs = llm.generate_with_spotlight(
            prompts=["Answer in JSON format. What is 2+2?"],
            emph_strings=["Answer in JSON format"],
            alpha=0.4,
            max_tokens=50,
            temperature=0.7,
        )
    """

    def generate_with_spotlight(
        self,
        prompts: Union[str, List[str]],
        emph_strings: Union[str, List[str], List[List[str]]],
        alpha: float = 0.2,
        sampling_params: Optional[SamplingParams] = None,
        **kwargs
    ):
        """
        Generate with Spotlight attention steering toward emphasized spans.

        Args:
            prompts: Input prompt(s)
            emph_strings: Text span(s) to emphasize:
                - Single string: applied to all prompts
                - List of strings: applied to all prompts
                - List of lists: one list per prompt
            alpha: Target attention proportion (0.0-1.0)
            sampling_params: vLLM SamplingParams
            **kwargs: Additional vLLM generate kwargs (max_tokens, temperature, etc.)

        Returns:
            vLLM RequestOutput objects
        """
        # Normalize inputs
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(emph_strings, str):
            emph_strings = [[emph_strings]] * len(prompts)
        elif isinstance(emph_strings[0], str):
            emph_strings = [emph_strings] * len(prompts)

        params_file = None
        try:
            # Tokenize prompts with offset mappings
            tokenized = self.tokenizer(
                prompts,
                return_tensors="pt",
                return_offsets_mapping=True,
                padding=True,
            )
            offset_mappings = tokenized.pop("offset_mapping")

            # Convert text spans to token ranges
            span_ranges = get_span_ranges(prompts, emph_strings, offset_mappings)

            # Write parameters to file for worker to read (cross-process safe)
            params_file = os.path.join(self._hook_dir, "spotlight_params.json")
            with open(params_file, 'w') as f:
                json.dump({
                    "span_ranges": span_ranges,
                    "alpha": alpha,
                }, f)

            # Generate with hooks enabled
            result = self.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                use_hook=True,
                **kwargs
            )
            return result

        finally:
            # Clean up parameters file after generation
            try:
                if params_file and os.path.exists(params_file):
                    os.remove(params_file)
            except (OSError, NameError, TypeError):
                pass

    def generate(self, prompts, sampling_params=None, use_hook=None, cleanup=True, **kwargs):
        """Override generate to use Spotlight worker by default."""
        # If no use_hook specified, use the instance's enable_hook setting
        hook = use_hook if use_hook is not None else self.enable_hook

        if not isinstance(prompts, list):
            prompts = [prompts]

        if hook:
            if "probe" in self.worker_name:
                return self.generate_with_encode_hook(
                    prompts, sampling_params, cleanup, **kwargs
                )
            elif "steer" in self.worker_name:
                return self.generate_with_decode_hook(
                    prompts, sampling_params, cleanup, **kwargs
                )

        if sampling_params is None:
            sampling_params = SamplingParams(**kwargs)
        return self.llm.generate(prompts, sampling_params)

    def generate_with_encode_hook(self, prompts, sampling_params, cleanup, **kwargs):
        """
        Generation with Spotlight hooks active throughout.

        Unlike the probe worker (which captures Q/K during prefill then generates
        separately), Spotlight must keep hooks active during the actual generation
        because:
        - The hook modifies attention *output* (hidden states), not the KV cache
        - The KV cache stores K/V values, not attention weights
        - A two-pass approach (prefill with hooks, generate without) has no effect
          since the second pass re-encodes from scratch

        So we run a single generate() call with hooks active the entire time.
        """
        self._setup_hooks(cleanup)

        try:
            if sampling_params is None:
                sampling_params = SamplingParams(**kwargs)
            return self.llm.generate(prompts, sampling_params)
        finally:
            self._cleanup_hooks()
