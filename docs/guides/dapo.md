# An in-depth Walkthrough of DAPO in NeMo RL

This guide covers the [Decoupled Clip and Dynamic Sampling for Adaptive Policy Optimization (DAPO)](https://arxiv.org/pdf/2503.14476) implementation in NeMo RL.

DAPO introduces 4 key improvements over GRPO:
1. **Clip-Higher**, which promotes the diversity of the system and avoids entropy collapse
2. **Dynamic Sampling**, which improves training efficiency and stability
3. **Token-Level Policy Gradient Loss**, which is critical in long-CoT RL scenarios
4. **Overlong Reward Shaping**, which reduces reward noise and stabilizes training

This document focuses on DAPO-specific features: Dynamic Sampling and Overlong Reward Shaping. For foundational concepts on GRPO including data handling, policy training, generation, and loss functions, see the [GRPO guide](grpo.md).


## Quickstart: Launch a DAPO Run

To get started quickly, use the example configuration [examples/configs/recipes/llm/dapo-qwen2.5-7b.yaml](../../examples/configs/recipes/llm/dapo-qwen2.5-7b.yaml). You can launch this using the same script as GRPO:

```bash
uv run examples/run_grpo_math.py --config examples/configs/recipes/llm/dapo-qwen2.5-7b.yaml {overrides}
```

**Reminder**: Don't forget to set your HF_HOME, WANDB_API_KEY, and HF_DATASETS_CACHE (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.

## Dynamic Sampling

Standard GRPO trains on all generated responses, even when they have identical rewards (zero gradient signal) within a prompt group of generations. Dynamic sampling filters to keep only groups with diverse rewards (`std > 0`), and accumulates them across batches until reaching the target batch size. 

**Algorithm**: For each training step:

1. Sample `dapo_batch_multiplier × num_prompts_per_step` prompts from the dataset. The default value of `dapo_batch_multiplier` is 3.
2. Generate `num_generations_per_prompt` responses per prompt and compute rewards
3. Calculate baseline and std for each prompt group
4. Filter prompt groups where `std > 0`
5. Accumulate in cache until reaching the training batch size = `num_prompts_per_step × num_generations_per_prompt` samples is reached.
6. You can accumulate until `max_num_gen_batches` number of batches is reached.
7. Perform training on the collected samples with non-zero standard deviation


### Configuration

```yaml
grpo:
  use_dynamic_sampling: true  # Enable DAPO dynamic sampling
  num_prompts_per_step: 512   # Target number of prompts per training step
  num_generations_per_prompt: 16  # Generations per prompt
  dapo_batch_multiplier: 3    # Dataloader batch size = dapo_batch_multiplier × num_prompts_per_step
  max_num_gen_batches: 10     # Maximum number of batches to be used for accumulating non-zero std prompts
  reward_scaling:
    enabled: true
    correct: 1.0
    incorrect: -1.0
  
  reward_shaping:
    enabled: true
    overlong_buffer_length: 4096     # Threshold before penalties apply (paper uses 4096)
    overlong_buffer_penalty: 1.0     # Penalty per excess token
    max_response_length: 20480       # Hard maximum generation length
```

**Key Parameters:**
- **`dapo_batch_multiplier`**: Factor that scales the initial prompt pool size for sampling
- **`max_num_gen_batches`**: Maximum number of batches to be used for accumulating non-zero std prompts
- **`reward_scaling`**: When enabled, maps binary rewards (1.0 for correct, 0.0 for incorrect) to configured values before applying reward shaping
- **`reward_shaping`**: When enabled, applies penalties for responses exceeding max_response_length to reduce reward noise and stabilize training. For more details on the overlong reward shaping mechanism, please refer to Section 3.4 of the [DAPO paper](https://arxiv.org/pdf/2503.14476).

> **Note**: When dynamic sampling is enabled, monitor the `filtered_reward` metric to track the average reward of the prompts with std > 0.


## References

- **DAPO Paper**: [Dynamic Sampling of Group Relative Policy Optimization (2025)](https://arxiv.org/pdf/2503.14476)
- **GRPO Paper**: [Group Relative Policy Optimization (2024)](https://arxiv.org/abs/2402.03300)
- **NeMo RL GRPO Guide**: [grpo.md](grpo.md)