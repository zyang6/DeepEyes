import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List

def build_parallel_envs(agent_config, tokenizer):
    pass


def agent_rollout_loop(engine, env, vllm_inputs, sampling_params, **kwargs):
    running_prompts = prompts
    running_masks = zeros_like(running_prompts)

    for step in range(env.config.max_turns):
        # vllm generation
        actions = self.inference_engine.generate(
            prompts=vllm_inputs,  # because we have already convert it to prompt token id
            sampling_params=sampling_params,
            use_tqdm=False
        )

        response_tokens_list = []
        for output in actions:
            for sample_id in range(len(output.outputs)):
                response_tokens_list.append(output.outputs[sample_id].token_ids)

        response_action_list = env.tokenizer.batch_decode(response_tokens_list)
        


class ParallelEnv:
    def __init__(self, env_config, tokenizer, **kwargs):
        self.config = env_config
        self.tokenizer = tokenizer

    def step(self, action):
        """
        The interface intentionally designed to be the same as: https://github.com/openai/gym
        Hope this could be easier to use for RLers.

        Usage example:
        ```python
        action = model.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        ```

        Input:
        - action: vllm.RequestOutput

        Output:
        - next_state: torch.Tensor, shape [ bs, seq_length ]
        - reward: torch.Tensor with shape [ bs, seq_length ].
                  each time after an action being executed, procedure rewards can be assigned to 
                  the last valid token of model outputs. This might be useful for ..., 
                  e.g., invalid action, code execution error, format error,
                  or video game envs where immediate feedback is available.
        - terminated: boolean
        - truncated: boolean
        - info: Dict, for debugging only
        """
        pass

    def reset(self):
        pass

