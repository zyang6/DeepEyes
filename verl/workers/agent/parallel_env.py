import torch
import deepcopy
import numpy as np
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length

def agent_rollout_loop(config, vllm_engine, vllm_inputs, prompts, sampling_params):
    agent_sampling_params = deepcopy.copy(sampling_params)
    agent_sampling_params.detokenize = True
    agent_sampling_params.skip_special_tokens = False
    agent_sampling_params.n = 1

    env = ParallelEnv(config)
    env.reset(vllm_inputs)

    batch_size = len(vllm_inputs)
    vllm_input_list = []
    running_states = []
    running_action_masks = []
    reward_tensor_list = []
    active_mask = []

    # interleaving inputs if sampling_params.n > 1
    for i in range(batch_size):
        for _ in range(sampling_params.n):
            vllm_input_list.append(vllm_inputs[i])
            prompt_ids = prompts.batch['input_ids'][i, :]
            running_states.append(prompt_ids)
            prompt_mask = prompts.batch['attention_mask'][i, :]
            running_action_masks.append(prompt_mask)
            reward_tensor = torch.zeros_like(prompt_ids, dtype=torch.float)
            reward_tensor_list.append(reward_tensor)
            active_mask.append(True)

    for step in range(config.agent.max_turns):
        if sum(active_mask) == 0:
            break

        active_indices = [idx for idx, is_active in enumerate(active_mask) if is_active]
        active_vllm_input = [vinput for vinput, is_active in zip(vllm_input_list, active_mask) if is_active]
        actions = vllm_engine.generate(
            prompts=active_vllm_input,
            sampling_params=agent_sampling_params,
            use_tqdm=False
        )
        observations, rewards, dones, info = env.step(actions)
        obs_token_ids = vllm_engine.tokenizer.batch_encode_plus(observations['prompt_token_ids'])['input_ids']
        # TODO: support mm input in obs!!!
        multi_modal_data = observations['multi_modal_data']

        for idx, obs, mm, act, rew, done in zip(active_indices, obs_token_ids, multi_modal_data, actions, rewards, done):
            # process response token ids
            response_token_ids = torch.tensor(act.outputs[0].token_ids, dtype=torch.int64)
            running_states[idx] = torch.cat([running_states[idx], response_token_ids])
            if isinstance(vllm_input_list[idx]['prompt_token_ids'], torch.Tensor):
                vllm_input_list[idx]['prompt_token_ids'] = torch.cat([
                    vllm_input_list[idx]['prompt_token_ids'], 
                    response_token_ids.unsqueeze(0)
                ], dim=-1)
            else:
                vllm_input_list[idx]['prompt_token_ids'] = np.concatenate([
                    vllm_input_list[idx]['prompt_token_ids'],
                    response_token_ids.unsqueeze(0).numpy()
                ], axis=-1)

            action_reward = torch.zeros_like(response_token_ids, dtype=torch.float)
            reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], action_reward])
            reward_tensor_list[idx][-1] += rew

            action_mask = torch.ones_like(response_token_ids, dtype=torch.int64)
            running_action_masks[idx] = torch.cat([running_action_masks[idx], action_mask])

            # NOTE: obs can be very long
            obs = obs[:config.agent.single_obs_max_length]
            running_states[idx] = torch.cat([running_states[idx], obs])

            obs_reward = torch.zeros(len(obs), dtype=torch.float)
            reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], obs_reward], dim=-1)

            obs_mask = torch.zeros(len(obs), dtype=torch.int64)
            running_action_masks[idx] = torch.cat([running_action_masks[idx], obs_mask])
            if done or running_states[idx].shape[-1] >= config.response_length:
                active_mask[idx] = False

    max_total_length = config.prompt_length + config.response_length
    target_device = prompts.batch['input_ids'].device
    running_states = [state[: max_total_length] for state in running_states]
    state_tensor = pad_2d_list_to_length(running_states, engine.tokenizer.pad_token_id, max_total_length).to(target_device)
    running_action_masks = [mask[: max_total_length] for mask in running_action_masks]
    action_mask_tensor = pad_2d_list_to_length(running_action_masks, 0, max_total_length).to(target_device)
    reward_tensor_list = [reward[: max_total_length] for reward in reward_tensor_list]
    reward_tensor = pad_2d_list_to_length(reward_tensor_list, 0.0, max_total_length).to(target_device)
    return state_tensor[:, -config.response_length:], action_mask_tensor, reward_tensor


class ParallelEnv:
    """
    The interface intentionally designed to be the same as: https://github.com/openai/gym
    Hope this could be easier to use for RLers.
    """
    def __init__(self, env_config, **kwargs):
        self.config = env_config

    def step(self, actions):
        """
        Input:
        - actions: vllm.RequestOutput

        Output:
        - observations: {"prompt_token_ids": ..., "multi_modal_data": ...}, 
                multi_modal_data only appears when there are images/videos in obs
        - rewards: List[ float ].
                each time after an action being executed, procedure rewards can be assigned to 
                the last valid token of model outputs. This might be useful for ..., 
                e.g., invalid action, code execution error, format error,
                or video game envs where immediate feedback is available.
        - dones: List[ Boolean ]
        - infos: Dict, for debugging only
        """
        dones = [act.outputs[0].finish_reason == 'length' for act in actions]
        valid_indices = [idx for idx, done in enumerate(dones) if not done]
        valid_actions = [act.outputs[0].text for act, done in zip(actions, dones) if not done]


    def reset(self, prompts):
        pass


