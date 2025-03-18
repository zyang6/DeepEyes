import torch
import deepcopy
import numpy as np
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.agent.tool_envs import ToolBase

def _extract_tool_call_contents(start_token, end_token, text):
    # pattern = r"<tool_call>(.*?)</tool_call>"
    pattern = re.escape(start_token) + r'(.*?)' + re.escape(end_token)
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def _concat_vllm_input(prompt_token_ids, response_token_ids):
    if isinstance(prompt_token_ids, torch.Tensor):
        return torch.cat([
            prompt_token_ids,
            response_token_ids.unsqueeze(0)
        ], dim=-1)
    else:
        return np.concatenate([
            prompt_token_ids,
            response_token_ids.unsqueeze(0).numpy()
        ], axis=-1)

def agent_rollout_loop(config, vllm_engine, vllm_inputs, prompts, sampling_params):
    agent_sampling_params = sampling_params.copy()
    agent_sampling_params.detokenize = True
    agent_sampling_params.skip_special_tokens = False
    agent_sampling_params.spaces_between_special_tokens = False
    agent_sampling_params.n = 1

    env = ParallelEnv(config.agent, engine.tokenizer)
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
        obs_token_ids = [obs['prompt_token_ids'] for obs in observations]
        # TODO: support mm input in obs!!!
        multi_modal_data = [obs['multi_modal_data'] for obs in observations]

        for idx, obs, mm, act, rew, done in zip(active_indices, obs_token_ids, multi_modal_data, actions, rewards, dones):
            if done:
                active_mask[idx] = False
                continue

            # process response token ids
            response_token_ids = torch.tensor(act.outputs[0].token_ids, dtype=torch.int64)
            running_states[idx] = torch.cat([running_states[idx], response_token_ids])
            vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(vllm_input_list[idx]['prompt_token_ids'], response_token_ids)

            action_reward = torch.zeros_like(response_token_ids, dtype=torch.float)
            reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], action_reward])
            reward_tensor_list[idx][-1] += rew

            action_mask = torch.ones_like(response_token_ids, dtype=torch.int64)
            running_action_masks[idx] = torch.cat([running_action_masks[idx], action_mask])

            # if obs gets too long, truncated obs will be fed into the model
            obs = obs[:config.agent.single_obs_max_length]
            running_states[idx] = torch.cat([running_states[idx], obs])
            vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(vllm_input_list[idx]['prompt_token_ids'], obs)

            # TODO: check whether the truncation is correct here
            if 'multi_modal_data' in vllm_input_list[idx].keys() and 'image' in mm.keys():
                vllm_input_list[idx]['multi_modal_data']['image'] += mm['image']

            obs_reward = torch.zeros(len(obs), dtype=torch.float)
            reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], obs_reward], dim=-1)

            obs_mask = torch.zeros(len(obs), dtype=torch.int64)
            running_action_masks[idx] = torch.cat([running_action_masks[idx], obs_mask])
            if running_states[idx].shape[-1] >= config.response_length:
                active_mask[idx] = False

    env.close()
    max_total_length = config.prompt_length + config.response_length
    target_device = prompts.batch['input_ids'].device
    running_states = [state[: max_total_length] for state in running_states]
    state_tensor = pad_2d_list_to_length(running_states, engine.tokenizer.pad_token_id, max_total_length).to(target_device)
    running_action_masks = [mask[: max_total_length] for mask in running_action_masks]
    action_mask_tensor = pad_2d_list_to_length(running_action_masks, 0, max_total_length).to(target_device)
    reward_tensor_list = [reward[: max_total_length] for reward in reward_tensor_list]
    reward_tensor = pad_2d_list_to_length(reward_tensor_list, 0.0, max_total_length).to(target_device)
    return state_tensor[:, -config.response_length:], action_mask_tensor, reward_tensor


def execute_tool_call(sample, pbar=None):
    action_list = sample.get('actions', [])
    available_tools = sample.get('available_tools', [])

    action_info_list = []
    for action_string in zip(action_list):
        try:
            action_info = json.loads(action_string)
            assert 'name' in action_info and 'arguments' in action_info
            assert action_info['name'] in available_tools
            action_info_list.append(action_info)
        except Exception as err:
            print(f' [ERROR] action json load {err=}')
            continue

    obs_list, reward_list = [], []
    if len(action_info_list) > 0:
        # openai/qwen25 style agent
        for action_info in action_info_list:
            target_tool = available_tools[action_info['name']]
            tool_result, reward = target_tool.execute(action_info)
    else:
        # text style agent (e.g. games)
        for key, tool in available_tools.items():
            tool_result, reward = tool.execute(action_list)
            obs_list += tool_result
            reward_list += reward

    if pbar is not None:
        pbar.update(1)
    return obs_list, reward_list


class ParallelEnv:
    """
    The interface intentionally designed to be the same as: https://github.com/openai/gym
    Hope this could be easier to use for RLers.
    """
    def __init__(self, env_config, tokenizer, **kwargs):
        self.config = env_config
        self.tokenizer = tokenizer
        # type: List[ Dict[ Str, ToolBase subclasses ] ]
        self.tools = []
        self.agent_meta = []

    def step(self, actions):
        """
        Input:
        - actions: vllm.RequestOutput

        Output:
        - observations: List[Dict], content like {"prompt_token_ids": ..., "multi_modal_data": ...}, 
                multi_modal_data only appears when there are images/videos in obs
        - rewards: List[ float ].
                each time after an action being executed, procedure rewards can be assigned to 
                the last valid token of model outputs. This might be useful for ..., 
                e.g., invalid action, code execution error, format error,
                or video game envs where immediate feedback is available.
        - dones: List[ Boolean ]
        - infos: Dict, for debugging only
        """
        dones = []
        valid_indices = []
        valid_actions = []
        
        # 1. filtering valid actions
        for idx, act in enumerate(actions):
            if act.outputs[0].finish_reason == 'length':
                dones.append(True)
                continue

            # NOTE: each agent response can have multiple function callings
            # multiple agent responses can be executed asyncronously
            action_text_list = _extract_tool_call_contents(
                self.config.action_start, 
                self.config.action_end, 
                act.outputs[0].text
            )
            if len(action_text_list) == 0:
                dones.append(True)
                continue

            dones.append(False)
            valid_indices.append(idx)
            valid_actions.append(action_text_list)

        agent_inputs = []
        for idx, actions in zip(valid_indices, valid_actions):
            agent_inputs.append(dict(
                idx=idx,
                actions=actions,
                available_tools=self.tools[idx],
                agent_meta=self.agent_meta[idx],
            ))

        # 2. executing actions
        num_workers = min(self.config.concurrent_workers, len(valid_actions))
        pbar = tqdm.tqdm(total=len(valid_actions), desc=f'Tool Calling on {num_workers=}') if self.config.show_tqdm else None
        if num_workers <= 1:
            agent_outputs = []
            reward_outputs = []
            for agi in agent_inputs:
                obs_list, reward_list = execute_tool_call(agi, pbar=pbar)
                reward_sum = sum(reward_list) if len(reward_list) > 0 else 0.0
                agent_outputs.append([{"role": "tool", "content": obs} for obs in obs_list])
                reward_outputs.append(reward_sum)
        else:
            partial_tool_func = partial(execute_tool_call, pbar=pbar)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                raw_outputs = list(executor.map(partial_tool_func, agent_inputs))
            obs_list = [raw[0] for raw in raw_outputs]
            reward_outputs = [sum(raw[1]) for raw in raw_outputs]
            agent_outputs = [[{"role": "tool", "content": obs} for obs in obs_list]]
        
        # 3. post-process tool output
        agent_output_final = [None] * len(actions)
        reward_output_final = [None] * len(actions)
        for idx, msg, rew in zip(valid_indices, agent_outputs, reward_outputs):
            obs_token_ids = self.tokenizer.apply_chat_template(msg, add_generation_prompt=True, return_tensors='pt')[0]
            # VERY UGLY: remove the system prompt added by qwen tokenizer
            eos_start_idx = torch.nonzero(obs_token_ids == self.tokenizer.eos_token_id)
            if eos_start_idx.shape[0] > 0:
                eos_start_idx = eos_start_idx[0].item()
                obs_token_ids = obs_token_ids[eos_start_idx : ]
            agent_output_final[idx] = {"prompt_token_ids": obs_token_ids}
            reward_output_final[idx] = rew
        
        return agent_output_tensors, reward_outputs, dones, {}

    def reset(self, prompts):
        self.tools = []
        self.agent_meta = []
        for i in range(len(prompts)):
            data_item = prompts[i]  # DataProtoItem
            tool_names = data_item.non_tensor_batch.get(self.config.tool_name_key, '')
            if tool_names:
                # init tools from config field `tool_name_key`
                tool_name_list = tool_names.split(',')
                tool_name_list = [name.strip() for name in tool_name_list]
                tool_fns = [ToolBase.create(name) for name in tool_name_list]
                self.tools.append(dict(zip(tool_name_list, tool_fns)))
            else:
                # try initialize tools using system prompt description
                raw_prompt = data_item.non_tensor_batch.get('raw_prompt', [])
                if len(raw_prompt) and raw_prompt[0]['role'] == 'system':
                    tool_fns = ToolBase.from_system_prompt(raw_prompt[0]['content'])
                    tool_name_list = [tool.name for tool in tool_fns]
                    self.tools.append(dict(zip(tool_name_list, tool_fns)))
                else:
                    self.tools.append({})

            tool_meta = data_item.non_tensor_batch.get(self.config.tool_meta_key, {})
            self.agent_meta.append(tool_meta)

    def close(self):
        self.tools = []
        self.agent_meta = []