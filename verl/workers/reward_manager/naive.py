# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch

import json
import datetime

class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

        self.step_cnt = 0

    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        action_or_attn_mask = data.batch['action_mask'] # if 'action_mask' in data.batch.keys() else data.batch['attention_mask']
        if 'env_reward' in data.batch.keys():
            reward_tensor += data.batch['env_reward']
            print(f' [DEBUG reward] mean={reward_tensor.mean().item()}, min={reward_tensor.min().item()}, max={reward_tensor.max().item()}')

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            # reward_tensor[i, valid_response_length - 1] = score
            eos_idx = torch.nonzero(action_or_attn_mask[i, prompt_length: prompt_length + valid_response_length])[-1]
            reward_tensor[i, eos_idx] = score

            # # FOR DEBUGGING ONLY!!! DO NOT COMMIT!!!
            # action_mask = action_or_attn_mask[i, prompt_length: prompt_length + valid_response_length]
            # env_reward = data_item.batch['env_reward'][:valid_response_length]
            # debug_output = dict(
            #     step=self.step_cnt,
            #     timetag=str(datetime.datetime.now()),
            #     prompt=prompt_str,
            #     response=response_str,
            #     ground_truth=str(ground_truth),
            #     score=float(score),
            #     env_reward_sum=float(env_reward.cpu().numpy().sum()),
            #     valid_prompt_length=int(valid_prompt_length.cpu().item()),
            #     valid_response_length=int(valid_response_length.cpu().item()),
            #     prompt_ids=valid_prompt_ids.cpu().numpy().tolist(),
            #     response_ids=valid_response_ids.cpu().numpy().tolist(),
            #     action_mask=action_mask.cpu().numpy().tolist(),
            #     reward_list=reward_tensor[i, :valid_response_length].cpu().numpy().tolist(),
            # )

            # debug_output_str = json.dumps(debug_output, ensure_ascii=False)
            # with open('/cpfs/user/fengyuan/code/github/verl/checkpoints/agent_ppo_debug/ppo_rewards_32b.jsonl', 'a+') as fout:
            #     fout.write(debug_output_str + '\n')

            # if data_source not in already_print_data_sources:
            #     already_print_data_sources[data_source] = 0

            # if already_print_data_sources[data_source] < self.num_examine:
            #     already_print_data_sources[data_source] += 1
            #     print("[prompt]", prompt_str)
            #     print("[response]", response_str)
            #     print("[ground_truth]", ground_truth)
            #     print("[score]", score)

        self.step_cnt += 1

        return reward_tensor
