import re
import evaluate

exact_match = evaluate.load("exact_match")

def compute_score(data_source: str, predict_str: str, ground_truth: str) -> float:
    if data_source.lower() == 'rag':
        predict_str = '<think>' + predict_str
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        think_match = re.search(think_pattern, predict_str)
        if not think_match:
            return 0.0

        predict_no_think = predict_str.split('</think>')[-1]
        answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        match_result = re.search(answer_pattern, predict_no_think)
        if match_result:
            answer_text = match_result.group(1).strip()
            score_info = exact_match.compute(references=[ground_truth], predictions=[answer_text], ignore_case=True, ignore_punctuation=True)
            acc_reward = float(score_info['exact_match'])
            format_reward = 1.0
        else:
            acc_reward = 0.0
            format_reward = 0.0
        return 0.8 * acc_reward + 0.2 * format_reward

    # For games the reward are given by env
    return 0.0
