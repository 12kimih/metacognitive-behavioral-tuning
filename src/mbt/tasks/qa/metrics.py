import re
import string
from collections import Counter, defaultdict

import evaluate
import numpy as np


# https://github.com/volcengine/verl/blob/3a394c9bd09c8bd321ae2070edaa3b5e84af201a/verl/utils/reward_score/search_r1_like_qa_em.py
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)))).strip()


def extract_answer(s: str) -> str:
    last_open_tag_pos = s.rfind("<answer>")
    if last_open_tag_pos == -1:
        return ""

    first_close_tag_pos = s.find("</answer>", last_open_tag_pos)
    if first_close_tag_pos == -1:
        return ""

    start = last_open_tag_pos + len("<answer>")
    end = first_close_tag_pos
    return s[start:end].strip()


def exact_match(prediction: str, references: list[str]) -> float:
    if not prediction:
        return 0.0
    return 1.0 if any(reference == prediction for reference in references) else 0.0


def substring_match(prediction: str, references: list[str]) -> float:
    if not prediction:
        return 0.0
    return 1.0 if any(reference in prediction for reference in references) else 0.0


def f1_score(prediction: str, references: list[str]) -> float:
    if not prediction:
        return 0.0
    max_f1 = 0.0
    prediction_tokens = prediction.split()
    for reference in references:
        reference_tokens = reference.split()
        if not reference_tokens:
            continue
        common_tokens = Counter(prediction_tokens) & Counter(reference_tokens)
        num_common = sum(common_tokens.values())
        if num_common == 0:
            continue
        precision = num_common / len(prediction_tokens)
        recall = num_common / len(reference_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        max_f1 = max(max_f1, f1)
    return max_f1


def bleu(prediction: str, references: list[str]) -> float:
    if not prediction:
        return 0.0
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=[prediction], references=[references])
    return results["bleu"]


def rouge_l(prediction: str, references: list[str]) -> float:
    if not prediction:
        return 0.0
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=[prediction], references=[references], use_stemmer=True)
    return results["rougeL"]


METRIC_FUNCTIONS = {
    "exact_match": exact_match,
    "substring_match": substring_match,
    "f1_score": f1_score,
    "bleu": bleu,
    "rouge_l": rouge_l,
}


METRIC_AGGREGATIONS = {
    "exact_match": ["avg", "maj", "pass"],
    "substring_match": ["avg", "maj", "pass"],
    "f1_score": ["avg", "maj"],
    "bleu": ["avg", "maj"],
    "rouge_l": ["avg", "maj"],
    "llm_as_judge": ["avg", "maj", "pass"],
}


def compute_metrics(example: dict, metrics: list[str], prediction_column: str, reference_columns: list[str]) -> dict:
    normalized_prediction = normalize_answer(example[prediction_column])
    references = [item for col in reference_columns if col in example for item in ([example[col]] if isinstance(example[col], str) else example[col])]
    normalized_references = [normalize_answer(reference) for reference in references]
    results = {}
    for metric in metrics:
        if metric in METRIC_FUNCTIONS:
            results.update({metric: METRIC_FUNCTIONS[metric](normalized_prediction, normalized_references)})
    return results


def aggregate_metrics(dataset, metrics: list[str], prediction_column: str) -> dict:
    sample_aggregates = {metric: defaultdict(list) for metric in metrics}
    for _, df in dataset.to_pandas().groupby("sample_id"):
        n = len(df)
        ks = [1]
        while ks[-1] * 2 <= n:
            ks.append(ks[-1] * 2)
        p = df[prediction_column].to_list()

        for metric in metrics:
            if metric in METRIC_AGGREGATIONS:
                s = df[metric].to_list()
                c = s.count(1.0)

                if "avg" in METRIC_AGGREGATIONS[metric]:
                    for k in ks:
                        sample_aggregates[metric][f"avg@{k}"].append(np.mean(s[:k]))
                if "maj" in METRIC_AGGREGATIONS[metric]:
                    for k in ks:
                        most_common = Counter(p[:k]).most_common(1)[0][0]
                        most_common_idx = p.index(most_common)
                        sample_aggregates[metric][f"maj@{k}"].append(s[most_common_idx])
                if "pass" in METRIC_AGGREGATIONS[metric]:
                    for k in ks:
                        sample_aggregates[metric][f"pass@{k}"].append(1.0 if n - c < k else 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))

    results = {}
    for metric, aggregations in sample_aggregates.items():
        results[metric] = {}
        for agg_key, values in aggregations.items():
            results[metric][agg_key] = round(np.mean(values) * 100, 2)

    return results
