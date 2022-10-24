import logging
import os
import torch
from torch import Tensor
from torch.nn import functional as F

def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def try_create_dir(path):
    os.system(f"mkdir -p {path}")


def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)

def repetition_penalty(input_ids, scores, penalty=1.):
    """
        input_ids: [batch_size, seq_len]
        scores: [batch_size, vocab_size] (logits before softmax)
        penalty: >1 for penalty, 1. no penalty
    """
    score = torch.gather(scores, 1, input_ids)

    # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
    score = torch.where(score < 0, score * penalty, score / penalty)

    scores.scatter_(1, input_ids, score)
    return scores

def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        idxs_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[idxs_to_remove] = filter_value
    if top_p > 0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cummulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_idx_to_remove = cummulative_probs > top_p
        sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
        sorted_idx_to_remove[..., 0] = 0

        idxs_to_remove = sorted_idx[sorted_idx_to_remove]
        logits[idxs_to_remove] = filter_value
    idxs_to_remove = logits < threshold
    logits[idxs_to_remove] = filter_value
    # print(logits.size())
    return logits
