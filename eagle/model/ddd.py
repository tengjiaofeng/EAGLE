from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch


PAPER_EXACT_CHECK_STEPS = (5, 7, 9)


@dataclass(frozen=True)
class DDDConfig:
    enabled: bool = False
    mode: str = "paper_exact"
    max_draft_calls: int = 11
    beam_width: int = 10
    check_steps: Tuple[int, ...] = PAPER_EXACT_CHECK_STEPS
    threshold: float = -0.3
    verbose: bool = False


def normalize_check_steps(check_steps: Optional[Iterable[int]]) -> Tuple[int, ...]:
    if check_steps is None:
        return PAPER_EXACT_CHECK_STEPS
    return tuple(int(step) for step in check_steps)


def compute_ddd_heuristic(logprobsum: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(logprobsum.float(), dim=0)


def update_ddd_logprobsum(
        old_logprobsum: torch.Tensor,
        selected_parent_idx: torch.Tensor,
        selected_token_logprobs: torch.Tensor,
) -> torch.Tensor:
    return old_logprobsum[selected_parent_idx] + selected_token_logprobs


def should_stop_ddd(config: DDDConfig, call_count: int, logprobsum: torch.Tensor):
    if not config.enabled or call_count not in config.check_steps:
        return False, None
    heuristic = compute_ddd_heuristic(logprobsum)
    return bool(heuristic.item() < config.threshold), heuristic


def make_ddd_debug(
        config: DDDConfig,
        draft_calls: int,
        stopped_by_ddd: bool,
        stop_call_count: Optional[int],
        checked_h,
        logprobsum: torch.Tensor,
):
    return {
        "ddd_enabled": bool(config.enabled),
        "ddd_mode": config.mode,
        "draft_calls": int(draft_calls),
        "stopped_by_ddd": bool(stopped_by_ddd),
        "stop_call_count": None if stop_call_count is None else int(stop_call_count),
        "checked_H": [(int(step), float(value)) for step, value in checked_h],
        "final_logprobsum_shape": tuple(logprobsum.shape),
    }
