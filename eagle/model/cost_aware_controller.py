from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


ConfidenceMode = Literal["top1", "entropy"]


@dataclass
class CostAwareController:
    """Cost-aware dynamic depth controller for EAGLE drafting."""

    tau: float = 0.6
    momentum: float = 0.9
    min_depth: int = 1
    max_depth: int | None = None
    confidence_mode: ConfidenceMode = "top1"
    utility_drop_tolerance: float = 0.0
    enable_confidence_stop: bool = True
    enable_cost_stop: bool = True
    measure_cost: bool = False
    eps: float = 1e-8

    ema_draft_time: float | None = None
    ema_verify_time: float | None = None
    cumulative_confidence: float = 1.0
    previous_utility: float | None = None
    last_depth: int = 0
    draft_depth_history: list[int] = field(default_factory=list)
    accepted_token_history: list[int] = field(default_factory=list)
    generated_token_history: list[int] = field(default_factory=list)

    def reset_tree(self) -> None:
        self.cumulative_confidence = 1.0
        self.previous_utility = None
        self.last_depth = 0

    def set_offline_costs(
        self,
        ema_draft_time: float,
        ema_verify_time: float,
        *,
        measure_cost: bool = False,
    ) -> None:
        self.ema_draft_time = float(ema_draft_time)
        self.ema_verify_time = float(ema_verify_time)
        self.measure_cost = measure_cost

    def record_draft_depth(self, depth: int) -> None:
        self.draft_depth_history.append(int(depth))

    def record_acceptance(self, accept_length: int) -> None:
        accept_length = int(accept_length)
        self.accepted_token_history.append(accept_length)
        self.generated_token_history.append(accept_length + 1)

    def stats_snapshot(self) -> dict[str, int]:
        return {
            "draft_depth_count": len(self.draft_depth_history),
            "accepted_count": len(self.accepted_token_history),
            "generated_count": len(self.generated_token_history),
        }

    def update_ema(self, real_draft_time: float, real_verify_time: float) -> None:
        if self.ema_draft_time is None:
            self.ema_draft_time = float(real_draft_time)
        else:
            self.ema_draft_time = (
                self.momentum * self.ema_draft_time
                + (1.0 - self.momentum) * float(real_draft_time)
            )

        if self.ema_verify_time is None:
            self.ema_verify_time = float(real_verify_time)
        else:
            self.ema_verify_time = (
                self.momentum * self.ema_verify_time
                + (1.0 - self.momentum) * float(real_verify_time)
            )

    def confidence_from_logprobs(self, logprobs: torch.Tensor) -> float:
        with torch.no_grad():
            probs = logprobs.float().exp()
            if self.confidence_mode == "top1":
                confidence = probs.max(dim=-1).values.mean()
            elif self.confidence_mode == "entropy":
                entropy = -(probs * logprobs.float()).sum(dim=-1)
                vocab_size = max(logprobs.shape[-1], 2)
                normalizer = torch.log(
                    torch.tensor(float(vocab_size), device=logprobs.device)
                )
                confidence = (1.0 - entropy / normalizer).clamp(0.0, 1.0).mean()
            else:
                raise ValueError(f"Unknown confidence_mode: {self.confidence_mode}")
        return float(confidence.detach().item())

    def update_confidence(self, logprobs: torch.Tensor) -> float:
        step_confidence = self.confidence_from_logprobs(logprobs)
        self.cumulative_confidence *= step_confidence
        return self.cumulative_confidence

    def expected_cost(self, current_depth: int) -> float:
        draft_time = self.ema_draft_time if self.ema_draft_time is not None else 1.0
        verify_time = self.ema_verify_time if self.ema_verify_time is not None else 1.0
        return draft_time * max(current_depth, 1) + verify_time + self.eps

    def expected_utility(self, current_depth: int, cumulative_confidence: float) -> float:
        expected_gain = cumulative_confidence * (current_depth + 1)
        return expected_gain / self.expected_cost(current_depth)

    def should_stop_drafting(
        self,
        current_depth: int,
        cumulative_confidence: float | None = None,
    ) -> bool:
        if cumulative_confidence is None:
            cumulative_confidence = self.cumulative_confidence
        self.last_depth = max(self.last_depth, current_depth)

        if self.max_depth is not None and current_depth >= self.max_depth:
            return True
        if current_depth < self.min_depth:
            return False
        if self.enable_confidence_stop and cumulative_confidence < self.tau:
            return True

        if not self.enable_cost_stop:
            return False

        utility = self.expected_utility(current_depth, cumulative_confidence)
        should_stop = False
        if self.previous_utility is not None:
            threshold = self.previous_utility * (1.0 - self.utility_drop_tolerance)
            should_stop = utility < threshold
        self.previous_utility = utility
        return should_stop
