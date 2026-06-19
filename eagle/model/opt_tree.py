from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class DraftNode:
    node_id: int
    parent_id: int | None
    depth: int
    token_id: int
    local_logprob: float
    path_logprob: float
    original_index: int | None = None


@dataclass(frozen=True)
class OptTreeConfig:
    enabled: bool = False
    budget: int = 60
    # Legacy compatibility knob from the old post-selection prototype. The
    # paper-style implementation expands/prunes dynamically and does not use it
    # to size an overexpanded fixed EAGLE tree.
    overexpand_factor: float = 1.0
    mode: str = "path_prob_greedy"
    debug: bool = False
    delta: float = 0.0
    lookahead_stop: bool = False
    lookahead_margin: float = 0.0
    min_expand_depth: int = 1
    max_expand_depth: int | None = None


def node_map(nodes: Iterable[DraftNode]) -> dict[int, DraftNode]:
    return {node.node_id: node for node in nodes}


def root_id(nodes: list[DraftNode]) -> int | None:
    for node in nodes:
        if node.parent_id is None:
            return node.node_id
    return None


def ancestor_ids(nodes_by_id: dict[int, DraftNode], node_id: int) -> list[int]:
    chain = []
    current = nodes_by_id[node_id]
    while current.parent_id is not None:
        chain.append(current.parent_id)
        current = nodes_by_id[current.parent_id]
    chain.reverse()
    return chain


def children_by_parent(nodes: list[DraftNode]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for node in nodes:
        if node.parent_id is not None:
            children.setdefault(node.parent_id, []).append(node.node_id)
    return children


def selected_leaf_ids(nodes: list[DraftNode], selected_ids: set[int]) -> list[int]:
    children = children_by_parent(nodes)
    leaves = []
    for node_id in selected_ids:
        if not any(child_id in selected_ids for child_id in children.get(node_id, [])):
            leaves.append(node_id)
    return leaves


def trim_to_budget(nodes: list[DraftNode], selected_ids: set[int], budget: int, root: int) -> None:
    nodes_by_id = node_map(nodes)
    while len(selected_ids) > budget:
        removable = [leaf for leaf in selected_leaf_ids(nodes, selected_ids) if leaf != root]
        if not removable:
            break
        worst_leaf = min(
            removable,
            key=lambda node_id: (nodes_by_id[node_id].path_logprob, nodes_by_id[node_id].depth, node_id),
        )
        selected_ids.remove(worst_leaf)


def select_opt_tree(
    nodes: list[DraftNode],
    budget: int,
    *,
    include_root: bool = True,
) -> set[int]:
    if budget <= 0:
        return set()
    root = root_id(nodes)
    if root is None:
        raise ValueError("OPT-Tree requires a root node.")
    nodes_by_id = node_map(nodes)
    selected_ids = {root} if include_root else set()
    non_root_budget = max(0, budget - 1 if include_root else budget)
    if non_root_budget == 0:
        return selected_ids

    candidates = [
        node
        for node in nodes
        if node.node_id != root
    ]
    # Paper OPT-Tree selects the nodes with largest path probability. Since a
    # parent has path probability >= any descendant, shallow nodes should win
    # exact ties to preserve the connected-subtree property.
    candidates.sort(key=lambda node: (node.path_logprob, -node.depth, -node.node_id), reverse=True)
    for node in candidates[:non_root_budget]:
        selected_ids.add(node.node_id)
    issues = validate_connected_subtree(nodes, selected_ids, budget=budget)
    if not issues:
        return selected_ids

    # Numerical ties or an approximate candidate pool can violate the theorem's
    # ideal connectedness. Fall back to ancestor completion and leaf trimming.
    selected_ids = {root} if include_root else set()
    for node in candidates:
        chain = ancestor_ids(nodes_by_id, node.node_id) + [node.node_id]
        selected_ids.update(chain)
        trim_to_budget(nodes, selected_ids, budget, root)
    return selected_ids


def validate_connected_subtree(
    nodes: list[DraftNode],
    selected_ids: set[int],
    budget: int | None = None,
) -> list[str]:
    issues = []
    root = root_id(nodes)
    nodes_by_id = node_map(nodes)
    if root is None:
        return ["missing root node"]
    if root not in selected_ids:
        issues.append("root is not selected")
    if budget is not None and len(selected_ids) > budget:
        issues.append(f"selected size {len(selected_ids)} exceeds budget {budget}")
    for node_id in selected_ids:
        if node_id not in nodes_by_id:
            issues.append(f"selected node {node_id} does not exist")
            continue
        current = nodes_by_id[node_id]
        while current.parent_id is not None:
            if current.parent_id not in selected_ids:
                issues.append(f"node {node_id} is missing ancestor {current.parent_id}")
                break
            current = nodes_by_id[current.parent_id]
    return issues


def raw_topn_threshold(nodes: list[DraftNode], budget: int) -> float | None:
    non_root = [node for node in nodes if node.parent_id is not None]
    non_root_budget = max(0, budget - 1)
    if non_root_budget <= 0 or len(non_root) < non_root_budget:
        return None
    non_root.sort(key=lambda node: (node.path_logprob, -node.depth, -node.node_id), reverse=True)
    return float(non_root[non_root_budget - 1].path_logprob)


def connected_selected_min_path_logprob(nodes: list[DraftNode], selected_ids: set[int]) -> float | None:
    nodes_by_id = node_map(nodes)
    values = [
        float(nodes_by_id[node_id].path_logprob)
        for node_id in selected_ids
        if node_id in nodes_by_id and nodes_by_id[node_id].parent_id is not None
    ]
    return min(values) if values else None


def connected_selected_leaf_min_path_logprob(nodes: list[DraftNode], selected_ids: set[int]) -> float | None:
    nodes_by_id = node_map(nodes)
    leaves = selected_leaf_ids(nodes, selected_ids)
    values = [
        float(nodes_by_id[node_id].path_logprob)
        for node_id in leaves
        if node_id in nodes_by_id and nodes_by_id[node_id].parent_id is not None
    ]
    return min(values) if values else None


def rebuild_selected_tree(nodes: list[DraftNode], selected_ids: set[int]) -> dict:
    root = root_id(nodes)
    if root is None:
        raise ValueError("OPT-Tree requires a root node.")
    nodes_by_id = node_map(nodes)
    selected_nodes = [nodes_by_id[node_id] for node_id in selected_ids]
    selected_nodes.sort(key=lambda node: (node.depth, node.parent_id is not None, node.node_id))
    root_node = nodes_by_id[root]
    selected_nodes = [root_node] + [node for node in selected_nodes if node.node_id != root]
    old_to_new_id = {node.node_id: index for index, node in enumerate(selected_nodes)}
    new_to_old_id = {index: node.node_id for index, node in enumerate(selected_nodes)}
    selected_parent_ids = [
        -1 if node.parent_id is None else old_to_new_id[node.parent_id]
        for node in selected_nodes
    ]
    selected_depths = [int(node.depth) for node in selected_nodes]
    selected_token_ids = [int(node.token_id) for node in selected_nodes]
    selected_path_logprobs = [float(node.path_logprob) for node in selected_nodes]
    return {
        "selected_nodes": selected_nodes,
        "old_to_new_id": old_to_new_id,
        "new_to_old_id": new_to_old_id,
        "selected_parent_ids": selected_parent_ids,
        "selected_depths": selected_depths,
        "selected_token_ids": selected_token_ids,
        "selected_path_logprobs": selected_path_logprobs,
    }


def build_tree_attention_from_rebuild(rebuilt: dict) -> dict:
    selected_nodes: list[DraftNode] = rebuilt["selected_nodes"]
    selected_parent_ids: list[int] = rebuilt["selected_parent_ids"]
    total_nodes = len(selected_nodes)
    tree_mask = torch.eye(total_nodes, dtype=torch.bool)
    tree_mask[:, 0] = True
    for new_id in range(1, total_nodes):
        parent_new_id = selected_parent_ids[new_id]
        if parent_new_id >= 0:
            tree_mask[new_id].add_(tree_mask[parent_new_id])
    position_ids = torch.tensor(rebuilt["selected_depths"], dtype=torch.long)
    children = {index: [] for index in range(total_nodes)}
    for new_id in range(1, total_nodes):
        children[selected_parent_ids[new_id]].append(new_id)
    leaves = [index for index in range(total_nodes) if not children[index]]
    max_depth = int(position_ids.max().item()) + 1 if total_nodes else 1
    retrieve_indices = torch.zeros(len(leaves), max_depth, dtype=torch.long) - 1
    for row, leaf in enumerate(leaves):
        chain = []
        current = leaf
        while current >= 0:
            chain.append(current)
            current = selected_parent_ids[current]
        chain.reverse()
        retrieve_indices[row, : len(chain)] = torch.tensor(chain, dtype=torch.long)
    return {
        "tree_mask": tree_mask,
        "position_ids": position_ids,
        "retrieve_indices": retrieve_indices,
        "leaves": leaves,
    }


def validate_tree_attention(rebuilt: dict, tree_data: dict) -> list[str]:
    issues = []
    selected_parent_ids: list[int] = rebuilt["selected_parent_ids"]
    depths: list[int] = rebuilt["selected_depths"]
    tree_mask: torch.Tensor = tree_data["tree_mask"]
    position_ids: torch.Tensor = tree_data["position_ids"]
    retrieve_indices: torch.Tensor = tree_data["retrieve_indices"]
    total_nodes = len(depths)
    if list(position_ids.cpu().tolist()) != depths:
        issues.append("position_ids do not match selected depths")
    for row in retrieve_indices.cpu().tolist():
        valid = [node_id for node_id in row if node_id >= 0]
        for node_id in valid:
            if node_id >= total_nodes:
                issues.append(f"retrieve path references missing node {node_id}")
        for parent, child in zip(valid, valid[1:]):
            if selected_parent_ids[child] != parent:
                issues.append(f"retrieve path {valid} is not a parent chain")
    mask = tree_mask.cpu().bool()
    for node_id in range(total_nodes):
        allowed = set()
        current = node_id
        while current >= 0:
            allowed.add(current)
            current = selected_parent_ids[current]
        actual = {index for index, flag in enumerate(mask[node_id].tolist()) if flag}
        if actual != allowed:
            issues.append(f"node {node_id} attention mask is not exactly its ancestor chain")
    return issues


def build_draft_nodes_from_eagle(
    path_logprobs: torch.Tensor,
    token_ids: torch.Tensor,
    parent_refs: torch.Tensor,
    sample_token_id: int,
    *,
    strict_parents: bool = False,
) -> list[DraftNode]:
    flat_scores = path_logprobs.detach().float().cpu().view(-1).tolist()
    flat_tokens = token_ids.detach().cpu().view(-1).tolist()
    flat_parents = parent_refs.detach().cpu().view(-1).tolist()
    nodes = [
        DraftNode(
            node_id=0,
            parent_id=None,
            depth=0,
            token_id=int(sample_token_id),
            local_logprob=0.0,
            path_logprob=0.0,
            original_index=None,
        )
    ]
    by_id = {0: nodes[0]}
    for original_index, (path_logprob, token_id, parent_ref) in enumerate(
        zip(flat_scores, flat_tokens, flat_parents)
    ):
        node_id = original_index + 1
        parent_id = int(parent_ref)
        parent = by_id.get(parent_id)
        if parent is None:
            if strict_parents:
                raise ValueError(
                    f"OPT-Tree node {node_id} references missing parent {parent_id}. "
                    "This usually means sparse draft node ids were mixed with compact "
                    "selected-node ids during dynamic pruning."
                )
            parent_id = 0
            parent = by_id[0]
        local_logprob = float(path_logprob) - float(parent.path_logprob)
        node = DraftNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=parent.depth + 1,
            token_id=int(token_id),
            local_logprob=local_logprob,
            path_logprob=float(path_logprob),
            original_index=original_index,
        )
        nodes.append(node)
        by_id[node_id] = node
    return nodes


def restrict_to_overexpanded_pool(nodes: list[DraftNode], max_nodes: int) -> list[DraftNode]:
    root = root_id(nodes)
    if root is None or max_nodes <= 0:
        return nodes[:1]
    nodes_by_id = node_map(nodes)
    non_root = [node for node in nodes if node.node_id != root]
    non_root.sort(key=lambda node: (node.path_logprob, node.depth, -node.node_id), reverse=True)
    selected_ids = {root}
    for node in non_root[: max(0, max_nodes - 1)]:
        selected_ids.update(ancestor_ids(nodes_by_id, node.node_id))
        selected_ids.add(node.node_id)
    return [node for node in nodes if node.node_id in selected_ids]


def depth_histogram(nodes: list[DraftNode]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for node in nodes:
        hist[str(node.depth)] = hist.get(str(node.depth), 0) + 1
    return dict(sorted(hist.items(), key=lambda item: int(item[0])))
