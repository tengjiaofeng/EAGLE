import copy
import json
import time
from collections import Counter

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
#from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
#from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .configs import EConfig


class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            ddd_enabled=False,
            ddd_mode="paper_exact",
            ddd_max_draft_calls=11,
            ddd_beam_width=10,
            ddd_check_steps=(5, 7, 9),
            ddd_threshold=-0.3,
            ddd_verbose=False,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        if use_eagle3:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d
        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.configure_ddd(
            ddd_enabled=ddd_enabled,
            ddd_mode=ddd_mode,
            ddd_max_draft_calls=ddd_max_draft_calls,
            ddd_beam_width=ddd_beam_width,
            ddd_check_steps=ddd_check_steps,
            ddd_threshold=ddd_threshold,
            ddd_verbose=ddd_verbose,
        )
        if hasattr(self.ea_layer, "configure_opt_tree"):
            self.ea_layer.configure_opt_tree(opt_tree_enabled=False)
        self.ea_layer.init_tree()
        self.ddd_runtime_metrics = None

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    def configure_ddd(
            self,
            ddd_enabled=False,
            ddd_mode="paper_exact",
            ddd_max_draft_calls=11,
            ddd_beam_width=10,
            ddd_check_steps=(5, 7, 9),
            ddd_threshold=-0.3,
            ddd_verbose=False,
    ):
        self.ea_layer.configure_ddd(
            ddd_enabled=ddd_enabled,
            ddd_mode=ddd_mode,
            ddd_max_draft_calls=ddd_max_draft_calls,
            ddd_beam_width=ddd_beam_width,
            ddd_check_steps=ddd_check_steps,
            ddd_threshold=ddd_threshold,
            ddd_verbose=ddd_verbose,
        )

    def configure_opt_tree(
            self,
            opt_tree_enabled=False,
            opt_tree_budget=60,
            opt_tree_overexpand_factor=1.0,
            opt_tree_mode="path_prob_greedy",
            opt_tree_debug=False,
            opt_tree_delta=0.0,
            opt_tree_lookahead_stop=False,
            opt_tree_lookahead_margin=0.0,
            opt_tree_min_expand_depth=1,
            opt_tree_max_expand_depth=None,
    ):
        if not hasattr(self.ea_layer, "configure_opt_tree"):
            if opt_tree_enabled:
                raise ValueError("OPT-Tree is only implemented for EAGLE-3 in this minimal version.")
            return
        self.ea_layer.configure_opt_tree(
            opt_tree_enabled=opt_tree_enabled,
            opt_tree_budget=opt_tree_budget,
            opt_tree_overexpand_factor=opt_tree_overexpand_factor,
            opt_tree_mode=opt_tree_mode,
            opt_tree_debug=opt_tree_debug,
            opt_tree_delta=opt_tree_delta,
            opt_tree_lookahead_stop=opt_tree_lookahead_stop,
            opt_tree_lookahead_margin=opt_tree_lookahead_margin,
            opt_tree_min_expand_depth=opt_tree_min_expand_depth,
            opt_tree_max_expand_depth=opt_tree_max_expand_depth,
        )

    def _reset_ddd_runtime_metrics(self):
        self.ddd_runtime_metrics = {
            "draft_debug": [],
            "opt_tree_debug": [],
            "draft_calls": [],
            "accepted_lengths": [],
            "early_stops": 0,
            "stop_depths": [],
        }
        if hasattr(self.ea_layer, "last_ddd_debug"):
            self.ea_layer.last_ddd_debug = None
        if hasattr(self.ea_layer, "last_opt_tree_debug"):
            self.ea_layer.last_opt_tree_debug = None

    def reset_ddd_runtime_metrics(self):
        self._reset_ddd_runtime_metrics()

    def _record_ddd_verify_debug(self):
        debug = getattr(self.ea_layer, "last_ddd_debug", None)
        if self.ddd_runtime_metrics is None or debug is None:
            opt_debug = getattr(self.ea_layer, "last_opt_tree_debug", None)
            if self.ddd_runtime_metrics is not None and opt_debug is not None:
                self.ddd_runtime_metrics["opt_tree_debug"].append(copy.deepcopy(opt_debug))
            return
        debug = copy.deepcopy(debug)
        self.ddd_runtime_metrics["draft_debug"].append(debug)
        self.ddd_runtime_metrics["draft_calls"].append(debug["draft_calls"])
        if debug["stopped_by_ddd"]:
            self.ddd_runtime_metrics["early_stops"] += 1
            self.ddd_runtime_metrics["stop_depths"].append(debug["stop_call_count"])
        opt_debug = getattr(self.ea_layer, "last_opt_tree_debug", None)
        if opt_debug is not None:
            self.ddd_runtime_metrics["opt_tree_debug"].append(copy.deepcopy(opt_debug))

    def _record_ddd_accept_length(self, accept_length):
        if self.ddd_runtime_metrics is None:
            return
        self.ddd_runtime_metrics["accepted_lengths"].append(int(accept_length))

    def get_ddd_runtime_metrics(self):
        metrics = self.ddd_runtime_metrics or {}
        draft_calls = metrics.get("draft_calls", [])
        stop_depths = metrics.get("stop_depths", [])
        accepted_lengths = metrics.get("accepted_lengths", [])
        avg_draft_calls = sum(draft_calls) / len(draft_calls) if draft_calls else 0.0
        avg_accept_length = sum(accepted_lengths) / len(accepted_lengths) if accepted_lengths else 0.0
        return {
            "average_draft_calls_per_verify": avg_draft_calls,
            "draft_call_distribution": dict(Counter(draft_calls)),
            "ddd_early_stops": metrics.get("early_stops", 0),
            "stop_depth_histogram": dict(Counter(stop_depths)),
            "average_accepted_length": avg_accept_length,
            "draft_debug": metrics.get("draft_debug", []),
        }

    def get_opt_tree_runtime_metrics(self):
        metrics = self.ddd_runtime_metrics or {}
        debug = metrics.get("opt_tree_debug", [])
        if not debug:
            return {
                "enabled": False,
                "num_verify_rounds": 0,
                "num_overexpanded_nodes_mean": 0.0,
                "num_selected_nodes_mean": 0.0,
                "selected_path_logprob_sum_mean": 0.0,
                "opt_tree_stop_reason_histogram": {},
                "posterior_delta_stops": 0,
                "lookahead_stop_rate": 0.0,
                "lookahead_stops": 0,
                "frontier_bound_stops": 0,
                "lookahead_stop_depth_mean": 0.0,
                "draft_calls_saved_estimate_mean": 0.0,
                "num_nodes_saved_mean": 0.0,
                "tree_expansion_time_s": 0.0,
                "tree_selection_time_s": 0.0,
                "tree_rebuild_time_s": 0.0,
                "tree_mask_build_time_s": 0.0,
                "opt_tree_overhead_s": 0.0,
                "selected_depth_histogram": {},
                "debug": [],
            }
        depth_hist = Counter()
        stop_reason_hist = Counter()
        for item in debug:
            for key, value in item.get("selected_depth_histogram", {}).items():
                depth_hist[str(key)] += int(value)
            reason = item.get("opt_tree_stop_reason")
            if reason:
                stop_reason_hist[str(reason)] += 1
        lookahead_stops = [item for item in debug if item.get("lookahead_stopped")]
        frontier_bound_stops = [
            item for item in debug if item.get("lookahead_stop_reason") == "frontier_bound"
        ]
        posterior_delta_stops = [
            item for item in debug if item.get("opt_tree_stop_reason") == "posterior_delta"
        ]
        stop_depths = [
            int(item["lookahead_stop_depth"])
            for item in lookahead_stops
            if item.get("lookahead_stop_depth") is not None
        ]
        node_savings = [
            int(item.get("num_nodes_without_stop_estimate", 0))
            - int(item.get("num_nodes_with_stop", 0))
            for item in debug
        ]
        return {
            "enabled": True,
            "num_verify_rounds": len(debug),
            "num_overexpanded_nodes_mean": sum(item.get("num_overexpanded_nodes", 0) for item in debug) / len(debug),
            "num_selected_nodes_mean": sum(item.get("num_selected_nodes", 0) for item in debug) / len(debug),
            "selected_path_logprob_sum_mean": sum(item.get("selected_path_logprob_sum", 0.0) for item in debug) / len(debug),
            "opt_tree_stop_reason_histogram": dict(sorted(stop_reason_hist.items())),
            "posterior_delta_stops": len(posterior_delta_stops),
            "lookahead_stop_rate": len(lookahead_stops) / len(debug),
            "lookahead_stops": len(lookahead_stops),
            "frontier_bound_stops": len(frontier_bound_stops),
            "lookahead_stop_depth_mean": sum(stop_depths) / len(stop_depths) if stop_depths else 0.0,
            "draft_calls_saved_estimate_mean": sum(item.get("draft_calls_saved_estimate", 0) for item in debug) / len(debug),
            "num_nodes_saved_mean": sum(node_savings) / len(node_savings) if node_savings else 0.0,
            "tree_expansion_time_s": sum(item.get("tree_expansion_time_s", 0.0) for item in debug),
            "tree_selection_time_s": sum(item.get("tree_selection_time_s", 0.0) for item in debug),
            "tree_rebuild_time_s": sum(item.get("tree_rebuild_time_s", 0.0) for item in debug),
            "tree_mask_build_time_s": sum(item.get("tree_mask_build_time_s", 0.0) for item in debug),
            "opt_tree_overhead_s": sum(item.get("opt_tree_overhead_s", 0.0) for item in debug),
            "selected_depth_histogram": dict(sorted(depth_hist.items(), key=lambda item: int(item[0]))),
            "debug": debug,
        }

    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=7,
            top_k=10,
            threshold=1.0,
            ddd_enabled=False,
            ddd_mode="paper_exact",
            ddd_max_draft_calls=11,
            ddd_beam_width=10,
            ddd_check_steps=(5, 7, 9),
            ddd_threshold=-0.3,
            ddd_verbose=False,
            **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen3ForCausalLM':
            base_model = KVQwen3ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            ddd_enabled=ddd_enabled,
            ddd_mode=ddd_mode,
            ddd_max_draft_calls=ddd_max_draft_calls,
            ddd_beam_width=ddd_beam_width,
            ddd_check_steps=ddd_check_steps,
            ddd_threshold=ddd_threshold,
            ddd_verbose=ddd_verbose,
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        self._reset_ddd_runtime_metrics()
        # prefill
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self._record_ddd_verify_debug()
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # Target model forward, get logits
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            self._record_ddd_accept_length(accept_length)
            # print(accept_length)
            # Adjusting the input sequence, draft model forward
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        self._reset_ddd_runtime_metrics()
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self._record_ddd_verify_debug()
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            self._record_ddd_accept_length(accept_length)
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
