"""
Agent Tool Definition 压缩实验脚本 
=====================================================================
核心逻辑：
  每个 trace 有多个 spans（多轮 LLM 调用），每个 span 的 input 包含从第一轮到
  当前轮的完整对话历史。本脚本取每个 trace 中间偏后的随机 span（避开开头和结尾的
  模板化动作如 finish/text_reply），确保 target 是真实的工具调用决策。

三种对比方式（仅压缩 Tool Definitions，历史上下文保持不变）：
  1. Full: 原始 Tool Definitions + 完整历史 → 计算 GT Action 的 Loss
  2. Truncation: 截断 Tool Definitions 前 N 个 token（与 C2KV 同压缩比）
  3. C2KV: C2KV 压缩 Tool Definitions → KV Cache + 完整历史 → Loss
"""

import argparse
import gc
import glob
import json
import os
import random
import time
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python", "inference"))

from reuse_pipeline import tokenize_for_reuse
from models import get_model_class, blend_gist_key_values

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/zhuyuhan/project/C2KV/datasets/agent-llm-traces/data"
DEFAULT_CHECKPOINT = "/home/zhuyuhan/project/C2KV/checkpoints/qwen3-4b/checkpoint-2000"
LOG_DIR = os.path.join(OUTPUT_DIR, "output")


# ---------------------------------------------------------------------------
# 1. 样本提取：每个 trace 只取最后一个 span
# ---------------------------------------------------------------------------

# 黑名单：finish/ 回复类 target，它们高度模板化，不利于评估压缩效果
BLACKLIST_TOOLS = {
    "finish", "mcp__environment__finish",
    "(text_reply)", "mcp__environment__message",
    "mcp__environment__transfer_to_human_agents",
    "TodoWrite",
}


def extract_samples(max_samples_per_benchmark: int = 30,
                    max_tooldef_tokens: int = None,
                    tokenizer=None) -> list[dict]:
    """
    每个 trace 从中间（20%~80% 时间范围）随机采样一个 span 作为测试点。
    筛选要求：
      - input 中包含 tool_call_response
      - target 不是 blacklist 中的模板化工具
      - (可选) tool definitions 的实际 token 数不超过 max_tooldef_tokens
    """
    files = sorted(glob.glob(os.path.join(DATA_DIR, "train-*.parquet")))
    print(f"Scanning {len(files)} parquet files for test samples...")

    samples = []
    seen_per_benchmark = defaultdict(int)
    random.seed(42)

    for f in files:
        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            bm = row["benchmark"]
            if seen_per_benchmark[bm] >= max_samples_per_benchmark:
                continue

            # 收集所有有效 span
            valid_spans = []
            for span in row["spans"]:
                attrs = span.get("attributes", {})
                tool_def_str = attrs.get("gen_ai.tool.definitions", "")
                input_str = attrs.get("gen_ai.input.messages", "")
                output_str = attrs.get("gen_ai.output.messages", "")
                if not (tool_def_str and input_str and output_str):
                    continue

                # 可选：筛掉长 tool definitions（基于实际 token 数）
                if max_tooldef_tokens is not None and tokenizer is not None:
                    td_tokens = len(tokenizer.encode(tool_def_str, add_special_tokens=True))
                    if td_tokens > max_tooldef_tokens:
                        continue

                try:
                    inputs = json.loads(input_str)
                    outputs = json.loads(output_str)
                except:
                    continue

                # 检查 target 是否是有效的工具调用（非 blacklist）
                target_name = None
                for msg in outputs:
                    for part in msg.get("parts", []):
                        if part.get("type") == "tool_call":
                            target_name = part.get("name", "")
                if target_name is None or target_name in BLACKLIST_TOOLS:
                    continue

                valid_spans.append({
                    "tool_definitions": tool_def_str,
                    "input_parsed": inputs,
                    "output_text": json.dumps(outputs, ensure_ascii=False),
                    "target_name": target_name,
                })

            if len(valid_spans) < 3:
                continue

            # 从中间范围 [20%, 80%] 随机选一个 span
            lo = max(0, int(0.2 * len(valid_spans)))
            hi = min(len(valid_spans), int(0.8 * len(valid_spans)))
            if lo >= hi:
                continue
            idx = random.randint(lo, hi - 1)
            chosen = valid_spans[idx]

            samples.append({
                "session_id": row["session_id"],
                "benchmark": bm,
                "tool_definitions": chosen["tool_definitions"],
                "input_parsed": chosen["input_parsed"],
                "output_text": chosen["output_text"],
            })
        if all(v >= max_samples_per_benchmark for v in seen_per_benchmark.values()):
            break

    print(f"Collected {len(samples)} samples (mid-trace random span, filtered).")
    return samples


# ---------------------------------------------------------------------------
# 2. 统一 Block 构造
# ---------------------------------------------------------------------------

def get_tokenized_blocks(sample: dict, tokenizer, truncate_ratio: int = None):
    """
    构造三个 Block：
      Block A: Tool Definitions（唯一被压缩/截断的部分）
      Block B: 完整对话历史（所有模式完全一致）
      Block C: Target Action

    Full 和 Truncation 将 A+B+C 拼接后输入 model.model。
    C2KV 将 A 单独 gist 压缩 → KV Cache，然后携带 cache 计算 B+C。
    """
    # ---- Block A: Tool Definitions（直接用原始字符串，不包装 system prompt）----
    tools_str = sample["tool_definitions"]
    ids_a = tokenizer.encode(tools_str, add_special_tokens=True)

    # Truncation: 按压缩比截断 Tool Definitions
    if truncate_ratio is not None and truncate_ratio > 1:
        keep_len = max(1, len(ids_a) // truncate_ratio)
        ids_a = ids_a[:keep_len]

    # ---- Block B: 完整对话历史 ----
    hist_text = ""
    for msg in sample.get("input_parsed", []):
        role = msg.get("role", "user")
        content = ""
        for part in msg.get("parts", []):
            if part.get("type") == "tool_call_response":
                content += json.dumps(part.get("result", []), ensure_ascii=False)
            elif part.get("type") == "tool_call":
                content += json.dumps(
                    {k: v for k, v in part.items() if k in ("name", "arguments")},
                    ensure_ascii=False)
            else:
                content += part.get("content", "")
        hist_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    ids_b = tokenizer.encode(hist_text, add_special_tokens=False)

    # ---- Block C: Target Action（仅提取 tool_call 的 name + arguments）----
    # 原始 output 是完整 JSON 数组 {"role":"assistant","parts":[...]}，
    # 直接序列化会引入大量 JSON 格式 token（{":[] 等），稀释 NLL 对核心语义的度量。
    # 改为只提取 tool_call 的 name + arguments。
    try:
        outputs = json.loads(sample["output_text"])
        tool_texts = []
        for msg in outputs:
            for part in msg.get("parts", []):
                if part.get("type") == "tool_call":
                    tc = {"name": part.get("name", ""),
                          "arguments": part.get("arguments", {})}
                    tool_texts.append(json.dumps(tc, ensure_ascii=False))
        if tool_texts:
            target_text = "\n".join(tool_texts)
        else:
            # 没有 tool_call 时回退到 text content
            text_parts = []
            for msg in outputs:
                for part in msg.get("parts", []):
                    if part.get("type") == "text":
                        text_parts.append(part.get("content", ""))
            target_text = " ".join(text_parts)
    except:
        target_text = sample["output_text"]

    text_c = "<|im_start|>assistant\n" + target_text + "<|im_end|>\n"
    ids_c = tokenizer.encode(text_c, add_special_tokens=False)

    return ids_a, ids_b, ids_c


# ---------------------------------------------------------------------------
# 3. Loss 与模型加载
# ---------------------------------------------------------------------------

def compute_nll_loss(logits, labels):
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    return loss_fct(logits, labels).item()


def dump_sample_log(sample: dict, tokenizer, mode: str, mode_result: dict, prefix: str = ""):
    """
    打印当前样本的详细 log 并写入 agent/output/ 目录。
    """
    import datetime
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%H%M%S")
    bm = sample["benchmark"]
    sid = sample["session_id"][:16]
    log_file = os.path.join(LOG_DIR, f"{prefix}sample_{bm}_{sid}_{mode}_{ts}.log")

    with open(log_file, "w", encoding="utf-8") as f:
        def w(line=""):
            print(line)
            f.write(line + "\n")

        w(f"{'='*70}")
        w(f"Mode: {mode}  |  Benchmark: {bm}  |  Session: {sample['session_id']}")
        w(f"{'='*70}")

        # Block A: Tool Defs (truncated preview)
        tools_str = sample["tool_definitions"]
        w(f"\n[Block A] Tool Definitions ({len(tools_str)} chars, raw string):")
        w(f"  First 500 chars: {tools_str[:500]}")
        w(f"  Last 200 chars:  ...{tools_str[-200:]}")

        # Block B: History messages
        msgs = sample.get("input_parsed", [])
        w(f"\n[Block B] History ({len(msgs)} messages):")
        for mi, m in enumerate(msgs):
            role = m.get("role", "?")
            for p in m.get("parts", []):
                pt = p.get("type", "?")
                if pt == "tool_call":
                    w(f"  msg[{mi}] role={role}  tool_call  name={p.get('name','?')}  args={json.dumps(p.get('arguments',{}), ensure_ascii=False)[:100]}")
                elif pt == "tool_call_response":
                    rstr = json.dumps(p.get("result",[]), ensure_ascii=False)
                    w(f"  msg[{mi}] role={role}  tool_call_response  len={len(rstr):<8}  preview={rstr[:120]}")
                else:
                    ct = p.get("content", "")
                    w(f"  msg[{mi}] role={role}  {pt:<20}  len={len(ct):<8}  preview={ct[:80]}")

        # Block C: Target
        try:
            outputs = json.loads(sample["output_text"])
            tool_texts = []
            for msg in outputs:
                for part in msg.get("parts", []):
                    if part.get("type") == "tool_call":
                        tool_texts.append(f"name={part.get('name','?')} args={json.dumps(part.get('arguments',{}), ensure_ascii=False)[:100]}")
            if tool_texts:
                target_display = " | ".join(tool_texts)
            else:
                text_parts = []
                for msg in outputs:
                    for part in msg.get("parts", []):
                        if part.get("type") == "text":
                            text_parts.append(part.get("content","")[:100])
                target_display = " | ".join(text_parts)
        except:
            target_display = sample["output_text"][:200]
        w(f"\n[Block C] Target Action (extracted): {target_display}")
        w(f"  Raw output_text ({len(sample['output_text'])} chars): {sample['output_text'][:200]}")

        # Result
        w(f"\n[Result] Loss: {mode_result.get('loss', 'N/A')}  PPL: {mode_result.get('ppl', 'N/A')}  "
          f"TargetTokens: {mode_result.get('target_tokens', 'N/A')}")
        if "compression_ratio" in mode_result:
            w(f"  CompRatio: {mode_result['compression_ratio']}x  "
              f"GistTokens: {mode_result.get('gist_tokens', 'N/A')}  "
              f"OrigTokens: {mode_result.get('original_tool_tokens', 'N/A')}")
        if "input_tokens" in mode_result:
            w(f"  InputTokens: {mode_result['input_tokens']}  Time: {mode_result.get('time_s', 'N/A')}s")
        w(f"{'='*70}")

    print(f"  Log saved to {log_file}")


def _get_device(model):
    return next(model.parameters()).device


def load_base_model(checkpoint: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, device_map=None,
        local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_c2kv_model(checkpoint: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    _, model_class = get_model_class(checkpoint, "qkv")
    model = model_class.from_pretrained(
        checkpoint, trust_remote_code=True,
        local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# 4. 三种评估模式
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_no_tools(tokenizer, model, sample: dict) -> dict:
    """No Tools Baseline: 完全去掉 Tool Definitions，只保留历史 + Target"""
    device = _get_device(model)
    ids_a, ids_b, ids_c = get_tokenized_blocks(sample, tokenizer)
    # 将 ids_a 置空（不输入任何工具定义）
    input_ids = torch.tensor([ids_b + ids_c], device=device)

    t0 = time.time()
    outputs = model.model(input_ids=input_ids, use_cache=False)
    gen_time = time.time() - t0

    L_b = len(ids_b)
    shift_hidden = outputs[0][0, L_b:-1, :].contiguous()
    shift_labels = torch.tensor(ids_c[1:], device=device).contiguous()
    loss_val = compute_nll_loss(model.lm_head(shift_hidden), shift_labels)

    return {
        "mode": "no_tools", "loss": round(loss_val, 4), "ppl": round(math.exp(loss_val), 4),
        "input_tokens": int(input_ids.shape[1]), "time_s": round(gen_time, 3),
        "target_tokens": len(shift_labels),
    }


@torch.inference_mode()
def run_full(tokenizer, model, sample: dict) -> dict:
    """Full Baseline: 原始 Tool Defs + 完整历史 + Target"""
    device = _get_device(model)
    ids_a, ids_b, ids_c = get_tokenized_blocks(sample, tokenizer)

    input_ids = torch.tensor([ids_a + ids_b + ids_c], device=device)

    t0 = time.time()
    outputs = model.model(input_ids=input_ids, use_cache=False)
    gen_time = time.time() - t0

    L_a, L_b = len(ids_a), len(ids_b)
    shift_hidden = outputs[0][0, L_a + L_b:-1, :].contiguous()
    shift_labels = torch.tensor(ids_c[1:], device=device).contiguous()
    loss_val = compute_nll_loss(model.lm_head(shift_hidden), shift_labels)

    return {
        "mode": "full", "loss": round(loss_val, 4), "ppl": round(math.exp(loss_val), 4),
        "input_tokens": int(input_ids.shape[1]), "time_s": round(gen_time, 3),
        "target_tokens": len(shift_labels),
    }


@torch.inference_mode()
def run_truncation(tokenizer, model, sample: dict, truncate_ratio: int) -> dict:
    """Truncation Baseline: Tool Defs 按 1/K 截断"""
    device = _get_device(model)
    ids_a, ids_b, ids_c = get_tokenized_blocks(sample, tokenizer,
                                                 truncate_ratio=truncate_ratio)

    input_ids = torch.tensor([ids_a + ids_b + ids_c], device=device)

    t0 = time.time()
    outputs = model.model(input_ids=input_ids, use_cache=False)
    gen_time = time.time() - t0

    L_a, L_b = len(ids_a), len(ids_b)
    shift_hidden = outputs[0][0, L_a + L_b:-1, :].contiguous()
    shift_labels = torch.tensor(ids_c[1:], device=device).contiguous()
    loss_val = compute_nll_loss(model.lm_head(shift_hidden), shift_labels)

    return {
        "mode": f"truncation_{truncate_ratio}x", "loss": round(loss_val, 4),
        "ppl": round(math.exp(loss_val), 4), "input_tokens": int(input_ids.shape[1]),
        "time_s": round(gen_time, 3), "target_tokens": len(shift_labels),
    }


@torch.inference_mode()
def run_c2kv(tokenizer, model, sample: dict, gist_ratio: int) -> dict:
    """C2KV 模式: 严格按时间线，只压缩 Tool Definitions"""
    device = _get_device(model)

    # ---- 1. 按时间线构建 Chunks ----
    chunks = []

    # Block A: Tool Definitions (标记为 tool_def 类型，走 gist 压缩)
    ids_a = tokenizer.encode(sample["tool_definitions"], add_special_tokens=True)
    chunks.append({"type": "tool_def", "ids": ids_a})

    # Block B: 完整对话历史（全部 normal，不做压缩）
    for msg in sample.get("input_parsed", []):
        role = msg.get("role", "user")
        chunks.append({"type": "normal", "ids": tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False)})
        for part in msg.get("parts", []):
            if part.get("type") == "tool_call_response":
                chunks.append({"type": "normal", "ids": tokenizer.encode(json.dumps(part.get("result", []), ensure_ascii=False), add_special_tokens=False)})
            elif part.get("type") == "tool_call":
                chunks.append({"type": "normal", "ids": tokenizer.encode(json.dumps({k: v for k, v in part.items() if k in ("name", "arguments")}, ensure_ascii=False), add_special_tokens=False)})
            else:
                chunks.append({"type": "normal", "ids": tokenizer.encode(part.get("content", ""), add_special_tokens=False)})
        chunks.append({"type": "normal", "ids": tokenizer.encode("<|im_end|>\n", add_special_tokens=False)})

    # Target
    try:
        outputs = json.loads(sample["output_text"])
        tool_texts = []
        for msg in outputs:
            for part in msg.get("parts", []):
                if part.get("type") == "tool_call":
                    tc = {"name": part.get("name", ""), "arguments": part.get("arguments", {})}
                    tool_texts.append(json.dumps(tc, ensure_ascii=False))
        target_text = "\n".join(tool_texts) if tool_texts else " ".join([p.get("content", "") for m in outputs for p in m.get("parts", []) if p.get("type") == "text"])
    except:
        target_text = sample["output_text"]
    ids_c = tokenizer.encode("<|im_start|>assistant\n" + target_text + "<|im_end|>\n", add_special_tokens=False)

    # ---- 2. 合并连续同类型 chunk + 收集信息 ----
    merged = []
    for ch in chunks:
        if not ch["ids"]:
            continue
        if not merged or merged[-1]["type"] != ch["type"]:
            merged.append({"type": ch["type"], "ids": ch["ids"][:]})
        else:
            merged[-1]["ids"].extend(ch["ids"])

    normal_ids_list = []   # [(ids, position_start)]
    tool_def_chunks = []   # [(ids, position_start)]
    pos = 0
    for ch in merged:
        ids = ch["ids"]
        if not ids:
            continue
        if ch["type"] == "normal":
            normal_ids_list.append((ids, pos))
            pos += len(ids)
        elif ch["type"] == "tool_def":
            tool_def_chunks.append((ids, pos))
            pos += len(ids)

    # ---- 3. forward：normal 合并一次，tool_def 逐个 gist ----
    t0 = time.time()

    # Normal: 所有 normal ids 拼成一条，一次 forward
    normal_concat = []
    for ids, _ in normal_ids_list:
        normal_concat.extend(ids)
    if normal_concat:
        tensor_n = torch.tensor([normal_concat], device=device)
        out_n = model.model(input_ids=tensor_n, use_cache=True)
        normal_cache = out_n.past_key_values
        del out_n
    else:
        normal_cache = None

    # Tool Defs: 按 2048 切分后逐个 generate_gist
    tool_caches = []
    for to_ids, pos_start in tool_def_chunks:
        if not to_ids:
            continue
        MAX_SEG = 2048
        for i in range(0, len(to_ids), MAX_SEG):
            sub_ids = to_ids[i:i + MAX_SEG]
            tensor_seg = torch.tensor([sub_ids], device=device)
            attn_seg = torch.ones_like(tensor_seg)
            model.model.config._attn_implementation = "sdpa"
            out_seg, gist_mask_seg, pos_ids_seg = model.model.generate_gist(
                tensor_seg, attn_seg, ratio=gist_ratio)
            pos_ids_seg = pos_ids_seg[:, -gist_mask_seg.shape[1]:]
            seg_cache, _ = blend_gist_key_values(
                model.config, [out_seg.past_key_values], [gist_mask_seg], [pos_ids_seg],
                model.model.rotary_emb, prefix_length=pos_start + i)
            tool_caches.append(seg_cache)
            del out_seg, seg_cache

    # 拼接 total cache = normal_kv + tool_def_gist_kv
    if normal_cache is not None:
        merged_cache = normal_cache
        for tc in tool_caches:
            for li in range(len(merged_cache.layers)):
                merged_cache.layers[li].keys = torch.cat([merged_cache.layers[li].keys, tc.layers[li].keys], dim=-2)
                merged_cache.layers[li].values = torch.cat([merged_cache.layers[li].values, tc.layers[li].values], dim=-2)
    elif tool_caches:
        merged_cache = tool_caches[0]
        for tc in tool_caches[1:]:
            for li in range(len(merged_cache.layers)):
                merged_cache.layers[li].keys = torch.cat([merged_cache.layers[li].keys, tc.layers[li].keys], dim=-2)
                merged_cache.layers[li].values = torch.cat([merged_cache.layers[li].values, tc.layers[li].values], dim=-2)
    else:
        return run_full(tokenizer, model, sample)

    gc.collect()
    torch.cuda.empty_cache()

    # ---- 4. 使用拼装好的 Cache 预测 Target ----
    tensor_c = torch.tensor([ids_c], device=device)
    pos_c = torch.arange(pos, pos + len(ids_c), dtype=torch.long, device=device).unsqueeze(0)

    outputs_tgt = model.model(
        input_ids=tensor_c, position_ids=pos_c, past_key_values=merged_cache, use_cache=False)
    gen_time = time.time() - t0

    shift_hidden = outputs_tgt[0][0, :-1, :].contiguous()
    shift_labels = torch.tensor(ids_c[1:], device=device).contiguous()
    loss_val = compute_nll_loss(model.lm_head(shift_hidden), shift_labels)

    gist_total = sum(int(tc.layers[0].keys.shape[-2]) for tc in tool_caches) if tool_caches else 0
    return {
        "mode": f"c2kv_{gist_ratio}x", "loss": round(loss_val, 4), "ppl": round(math.exp(loss_val), 4),
        "original_tool_tokens": len(ids_a), "gist_tokens": gist_total,
        "compression_ratio": round(len(ids_a) / max(gist_total, 1), 2),
        "time_s": round(gen_time, 3), "target_tokens": len(shift_labels),
    }

# ---------------------------------------------------------------------------
# 5. 主循环
# ---------------------------------------------------------------------------

def evaluate(args):
    # 如果需要按 token 数筛选，先加载 tokenizer 用于 extract_samples
    if args.max_tooldef_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    else:
        tokenizer = None

    samples = extract_samples(args.max_samples_per_benchmark,
                              max_tooldef_tokens=args.max_tooldef_tokens,
                              tokenizer=tokenizer)
    if args.max_examples:
        samples = samples[:args.max_examples]

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "npu" if hasattr(torch, "npu") and torch.npu.is_available()
        else "cpu")
    print(f"Using device: {device}")

    all_results = []

    # ---- No Tools (下限 baseline) ----
    if "no_tools" in args.modes or "no-tools" in args.modes:
        print("\n=== Running No Tools baseline ===")
        model, tokenizer = load_base_model(args.checkpoint, device)
        for s in tqdm(samples, desc="NoTools"):
            res = run_no_tools(tokenizer, model, s)
            res["benchmark"] = s["benchmark"]
            if args.log_samples:
                dump_sample_log(s, tokenizer, "no_tools", res, prefix=args.log_prefix)
            all_results.append(res)
        del model; gc.collect()
        torch.cuda.empty_cache()

    # ---- Full ----
    if "full" in args.modes:
        print("\n=== Running Full baseline ===")
        model, tokenizer = load_base_model(args.checkpoint, device)
        for s in tqdm(samples, desc="Full"):
            res = run_full(tokenizer, model, s)
            res["benchmark"] = s["benchmark"]
            if args.log_samples:
                dump_sample_log(s, tokenizer, "full", res, prefix=args.log_prefix)
            all_results.append(res)
        del model; gc.collect()
        torch.cuda.empty_cache()

    # ---- Truncation ----
    for r in args.truncation_ratios:
        mode_tag = f"truncation_{r}x"
        if mode_tag not in args.modes and "truncation" not in args.modes:
            continue
        print(f"\n=== Running Truncation ({r}x) ===")
        model, tokenizer = load_base_model(args.checkpoint, device)
        for s in tqdm(samples, desc=f"Truncation@{r}x"):
            res = run_truncation(tokenizer, model, s, r)
            res["benchmark"] = s["benchmark"]
            if args.log_samples:
                dump_sample_log(s, tokenizer, f"truncation_{r}x", res, prefix=args.log_prefix)
            all_results.append(res)
        del model; gc.collect()
        torch.cuda.empty_cache()

    # ---- C2KV ----
    for r in args.gist_ratios:
        mode_tag = f"c2kv_{r}x"
        if mode_tag not in args.modes and "c2kv" not in args.modes:
            continue
        print(f"\n=== Running C2KV (ratio={r}) ===")
        model, tokenizer = load_c2kv_model(args.checkpoint, device)
        for s in tqdm(samples, desc=f"C2KV@{r}x"):
            res = run_c2kv(tokenizer, model, s, r)
            res["benchmark"] = s["benchmark"]
            if args.log_samples:
                dump_sample_log(s, tokenizer, f"c2kv_{r}x", res, prefix=args.log_prefix)
            all_results.append(res)
        del model; gc.collect()
        torch.cuda.empty_cache()

    # ---- Summary ----
    groups = defaultdict(list)
    for r in all_results:
        groups[r["mode"]].append(r)

    print("\n" + "=" * 80)
    print(f"{'Mode':<20} {'Loss (↓)':<10} {'PPL (↓)':<10} {'CompRatio':<12} {'Time(s)':<10} {'Samples':<8}")
    print("-" * 80)
    for mode, group in sorted(groups.items()):
        # 加权平均：每个样本的 loss 按 target_tokens 加权
        total_weight = sum(r.get("target_tokens", 1) for r in group)
        weighted_loss = sum(r["loss"] * r.get("target_tokens", 1) for r in group) / total_weight
        # PPL 从加权 loss 重新计算
        weighted_ppl = math.exp(weighted_loss)
        avg_cr = float(np.mean([r.get("compression_ratio", 1.0) for r in group]))
        avg_time = float(np.mean([r["time_s"] for r in group]))
        cr_str = f"{avg_cr:.1f}x" if avg_cr > 1.0 else "N/A"
        print(f"{mode:<20} {weighted_loss:<10.4f} {weighted_ppl:<10.2f} {cr_str:<12} {avg_time:<10.3f} {len(group):<8}")
    print("=" * 80)

    # ---- 保存结果 ----
    results_path = os.path.join(OUTPUT_DIR, "agent_eval_tool_def_results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--max_samples_per_benchmark", type=int, default=10)
    parser.add_argument("--modes", type=str, nargs="+",
                        default=["full", "truncation", "c2kv"])
    parser.add_argument("--truncation_ratios", type=int, nargs="+",
                        default=[2, 4, 8])
    parser.add_argument("--gist_ratios", type=int, nargs="+",
                        default=[2, 4, 8])
    parser.add_argument("--max_tooldef_tokens", type=int, default=None,
                        help="Filter samples with tool definitions <= this many tokens (char-based: tokens*4)")
    parser.add_argument("--log_samples", action="store_true",
                        help="Dump detailed sample logs to agent/output/")
    parser.add_argument("--log_prefix", type=str, default="",
                        help="Optional prefix for log filenames")
    args = parser.parse_args()
    evaluate(args)
