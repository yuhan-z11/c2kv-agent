## 一、原始数据格式

数据来自 [agent-llm-traces](https://huggingface.co/datasets/exgentic/agent-llm-traces) 数据集，每条 **trace** 包含多个 **spans**（LLM 调用记录）。

## Trace Format

Each trace contains OpenTelemetry spans representing discrete operations in the LLM inference pipeline.

### Span Structure

```json
{
  "trace_id": "string",
  "span_id": "string",
  "parent_span_id": "string or null",
  "name": "string",
  "kind": "string",
  "start_time": "ISO 8601 timestamp",
  "end_time": "ISO 8601 timestamp",
  "attributes": {
    "gen_ai.operation.name": "chat",
    "gen_ai.request.model": "aws/claude-opus-4-5",
    "gen_ai.response.model": "aws/claude-opus-4-5",
    "gen_ai.usage.input_tokens": 18958,
    "gen_ai.usage.output_tokens": 92,
    "gen_ai.response.id": "chatcmpl-89052594-c0cd-49a4-a002-b3e75beca5f8",
    "gen_ai.response.finish_reasons": ["stop"],
    "gen_ai.input.messages": "[{\"role\": \"user\", \"parts\": [{\"type\": \"text\", \"content\": \"Complete this task...\"}]}]",
    "gen_ai.output.messages": "[{\"role\": \"assistant\", \"finish_reason\": \"stop\", \"parts\": [{\"type\": \"tool_call\", \"id\": \"tooluse_YQlkLGUuSk6Dp9WPwlmHzg\", \"name\": \"mcp__environment__bash\", \"arguments\": {...}}]}]",
    "gen_ai.tool.definitions": "[{\"type\": \"function\", \"name\": \"Task\", \"description\": \"...\", \"parameters\": {...}}]"
  },
  "resource_attributes": {
    "telemetry.sdk.language": "python",
    "telemetry.sdk.name": "opentelemetry",
    "telemetry.sdk.version": "1.40.0",
    "service.name": "litellm",
    "service.version": "1.0.0"
  },
  "status": {
    "code": 1,
    "message": ""
  }
}
```





### 主要的数据结构

```
trace.sessions
  ↓
spans[]                    # 多轮 LLM 调用记录，每个 span 一次 LLM 调用
  ↓
attributes:
  ├── gen_ai.tool.definitions: str   # JSON 工具定义列表
  ├── gen_ai.input.messages:  str    # JSON 对话历史（从第 1 轮到当前轮的完整累积）
  └── gen_ai.output.messages: str    # JSON 模型回复
```

**`gen_ai.input.messages` 结构**：

```json
[
  {"role": "user",      "parts": [{"type": "text", "content": "..."}]},
  {"role": "assistant", "parts": [
    {"type": "text",      "content": "Let me search..."},
    {"type": "tool_call", "name": "mcp__search", "arguments": {"query": "..."}}
  ]},
  {"role": "user",      "parts": [{"type": "tool_call_response", "result": [...]}]}
]
```

每个 span 的 `input.messages` 包含从第一轮到当前轮的**完整对话历史**（非增量）。消息模式为 `user text → assistant(tool_call) → user(tool_call_response)` 交替累积。

###  Part 类型

| type                 | 说明                    | 关键字段               |
| -------------------- | ----------------------- | ---------------------- |
| `text`               | 自然语言文本            | `content`              |
| `tool_call`          | LLM 发起的工具调用      | `name`, `arguments`    |
| `tool_call_response` | 工具执行返回结果        | `result`（JSON array） |
| `thinking`           | (browsecompplus) 思维链 | `content`, `signature` |

### 样本筛选策略

**当前做法**：每个 trace 从中间 [20%, 80%] 区间随机采样一个 span 作为测试点，并过滤掉finish工具：

```python
BLACKLIST_TOOLS = {
    "finish", "mcp__environment__finish",
    "(text_reply)", "mcp__environment__message",
    "mcp__environment__transfer_to_human_agents",
    "TodoWrite",
}
```

---

## 二、`tool_definiton_compression.py` — Tool Definitions 压缩

评估对 **Tool Definitions**（工具定义/函数签名）进行压缩的效果。

### 评估模式

#### Full Baseline

```
[ids_a + ids_b + ids_c] → model.model() → Loss on ids_c[1:]
```

- ids_a = tokenizer.encode(sample["tool_definitions"])，原始字符串
- ids_b = 完整对话历史（含所有 tool_call_response 原文）
- ids_c = target action（只取 tool_call 的 name+arguments，去除了 JSON 格式 token）

#### Truncation Baseline

```
ids_a' = ids_a[:len(ids_a)//K]
[ids_a' + ids_b + ids_c] → model.model() → Loss
```

#### C2KV 模式

```
流程：
  1. 按 2048 tokens 切分 ids_a 为多个 segment
  2. 每个 segment → generate_gist(ratio=K) → blend(prefix_length=累积原始长度)
  3. 所有 segment 的 gist KV 按层 torch.cat 拼接 → cache_gist
  4. Query = ids_b + ids_c
  5. position_ids 从 original_tool_tokens 起递增
  6. model.model(query, position_ids=..., past_key_values=cache_gist) → Loss
```

### 说明

- `ids_c` 只提取 `tool_call.name + arguments`，去除 JSON 格式 token
- 总结按 `target_tokens` 加权平均

---

## 三、`tool_output_compression.py` — Tool Output 压缩

评估对 **Tool Definitions** 保持不变，仅压缩 **tool_call_response**（工具返回结果）的效果。

### 多轮 Tool Output 压缩

**对测试 span 的完整历史中的所有 tool_call_response 做压缩**

由于 `input.messages` 包含从第一轮到当前轮的完整累积历史，遍历消息时会经过所有前序轮次的 `tool_call_response`，逐条压缩后按时间线拼入 KV Cache。这天然包含了"多轮"工具返回结果的压缩效果，无需单独写一个跨 spans 的脚本。

```
Span[N] 的完整累积历史:
  User指令 → Agent调工具1 → 工具1返回 → Agent调工具2 → 工具2返回 → ... → Agent当前调用
  
  time ──────────────────────────────────────────────────────────────────►

C2KV 逐块处理:
  [normal] [normal] [tool_output] [normal] [tool_output] [normal] ...
      │         │           │          │          │          │
      ▼         ▼           ▼          ▼          ▼          ▼
  forward  forward  generate_gist  forward  generate_gist  forward
                    blend(偏移)                blend(偏移)
```

### 时间线顺序

**正确的物理时间线**：

```
User指令 → Agent调用工具1 → 工具1返回结果 → Agent调用工具2 → 工具2返回结果 → ...
```

C2KV 不能把 tool output 提前塞到前面——每个 tool_call_response 在原文中出现的位置被 gist KV **原地替换**。

### 评估模式

#### Full Baseline

```
ids_a + ids_b_full + ids_c → model.model() → Loss
```

ids_b_full 包含完整的 tool_call_response 原文。

#### No Tool Outputs (no_tcr) Baseline

```
ids_a + ids_b_stripped + ids_c → model.model() → Loss
```

ids_b_stripped 中 tool_call_response 被完全移除（仅保留 text + tool_call 格式帧）。

#### Truncation Baseline

```
每个 tool_call_response 的 result 按 1/K 截断
ids_a + ids_b_truncated + ids_c → model.model() → Loss
```

#### C2KV 模式

```
流程：
  1. 将上下文拆分为 Chunk 队列（严格按时间线）
     - 遍历每条消息的每个 part：
       - text/tool_call → normal
       - tool_call_response → tool_output
  2. 合并连续 normal chunk 为一条序列
  3. 所有 normal chunk 合并为一条序列，1 次 forward → 得到正常 KV
  4. 每个 tool_output chunk → generate_gist → blend(prefix_length=累积)
  5. 所有 normal KV + gist KV 按层 torch.cat → merged_cache
  6. target (ids_c) → model.model(input=ids_c, position_ids, past_kv=merged_cache) → Loss
```

**设计细节**：

1. **Chunk 队列构建**：严格按时间线顺序遍历 input_parsed，每个 text/tool_call 标记为 normal，每个 tool_call_response 标记为 tool_output，连 `<|im_start|>role\n` 和 `<|im_end|\n>` 都作为单独的 normal chunk 保留。

2. **合并连续 normal chunk**：所有相邻的 normal chunk 拼接为一条序列，只做 **1 次 forward**（而非 N 次）。速度与 Full Baseline 接近。

3. **RoPE 对齐**：`blend_gist_key_values(prefix_length=original_accumulated)` 确保 gist token 继承原始文本中的绝对位置，不依赖 Cache 物理索引。

4. **不需要占位符**：不同于旧版用 `[Tool output compressed]` 替换历史文本，新版在 C2KV 的 forward 过程中直接从 token 级别处理——normal chunk 走正常 forward，tool_output chunk 走 gist 压缩并拼入 cache，query 只包含 target (ids_c)。

5. **多轮自然覆盖**：由于 input_parsed 包含完整累积历史，一次 span 内就能覆盖多轮 tool_call_response 的压缩效果，无需跨 spans 收集。

### 3.5 实验结果

**采样 100 条样本**（中间随机决策点）：

| Mode          | NLL (↓)    | PPL (↓) | CompRatio | Time(s) | vs Full |
| ------------- | ---------- | ------- | --------- | ------- | ------- |
| **Full**      | **0.1011** | 1.11    | —         | 16.06s  | —       |
| **C2KV 2x**   | 0.2449     | 1.28    | 2.0x      | 32.70s  | +0.1438 |
| **C2KV 4x**   | 0.2651     | 1.30    | 4.0x      | 32.67s  | +0.1640 |
| **C2KV 8x**   | 0.2816     | 1.33    | 7.8x      | 32.64s  | +0.1805 |
| Truncation 2x | 0.2696     | 1.31    | —         | 15.73s  | +0.1685 |
| Truncation 4x | 0.3222     | 1.38    | —         | 15.57s  | +0.2211 |
| Truncation 8x | 0.3295     | 1.39    | —         | 15.49s  | +0.2284 |

**分析**：

- C2KV 在所有压缩比下均优于 Truncation
- C2KV 随压缩比增加 NLL 上升（2x: 0.24 → 8x: 0.28），Truncation 上升更快（2x: 0.27 → 8x: 0.33）
- Full 的 NLL 非常低（0.10），因为 Target 只预测 `name+arguments`（~20 tokens），模式相对简单
- **C2KV 时间约为 Full 的 2 倍（32s vs 16s），主要开销在 generate_gist 的 eager attention**
