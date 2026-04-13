# Activation / Inference Input Alignment

## Problem

Activation extraction (`sae/extract_spv.py`) and inference (`util/inference_util.py`) must see the **exact same token sequence** for the activations to be meaningful. If the inputs diverge, the activations no longer reflect what the model processed during inference.

Currently both use `t.prompt` (the raw HumanEval-XL prompt), so they are consistent. However, this consistency is implicit and fragile — it breaks the moment inference switches to a chat API format, which prepends special tokens (`[INST]`, system tokens, etc.) that TransformerLens never sees.

## Root Cause

The chat endpoint (`/v1/chat/completions`) applies the model's chat template before tokenizing, producing a different token sequence than the raw prompt. TransformerLens in `extract_spv.py` tokenizes the raw prompt directly, so it would see a different input than what vLLM processed during inference.

The completions endpoint (`/v1/completions`) passes the prompt as-is, so raw prompt == what the model sees == what TransformerLens sees. This is why the current base model setup is consistent.

## Solution: Store `formatted_prompt` in the DB

Rather than having activation extraction independently reconstruct the input, inference should record exactly what it sent to the model. Activation extraction then reads that stored value instead of `t.prompt`.

### Schema change

Add a `formatted_prompt` column to `model_task_result`:

```sql
ALTER TABLE model_task_result ADD COLUMN formatted_prompt TEXT;
```

### Inference change (`util/inference_util.py`)

Before calling the API, construct the formatted prompt and store it:

**Base model (completions API):**
```python
formatted_prompt = problem.prompt  # no change
response = create_fn(model=model_str, prompt=formatted_prompt, ...)
# store formatted_prompt alongside result
```

**Instruct model (chat API):**
```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": problem.prompt},
]
formatted_prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
response = client.chat.completions.create(model=model_str, messages=messages, ...)
# store formatted_prompt alongside result
```

`SYSTEM_PROMPT` should be defined once in `.env` and used in both inference and activation extraction.

### Activation extraction change (`sae/extract_spv.py`)

Replace:
```python
problems = [(t.id, t.prompt) for t in tasks]
```

With a join that reads `formatted_prompt` from `model_task_result`:
```python
results = model_task_result_dao.get_all_for_model_split(model_name, split_id, is_test)
prompt_map = {r.task_id: r.formatted_prompt for r in results}
problems = [(t.id, prompt_map[t.id]) for t in tasks if t.id in prompt_map]
```

This makes activation extraction model-agnostic — it uses whatever was actually fed to the model during inference, with no knowledge of whether that was raw or chat-formatted.

## Base Model Compatibility

Using the chat API with a base model (e.g. `Llama-3.2-1B`, not `-Instruct`) would break it. Base models are not trained to follow chat templates — they have no concept of `[INST]` or system prompts. Sending chat-formatted input to a base model produces garbage output.

The completions API must be used with base models. The chat API should only be used with instruct models (`-Instruct` variants).

This means the `formatted_prompt` approach supports both cleanly:
- Base model + completions API → `formatted_prompt = raw prompt`
- Instruct model + chat API → `formatted_prompt = chat-templated string`

Activation extraction doesn't care which case it is — it just uses `formatted_prompt` either way.

## Tokenizer Consistency Caveat

This approach guarantees identical input strings. In practice this means identical token sequences, since both TransformerLens and vLLM use the same HuggingFace tokenizer for the same model. The one edge case where they could diverge is differing truncation behavior (`--max-model-len` in vLLM vs `truncate=True` in TransformerLens). Both should be set to the same max length to be safe.

## Implementation Checklist

- [ ] Add `formatted_prompt TEXT` column to `model_task_result` in `database_util.py`
- [ ] Add `formatted_prompt` field to `ModelTaskResult` dataclass in `model_task_result_dao.py`
- [ ] Update `_upsert` and `_map` in `model_task_result_dao.py` to include the new field
- [ ] Update `inference_util.py` to construct and store `formatted_prompt`
- [ ] Update `extract_spv.py` to read `formatted_prompt` from DB instead of `t.prompt`
- [ ] Add `SYSTEM_PROMPT` to `.env` (used when switching to instruct/chat API)
- [ ] Add `model_task_result_dao.get_formatted_prompts_for_model_split()` if needed for clean DAO access
