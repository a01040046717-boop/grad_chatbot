"""
무료 오픈소스 LLM (API 키 불필요)

Hugging Face 의 공개 모델(Qwen2.5-Instruct)을 내려받아 직접 실행한다.
OpenAI 클라이언트와 같은 모양(client.chat.completions.create)을 제공하므로
rag/llm.py 의 RAG 답변·도구 호출(tool calling) 코드를 그대로 사용할 수 있다.

  - GPU 있음(Colab T4 등): Qwen/Qwen2.5-3B-Instruct  (약 6GB, 답변 수 초~수십 초)
  - GPU 없음            : Qwen/Qwen2.5-1.5B-Instruct (CPU, 느림)
  - LOCAL_MODEL 환경변수로 다른 모델 지정 가능
"""
from __future__ import annotations

import json
import os
import re
import uuid
from types import SimpleNamespace

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
END_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")


class _ToolCall:
    def __init__(self, name: str, arguments: dict):
        self.id = "call_" + uuid.uuid4().hex[:8]
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments, ensure_ascii=False))

    def model_dump(self) -> dict:
        return {"id": self.id, "type": "function",
                "function": {"name": self.function.name, "arguments": self.function.arguments}}


def parse_output(text: str) -> SimpleNamespace:
    """모델 출력에서 <tool_call>{...}</tool_call> 을 찾아 OpenAI 형식 메시지로 바꾼다."""
    for t in END_TOKENS:
        text = text.split(t)[0]
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if obj.get("name"):
            calls.append(_ToolCall(obj["name"], args or {}))
    content = TOOL_CALL_RE.sub("", text).strip()
    return SimpleNamespace(content=content, tool_calls=calls or None)


def to_template_messages(messages: list[dict]) -> list[dict]:
    """OpenAI 형식 대화 → Hugging Face chat template 형식 (도구 호출 인자는 dict 로)."""
    out = []
    for m in messages:
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = tc["function"] if isinstance(tc, dict) else tc.function
                name = fn["name"] if isinstance(fn, dict) else fn.name
                args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tcs.append({"type": "function", "function": {"name": name, "arguments": args}})
            out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs})
        elif role == "tool":
            out.append({"role": "tool", "content": m["content"]})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    return out


class LocalLLMClient:
    """OpenAI 클라이언트와 같은 인터페이스의 로컬 모델 래퍼."""
    is_local = True

    def __init__(self, model_name: str | None = None, max_new_tokens: int = 700, _tokenizer=None, _model=None):
        if _tokenizer is not None:            # 테스트용 주입
            self.tok, self.model, self.model_name = _tokenizer, _model, model_name or "test-model"
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            gpu = torch.cuda.is_available()
            self.model_name = model_name or os.getenv("LOCAL_MODEL") or (
                "Qwen/Qwen2.5-3B-Instruct" if gpu else "Qwen/Qwen2.5-1.5B-Instruct")
            print(f"[local-llm] {self.model_name} 불러오는 중… (처음 한 번은 내려받기에 몇 분 걸릴 수 있어요)")
            self.tok = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16 if gpu else torch.float32,
                device_map="auto" if gpu else None)
            self.model.eval()
            print(f"[local-llm] 준비 완료 ({'GPU' if gpu else 'CPU'})")
        self.max_new_tokens = max_new_tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model=None, messages=None, tools=None, temperature=0.1, **kw):
        conv = to_template_messages(messages or [])
        prompt = self.tok.apply_chat_template(conv, tools=tools, add_generation_prompt=True, tokenize=False)
        inputs = self.tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        n_in = inputs["input_ids"].shape[1]
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  repetition_penalty=1.05, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][n_in:], skip_special_tokens=False)
        msg = parse_output(text)
        if not tools:
            msg.tool_calls = None
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
