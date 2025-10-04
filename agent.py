"""
Agent lokalny – NARZĘDZIOWY (Transformers/CPU, WSL)
====================================================
• Zero GPU wymagane. Działa na małych modelach 1B–3B (CPU). 7B ruszy, ale będzie ociężale.
• Styl: AGENT (cel → plan → kroki → wynik), ReAct‑like z wołaniem narzędzi.
• Pamięć: ./memory/state.json (historia) + ./memory/notes.md (notatki).
• KB/RAG: prosta wyszukiwarka TF‑IDF po ./kb/*.txt + bezpieczny odczyt plików.
• Narzędzia: search_kb, read_kb_file, list_kb, calc, save_note, http_get.

Szybki start (WSL/Ubuntu):
-------------------------
    python3 -m venv .venv && source .venv/bin/activate
    pip install --upgrade pip
    pip install "transformers>=4.44" torch numpy requests
    mkdir -p kb memory
    python agent.py

Zmienny model (polecane CPU):
    export AGENT_MODEL="google/gemma-2-2b-it"  # alternatywy: TinyLlama 1.1B, Llama-3.2-3B-Instruct, Qwen2.5-1.5B-Instruct

Wydajność CPU (opcjonalnie):
    export OMP_NUM_THREADS=$(nproc)
    export MKL_NUM_THREADS=$(nproc)

Uwaga: pierwsze uruchomienie pobierze i zbuforuje model.
"""

from __future__ import annotations
import os, re, json, time, math, glob, pathlib
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Tuple

import numpy as np
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID         = os.environ.get("AGENT_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MAX_NEW_TOKENS   = int(os.environ.get("AGENT_MAX_NEW_TOKENS", "256"))
TEMPERATURE      = float(os.environ.get("AGENT_TEMPERATURE", "0.1"))
TOP_P            = float(os.environ.get("AGENT_TOPP", "0.9"))
HISTORY_TURNS    = int(os.environ.get("AGENT_HISTORY_TURNS", "8"))
MAX_STEPS        = int(os.environ.get("AGENT_MAX_STEPS", "4"))
SEED             = int(os.environ.get("AGENT_SEED", "42"))
MAX_INPUT_TOKENS = int(os.environ.get("AGENT_MAX_INPUT_TOKENS", "384"))
MAX_HISTORY_TOKENS = int(os.environ.get("AGENT_MAX_HISTORY_TOKENS", "450"))

DATA_DIR  = pathlib.Path("./kb"); DATA_DIR.mkdir(exist_ok=True)
MEM_DIR   = pathlib.Path("./memory"); MEM_DIR.mkdir(exist_ok=True)
STATE_PATH = MEM_DIR/"state.json"
NOTES_PATH = MEM_DIR/"notes.md"

SYSTEM_ROLE = (
    "You are a decisive, goal-driven AGENT. You plan, call tools, and deliver a final result. "
    "Prefer short reasoning. If the user gives a goal (starts with 'cel:'), break it into steps."
)

TOOL_PROTOCOL = r"""
TOOL CALL PROTOCOL — JSON only
- To call a tool, respond with EXACTLY one JSON object: {"tool_name":"...","arguments":{...}}
- After a tool runs, you'll receive a TOOL_RESULT message. Use it and continue.
- When done, return: {"final_answer":"RESULT"}.
- Do not mix prose with tool JSON. Keep outputs strictly JSON during tool calls.
Tools & args:
  search_kb:    {"query": str, "k": int=5}
  list_kb:      {"pattern": str="*.txt"}
  read_kb_file: {"path": str, "max_chars": int=4000}
  calc:         {"expr": str}
  save_note:    {"text": str}
  http_get:     {"url": str, "timeout": int=10}
"""

def load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"scratch": {}, "history": []}

def save_state(state: Dict[str, Any]):
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

class MiniTfidf:
    def __init__(self, docs: List[str], paths: List[str]):
        self.paths = paths
        self.docs  = docs
        self.vocab: Dict[str,int] = {}
        self.df = None; self.tf = None; self.idf = None; self.doc_norms = None
        self._build()
    @staticmethod
    def _tok(s: str) -> List[str]:
        return re.findall(r"[\wąćęłńóśźż]+", s.lower())
    def _build(self):
        toks_per_doc = [self._tok(d) for d in self.docs]
        for toks in toks_per_doc:
            for t in toks:
                self.vocab.setdefault(t, len(self.vocab))
        V, D = len(self.vocab), len(toks_per_doc)
        self.tf = np.zeros((D, V), dtype=np.float32)
        self.df = np.zeros(V, dtype=np.float32)
        for i, toks in enumerate(toks_per_doc):
            counts: Dict[int,int] = {}
            for t in toks:
                j = self.vocab[t]
                counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                self.tf[i, j] = float(c)
                self.df[j]   += 1.0
        self.idf = np.log((1 + D) / (1 + self.df)) + 1
        self.tf *= self.idf
        self.doc_norms = np.linalg.norm(self.tf, axis=1) + 1e-8
    def search(self, query: str, k: int = 5) -> List[Tuple[float,str,str]]:
        toks = self._tok(query)
        q = np.zeros(len(self.vocab), dtype=np.float32)
        for t in toks:
            j = self.vocab.get(t)
            if j is not None:
                q[j] += self.idf[j]
        qn = np.linalg.norm(q) + 1e-8
        sims = (self.tf @ q) / (self.doc_norms * qn)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.paths[i], self.docs[i][:1200]) for i in idx]

def build_kb() -> Optional[MiniTfidf]:
    files = sorted(glob.glob(str(DATA_DIR/"*.txt")))
    if not files:
        return None
    docs, paths = [], []
    for p in files:
        try:
            txt = pathlib.Path(p).read_text(encoding="utf-8", errors="ignore")
            docs.append(txt); paths.append(p)
        except Exception:
            pass
    return MiniTfidf(docs, paths) if docs else None

@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[[Dict[str,Any]], Dict[str,Any]]

SAFE_CALC_RE = re.compile(r"^[0-9\.\+\-\*\/\(\)\^\%\s]+$")

def tool_calc(args: Dict[str,Any]) -> Dict[str,Any]:
    expr = str(args.get("expr", ""))
    if not SAFE_CALC_RE.match(expr):
        return {"error": "unsafe expression"}
    try:
        ns = {k:getattr(math,k) for k in dir(math) if not k.startswith("_")}
        ns["__builtins__"] = {}
        return {"result": eval(expr, ns, {})}
    except Exception as e:
        return {"error": f"calc failed: {e}"}

def tool_save_note(args: Dict[str,Any]) -> Dict[str,Any]:
    text = str(args.get("text", "")).strip()
    if not text:
        return {"error": "empty note"}
    NOTES_PATH.parent.mkdir(exist_ok=True)
    with open(NOTES_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}")
    return {"ok": True, "path": str(NOTES_PATH)}

def tool_search_kb(args: Dict[str,Any], kb: Optional[MiniTfidf]) -> Dict[str,Any]:
    if kb is None:
        return {"error": "KB empty. Put .txt files into ./kb"}
    q = str(args.get("query", "")).strip(); k = int(args.get("k", 5))
    res = kb.search(q, k)
    return {"results": [{"score":s, "path":p, "snippet":d[:800]} for s,p,d in res]}

def tool_list_kb(args: Dict[str,Any]) -> Dict[str,Any]:
    pattern = str(args.get("pattern", "*.txt"))
    files = [str(p) for p in DATA_DIR.glob(pattern)]
    return {"files": files}

def tool_read_kb_file(args: Dict[str,Any]) -> Dict[str,Any]:
    rel = str(args.get("path", "")); lim = int(args.get("max_chars", 4000))
    p = pathlib.Path(rel)
    if not p.is_absolute():
        p = (DATA_DIR / p.name).resolve()
    if DATA_DIR.resolve() not in p.parents and p.parent != DATA_DIR.resolve():
        return {"error": "access denied (outside ./kb)"}
    if not p.exists():
        return {"error": f"not found: {p}"}
    return {"path": str(p), "content": p.read_text(encoding="utf-8", errors="ignore")[:lim]}

def tool_http_get(args: Dict[str,Any]) -> Dict[str,Any]:
    url = str(args.get("url", "")); timeout = int(args.get("timeout", 10))
    if not (url.startswith("http://") or url.startswith("https://")):
        return {"error": "invalid url"}
    try:
        r = requests.get(url, timeout=timeout)
        return {"status": r.status_code, "headers": dict(r.headers), "text": r.text[:2000]}
    except Exception as e:
        return {"error": f"http failed: {e}"}

def build_tools(kb: Optional[MiniTfidf]) -> Dict[str,Tool]:
    return {
        "calc":        Tool("calc", "Safe arithmetic calculator", tool_calc),
        "save_note":   Tool("save_note", "Append a note to memory/notes.md", tool_save_note),
        "search_kb":   Tool("search_kb", "TF‑IDF search over ./kb", lambda a: tool_search_kb(a, kb)),
        "list_kb":     Tool("list_kb", "List files in ./kb", tool_list_kb),
        "read_kb_file":Tool("read_kb_file", "Read a file from ./kb (safe)", tool_read_kb_file),
        "http_get":    Tool("http_get", "Simple HTTP GET (webhooks / local services)", tool_http_get),
    }

def _seed_everything(seed: int):
    try:
        import random
        random.seed(seed)
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

_seed_everything(SEED)

def use_chat_template(tok: AutoTokenizer, messages: List[Dict[str,str]]):
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        buf = []
        for m in messages:
            buf.append(f"<|{m['role']}|> {m['content']}")
        buf.append("<|assistant|>")
        text = "\n".join(buf)
    # Przytnij wejście na poziomie tokenów, żeby nie rozkręcać RAM
    return tok(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)

def trim_history_by_tokens(tok: AutoTokenizer, messages: List[Dict[str,str]], max_tokens: int) -> List[Dict[str,str]]:
    """Przycina historię czatu tak, aby łączny prompt (system + historia) mieścił się w limicie tokenów.
    Zostawia najnowsze wiadomości; starsze streszcza krótką notką."""
    if not messages:
        return messages
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    other   = [m for m in messages if m.get("role") != "system"]

    def count_tokens(msgs: List[Dict[str,str]]) -> int:
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            buf = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                buf.append(f"<|{role}|> {content}")
            text = "\n".join(buf)
        encoding = tok(text)
        return len(encoding["input_ids"])

    kept: List[Dict[str,str]] = []
    kept.extend(sys_msgs)

    rev = list(reversed(other))
    rev_kept: List[Dict[str,str]] = []
    for m in rev:
        candidate = kept + list(reversed(rev_kept + [m]))
        if count_tokens(candidate) <= max_tokens:
            rev_kept.append(m)
        else:
            break

    kept = kept + list(reversed(rev_kept))
    if len(kept) < len(messages):
        summary_note = {"role":"system","content":"[summary] Older turns omitted for brevity; continue concisely based on latest context."}
        kept = sys_msgs + [summary_note] + [m for m in kept if m.get("role") != "system"]
    return kept

class LocalAgent:
    def __init__(self, model_id: str = MODEL_ID):
        print(f"[agent] loading model: {model_id}")
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        import torch
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        self.model.eval()
        # CPU hygiene
        try:
            torch.set_grad_enabled(False)
            torch.set_num_threads(max(1, os.cpu_count() - 1))
        except Exception:
            pass

        self.state = load_state()
        self.kb = build_kb()
        self.tools = build_tools(self.kb)
        self.stream_output = os.environ.get("AGENT_STREAM_STDOUT", "0") == "1"

    def _messages(self, user_msg: str) -> List[Dict[str,str]]:
        hist = self.state.get("history", [])[-HISTORY_TURNS:]
        msgs = [{"role":"system","content": SYSTEM_ROLE + "\n" + TOOL_PROTOCOL}]
        msgs += hist
        # Rolling window po tokenach – przytnij zanim ztokenizujesz do generacji
        msgs = trim_history_by_tokens(self.tok, msgs, MAX_HISTORY_TOKENS)
        return msgs

    def _generate_text(self, model_inputs):
        from transformers import TextIteratorStreamer
        import threading

        generation_kwargs = dict(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs.get("attention_mask"),
            max_new_tokens=min(MAX_NEW_TOKENS, 128),
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            use_cache=False,  # mniej pamięci na CPU
            repetition_penalty=1.05,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
        )

        if self.stream_output:
            streamer = TextIteratorStreamer(
                self.tok,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generation_kwargs["streamer"] = streamer

            thread = threading.Thread(
                target=self.model.generate,
                kwargs=generation_kwargs,
                daemon=True,
            )
            thread.start()

            collected: List[str] = []
            for text_piece in streamer:
                print(text_piece, end="", flush=True)
                collected.append(text_piece)

            thread.join()
            text = "".join(collected).strip()
            if text:
                return text

        out = self.model.generate(**generation_kwargs)
        return self.tok.decode(out[0], skip_special_tokens=True)

    @staticmethod
    def _extract_last_json(text: str) -> Optional[Dict[str,Any]]:
        s = text.strip()
        last_open = s.rfind('{')
        while last_open != -1:
            sub = s[last_open:]
            bal = 0
            for i,ch in enumerate(sub):
                if ch == '{': bal += 1
                elif ch == '}':
                    bal -= 1
                    if bal == 0:
                        try:
                            return json.loads(sub[:i+1])
                        except Exception:
                            break
            last_open = s.rfind('{', 0, max(0, last_open))
        return None

    def step(self, user_msg: str) -> str:
        self.state["history"].append({"role":"user","content": user_msg})
        current_msg = user_msg
        for _ in range(MAX_STEPS):
            msgs = self._messages(current_msg)
            model_inputs = use_chat_template(self.tok, msgs)
            txt = self._generate_text(model_inputs)
            obj = self._extract_last_json(txt)
            if not obj:
                self.state["history"].append({"role":"assistant","content": txt.strip()})
                save_state(self.state)
                return txt.strip()
            if "final_answer" in obj:
                final = str(obj["final_answer"]).strip()
                # odrzuć placeholdery – wewnątrz tego bloku
                if final in {"RESULT", "<concise result>"} or (final.startswith("<") and final.endswith(">")):
                    current_msg = "[tool result received] now give a concrete final_answer with actual content, not RESULT or <>."
                    continue
                self.state["history"].append({"role":"assistant","content": final})
                save_state(self.state)
                return final
            if "tool_name" in obj:
                name = str(obj.get("tool_name"))
                args = obj.get("arguments", {})
                tool = self.tools.get(name)
                if not tool:
                    tool_res = {"error": f"unknown tool: {name}"}
                else:
                    try:
                        tool_res = tool.handler(args)
                    except Exception as e:
                        tool_res = {"error": f"tool crash: {e}"}
                # Mapuj wynik narzędzia na assistant (chat templates często ignorują role custom)
                self.state["history"].append({"role":"assistant","content": "TOOL_RESULT(" + name + "): " + json.dumps(tool_res, ensure_ascii=False)})
                current_msg = "[tool result received] continue."
                continue
            # jeśli nic z powyższych – zapisz surowy tekst
            self.state["history"].append({"role":"assistant","content": txt})
            save_state(self.state)
            return txt
        fallback = "Task stopped after MAX_STEPS."
        self.state["history"].append({"role":"assistant","content": fallback})
        save_state(self.state)
        return fallback

def main():
    print("=== Local Tool-Using AGENT (Transformers/CPU) ===")
    print(f"Model: {MODEL_ID}")
    print("KB: ./kb | Memory: ./memory | Commands: exit/quit")
    agent = LocalAgent(MODEL_ID)
    while True:
        try:
            msg = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if msg.lower() in {"exit","quit"}:
            print("Bye.")
            break
        ans = agent.step(msg)
        print("\nAgent>", ans)

if __name__ == "__main__":
    main()
