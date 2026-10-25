import json, subprocess, threading, time, urllib.request

class VramSampler:
    """Polls nvidia-smi; reports absolute peak (model may already be resident)."""
    def __init__(self, interval=0.1, gpu=0):
        self.interval, self.gpu, self.peak, self.base = interval, gpu, None, None
    def _read(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--id={self.gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL)
            return int(out.strip().splitlines()[0])
        except Exception:
            return None
    def _loop(self):
        while not self._stop:
            v = self._read()
            if v is not None and (self.peak is None or v > self.peak):
                self.peak = v
            time.sleep(self.interval)
    def __enter__(self):
        self._stop = False
        self.base = self.peak = self._read()
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
        return self
    def __exit__(self, *a):
        self._stop = True; self._t.join()

class OllamaLLM:
    def __init__(self, model, num_ctx=8192, host="http://localhost:11434", json_mode=True):
        self.model, self.num_ctx, self.host, self.json_mode = model, num_ctx, host, json_mode
    def generate(self, system, user, max_tokens=200):
        payload = {
            "model": self.model, "stream": False, "keep_alive": "10m",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx, "num_predict": max_tokens},
        }
        if self.json_mode:
            payload["format"] = "json"
        req = urllib.request.Request(self.host + "/api/chat", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        wall = (time.perf_counter() - t0) * 1000
        stats = dict(prompt_tokens=d.get("prompt_eval_count", 0), gen_tokens=d.get("eval_count", 0),
                     prefill_ms=d.get("prompt_eval_duration", 0) / 1e6,
                     decode_ms=d.get("eval_duration", 0) / 1e6,
                     total_ms=d.get("total_duration", 0) / 1e6, wall_ms=wall)
        return d["message"]["content"], stats

class MockLLM:
    """Plumbing test only: answers with the first file header in the prompt."""
    def generate(self, system, user, max_tokens=200):
        first = next((l.strip().strip('"').split('":')[0] for l in user.splitlines()
                      if l and not l.startswith((" ", "#", "REPO", "ISSUE", "CANDIDATE"))), "")
        first = first.replace("### ", "")
        txt = json.dumps({"files": [first], "symbol": ""})
        return txt, dict(prompt_tokens=len(user) // 4, gen_tokens=10, prefill_ms=1.0, decode_ms=1.0,
                         total_ms=2.0, wall_ms=2.0)
