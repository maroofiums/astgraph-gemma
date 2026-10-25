import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bench_smoke_script():
    """Runs the end-to-end benchmark plumbing (mock LLM, resume, report) from the original v3 smoke test."""
    env = {"PYTHONPATH": f"{ROOT / 'src'}:{ROOT}", "PATH": "/usr/bin:/usr/local/bin"}
    import os
    env = {**os.environ, **env}
    r = subprocess.run([sys.executable, "tests/smoke_bench.py"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert "SMOKE OK" in r.stdout
