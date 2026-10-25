import subprocess
from pathlib import Path

def checkout(repo: str, commit: str, cache="~/.cache/astgraph_bench") -> Path:
    base = Path(cache).expanduser() / repo.replace("/", "__")
    if not base.exists():
        base.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", f"https://github.com/{repo}.git", str(base)], check=True)
    subprocess.run(["git", "-C", str(base), "checkout", "-f", "--quiet", commit], check=True)
    subprocess.run(["git", "-C", str(base), "clean", "-fdq"], check=True)
    return base
