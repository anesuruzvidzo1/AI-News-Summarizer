import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# assume venv is at PROJECT_ROOT/venv
VENV_PY = PROJECT_ROOT / "venv" / "bin" / "python"
if not VENV_PY.exists():
    # fallback to system python if venv not found 
    VENV_PY = Path(sys.executable)

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Ordered steps and their script filenames 
STEP_SCRIPTS = [
    ("fetch", "fetch_news.py"),
    ("fetch", "fetch_and_store.py"),           
    ("preprocess", "preprocess_news.py"),
    ("summarize", "summarize_extractive.py"),  
    ("train", "train_model.py"),               # run only with --retrain
    ("classify", "newsClassifier.py"),
]

def run_script(python_exe: Path, script_path: Path, timeout: int = None, verbose: bool = False):
    """
    Run a single script using the given python executable.
    Returns (ok:bool, stdout:str, stderr:str)
    """
    cmd = [str(python_exe), str(script_path)]
    env = os.environ.copy()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(PROJECT_ROOT), env=env)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return False, out.decode(errors="replace"), f"TIMEOUT after {timeout}s\n" + err.decode(errors="replace")

    ok = (proc.returncode == 0)
    sout = out.decode(errors="replace")
    serr = err.decode(errors="replace")
    if verbose:
        print(f"--- OUTPUT for {script_path.name} ---")
        print(sout)
        if serr:
            print(f"--- STDERR for {script_path.name} ---")
            print(serr)
    return ok, sout, serr

def pipeline_run(retrain: bool=False, timeout: int=600, verbose: bool=False):
    """
    Run the pipeline steps in order. Steps whose script files don't exist are skipped.
    Returns a dict with step status and outputs.
    """
    results = {}
    started = datetime.utcnow().isoformat()
    results["started_at"] = started
    print(f"[pipeline] started at {started}")
    last_ok = True

    for key, fname in STEP_SCRIPTS:
        # skip train unless retrain True
        if key == "train" and not retrain:
            print(f"[pipeline] skipping step '{key}' (use --retrain to enable)")
            results[key] = {"skipped": True}
            continue

        script_path = SCRIPTS_DIR / fname
        if not script_path.exists():
            print(f"[pipeline] script not found, skipping step '{key}': {script_path.name}")
            results[key] = {"skipped": True}
            continue

        print(f"[pipeline] running step '{key}': {script_path.name}")
        ok, out, err = run_script(VENV_PY, script_path, timeout=timeout, verbose=verbose)
        results[key] = {"ok": ok, "stdout": out, "stderr": err}
        if not ok:
            print(f"[pipeline] step '{key}' FAILED.")
            last_ok = False
            # For fetch or preprocess failure, we stop further steps.
            if key in ("fetch", "preprocess", "train"):
                print("[pipeline] stopping pipeline due to failure in an earlier step.")
                break
            # if classify failed, we keep last_ok False but finished.
        else:
            print(f"[pipeline] step '{key}' completed successfully.")

    finished = datetime.utcnow().isoformat()
    results["finished_at"] = finished
    print(f"[pipeline] finished at {finished}")
    return results, last_ok

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="run train_model.py before classification")
    parser.add_argument("--timeout", type=int, default=600, help="timeout per step in seconds")
    parser.add_argument("--verbose", action="store_true", help="print stdout/stderr of each step")
    args = parser.parse_args()

    res, last_ok = pipeline_run(retrain=args.retrain, timeout=args.timeout, verbose=args.verbose)

    # Print brief summary
    print("\n=== Pipeline summary ===")
    for k, v in res.items():
        if k in ("started_at", "finished_at"):
            print(f"{k}: {v}")
            continue
        if v.get("skipped"):
            print(f"{k}: SKIPPED")
        else:
            print(f"{k}: ok={v.get('ok', False)}, stdout_len={len(v.get('stdout',''))}, stderr_len={len(v.get('stderr',''))}")

    # exit 0 if classification succeeded (last_ok True), else exit 1
    sys.exit(0 if last_ok else 1)

if __name__ == "__main__":
    main()