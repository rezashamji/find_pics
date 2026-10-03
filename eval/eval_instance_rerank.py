"""Guarded entry point for `eval_instance.py <kind> rerank`: vLLM starts worker processes with 'spawn', which re-runs
the main module; eval_instance.py is a top-level script, so it must not BE the main module (10-03 failure:
"An attempt has been made to start a new process before the current process has finished its bootstrapping phase").
Usage (vLLM env, GPU): python eval/eval_instance_rerank.py <things|places|dogs|copies>"""
import runpy
import sys

if __name__ == "__main__":
    kind = sys.argv[1]
    sys.argv = ["eval/eval_instance.py", kind, "rerank"]
    runpy.run_path("eval/eval_instance.py", run_name="eval_instance")
