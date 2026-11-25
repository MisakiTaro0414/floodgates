"""Locations used across the repository.  Override them with environment variables; the defaults assume the layout
    <parent>/floodgates                       this repository
    <parent>/looped_refs/loopspec             models/<name>: the base models; runs/: the LoopSpec harness runs (task-accuracy scoring only)
    <parent>/looped_refs/wavefront-decoding   the WaveFront code (WaveFront baseline only)
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
PARENT = os.path.dirname(ROOT)
LOOPSPEC = os.environ.get("LOOPSPEC_DIR", os.path.join(PARENT, "looped_refs", "loopspec"))
WAVEFRONT = os.environ.get("WAVEFRONT_DIR", os.path.join(PARENT, "looped_refs", "wavefront-decoding"))
RUNS = os.environ.get("FLOODGATES_RUNS", os.path.join(ROOT, "runs"))      # result files and drafter checkpoints
DATA = os.environ.get("FLOODGATES_DATA", os.path.join(ROOT, "data"))      # self-generated training data
TRACES = os.path.join(ROOT, "traces")                                      # benchmark prompts of the paper
PYTHON_EVAL = os.environ.get("FLOODGATES_PYTHON_EVAL", "python")          # interpreter with transformers and the evaluation harness (scoring)

def model_dir(path):
    """A model directory recorded in a drafter's args.json: the path itself when it exists, else MODEL_DIR, else the same
    model name under $LOOPSPEC_DIR/models/ (released drafters record the Hugging Face id of their base model)."""
    if os.path.isdir(path): return path
    if os.environ.get("MODEL_DIR"): return os.environ["MODEL_DIR"]
    return os.path.join(LOOPSPEC, "models", os.path.basename(path.rstrip("/")))

def trace(model, task):
    """The paper's prompts of a model (directory or name) on a task: traces/<model name>-<task>.jsonl."""
    return os.path.join(TRACES, f"{os.path.basename(str(model).rstrip('/'))}-{task}.jsonl")
