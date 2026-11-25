"""bench_ouro_baselines.py with engine_baselines_own.OuroOwnReadoutEngine (own-position readout refinement); same arguments,
envs and outputs (put _own in EVAL_TAG to tell the files apart)."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import os, runpy, sys
from floodgates.engines import engine_baselines
from floodgates.engines.engine_baselines_own import OuroOwnReadoutEngine
engine_baselines.OuroBaselineEngine = OuroOwnReadoutEngine
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_ouro_baselines.py"), run_name="__main__")
