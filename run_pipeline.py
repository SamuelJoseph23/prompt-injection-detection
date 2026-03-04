"""
Unified Pipeline for Ghost in the Machine
===========================================
Executes the entire research pipeline in sequence:
1. Data Preprocessing
2. Model Training (Siamese & Baseline)
3. Evaluation & Benchmarking
4. Notebook Execution & Figure Generation
5. Final Reporting (Dashboard & DOCX generation)

Usage::

    # Run full pipeline
    python run_pipeline.py

    # Run quick smoke test
    python run_pipeline.py --quick

    # Run specific stages
    python run_pipeline.py --stages preprocess train evaluate report
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(command: list[str], description: str) -> bool:
    """Execute a shell command and handle errors."""
    print(f"\n>>> Stage: {description}")
    print(f"    Running: {' '.join(command)}")
    
    start_time = time.time()
    try:
        # Use subprocess.run to execute the command
        # Capture output or let it stream to terminal
        result = subprocess.run(
            command,
            check=True,
            text=True,
            env={**dict(subprocess.os.environ), "PYTHONUTF8": "1"},
        )
        elapsed = time.time() - start_time
        print(f"<<< Completed {description} in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!! Error during {description}: {e}")
        return False
    except Exception as e:
        print(f"!!! Unexpected error during {description}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Ghost in the Machine - Unified Pipeline")
    parser.add_argument("--quick", action="store_true", help="Run in quick smoke-test mode")
    parser.add_argument("--stages", nargs="+", help="Specific stages to run (preprocess, train, evaluate, visualize, report)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Determine paths
    root = Path(__file__).parent.resolve()
    python_exe = sys.executable
    
    # Check if we are in a venv and if not, try to find one
    if "venv" not in python_exe.lower():
        potential_venv = root / "venv" / "Scripts" / "python.exe"
        if potential_venv.exists():
            python_exe = str(potential_venv)
            print(f"Using discovered venv at: {python_exe}")

    all_stages = ["preprocess", "train", "evaluate", "visualize", "report"]
    stages_to_run = args.stages if args.stages else all_stages
    
    print("=" * 60)
    print("  GHOST IN THE MACHINE - UNIFIED PIPELINE")
    print(f"  Mode: {'QUICK SMOKE TEST' if args.quick else 'FULL RESEARCH RUN'}")
    print("=" * 60)

    # 1. Preprocessing
    if "preprocess" in stages_to_run:
        cmd = [python_exe, "scripts/preprocess_data.py"]
        if not run_command(cmd, "Data Preprocessing"):
            sys.exit(1)

    # 2. Training
    if "train" in stages_to_run:
        cmd = [python_exe, "src/train.py", "--config", args.config]
        if args.quick:
            cmd.append("--quick")
        if not run_command(cmd, "Model Training"):
            sys.exit(1)

    # 3. Evaluation
    if "evaluate" in stages_to_run:
        cmd = [python_exe, "src/evaluate.py", "--config", args.config]
        if not run_command(cmd, "Model Evaluation"):
            sys.exit(1)

    # 4. Visualization (Notebook execution)
    if "visualize" in stages_to_run:
        nb_path = "notebooks/results_analysis.ipynb"
        # We use nbconvert to execute the notebook and save it in place
        cmd = [
            python_exe, "-m", "jupyter", "nbconvert", 
            "--to", "notebook", "--execute", 
            "--ExecutePreprocessor.timeout=600",
            nb_path, "--output", "results_analysis.ipynb"
        ]
        if not run_command(cmd, "Results Visualization (Notebook Execution)"):
            print("!!! Warning: Visualization failed. You may need to install jupyter/nbconvert.")
            # Don't exit here as core pipeline might be done
        else:
            print(f"\n    Notebook outputs and figures updated in: results/figures/")

    # 5. Final Reporting
    if "report" in stages_to_run:
        print("\n=== Generating Final Reports & Dashboards ===")
        # Dashboard
        cmd = [python_exe, "scripts/generate_final_dashboard.py"]
        run_command(cmd, "Final Results Dashboard Generation")
        
        # Original Paper DOCX
        cmd = [python_exe, "paper/convert_to_docx.py"]
        run_command(cmd, "Original Paper DOCX Generation")
        
        # Simplified Report DOCX
        cmd = [python_exe, "paper/convert_report_to_docx.py"]
        run_command(cmd, "Simplified Report DOCX Generation")

    print("\n" + "=" * 60)
    print("  PIPELINE EXECUTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
