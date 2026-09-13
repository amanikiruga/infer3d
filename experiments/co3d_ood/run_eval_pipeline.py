#!/usr/bin/env python3
"""
Automated evaluation pipeline orchestrator.

Takes checkpoint_dir as input and runs the full evaluation pipeline:
1. Generate splats (pseudo_gt, ours, baseline_ood)
2. Convert to point clouds via gs2mesh (3 parallel jobs)
3. Generate point cloud lists (3 parallel jobs)
4. Evaluate with ICP alignment (2 parallel jobs)

All output paths are automatically derived from checkpoint_dir.
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
from datetime import datetime

from infer3d import config as _cfg


class PipelineOrchestrator:
    def __init__(self, checkpoint_dir, output_base_dir=None, wandb_project=None,
                 skip_stages=None, dry_run=False, dataset_name="hydrants", pretrained_ckpt=None):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.dry_run = dry_run
        self.wandb_project = wandb_project
        self.skip_stages = set(skip_stages) if skip_stages else set()
        self.dataset_name = dataset_name
        self.pretrained_ckpt = pretrained_ckpt
        # Validate checkpoint_dir exists
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")

        

        # Derive output paths
        if output_base_dir:
            self.base_dir = Path(output_base_dir).resolve()
        else:
            self.base_dir = self.checkpoint_dir.parent / "eval_output"

        # make dir for output_base_dir if it doesn't exist
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)
        

        checkpoint_name = self.checkpoint_dir.name

        # Stage 1 outputs
        self.splats_dir = self.base_dir / checkpoint_name / "splats"
        self.pseudo_gt_plys = self.splats_dir / "pseudo_gt_plys"
        self.ours_plys = self.splats_dir / "ours_plys"
        self.baseline_ood_plys = self.splats_dir / "baseline_ood_plys"

        # Stage 2 outputs (gs2mesh)
        self.gs2mesh_output = self.base_dir / checkpoint_name / "gs2mesh_output"
        self.pseudo_gt_pcs = self.gs2mesh_output / "pseudo_gt_pointclouds"
        self.ours_pcs = self.gs2mesh_output / "ours_pointclouds"
        self.baseline_ood_pcs = self.gs2mesh_output / "baseline_ood_pointclouds"

        # Stage 3 outputs
        self.lists_dir = self.base_dir / checkpoint_name / "lists"
        self.pseudo_gt_list = self.lists_dir / "pseudo_gt.txt"
        self.ours_list = self.lists_dir / "ours.txt"
        self.baseline_ood_list = self.lists_dir / "baseline_ood.txt"

        # Stage 4 outputs
        self.results_dir = self.base_dir / checkpoint_name / "results"
        self.videos_dir = self.results_dir / "videos"
        self.ours_csv = self.results_dir / "ours_with_icp.csv"
        self.baseline_ood_csv = self.results_dir / "baseline_ood_with_icp.csv"

        # Logging
        self.log_dir = self.base_dir / checkpoint_name / "logs"
        self.pipeline_log = self.log_dir / "pipeline.log"

        # Constants
        self.gs2mesh_root = Path(_cfg.GS2MESH_ROOT)
        # The eval stages are this repo's own scripts, run with this interpreter --
        # they used to be invoked out of the external research repo with bare "python".
        self.eval_dir = Path(__file__).resolve().parent
        self.conda_env = "test2"

        # Timing
        self.stage_times = {}

    def create_directories(self):
        """Create all output directories."""
        dirs = [
            self.splats_dir, self.pseudo_gt_plys, self.ours_plys, self.baseline_ood_plys,
            self.gs2mesh_output, self.pseudo_gt_pcs, self.ours_pcs, self.baseline_ood_pcs,
            self.lists_dir, self.results_dir, self.videos_dir, self.log_dir
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        # Initialize log file
        with open(self.pipeline_log, 'w') as f:
            f.write(f"Pipeline started: {datetime.now()}\n")
            f.write(f"Checkpoint dir: {self.checkpoint_dir}\n")
            f.write(f"Output base dir: {self.base_dir}\n\n")

    def log(self, message):
        """Log message to both stdout and file."""
        print(message)
        with open(self.pipeline_log, 'a') as f:
            f.write(f"{message}\n")

    def print_stage_header(self, stage_num, stage_name):
        """Print formatted stage header."""
        header = f"\n{'='*80}\nSTAGE {stage_num}: {stage_name}\n{'='*80}"
        self.log(header)

    def run_command(self, cmd, log_file=None, shell=False):
        """Run command and capture output."""
        cmd_str = cmd if isinstance(cmd, str) else ' '.join(str(x) for x in cmd)
        self.log(f"Running: {cmd_str}")

        if self.dry_run:
            self.log("[DRY RUN] Command not executed")
            return 0

        # Open log file if provided
        if log_file:
            log_path = self.log_dir / log_file
            f_out = open(log_path, 'w')
        else:
            f_out = subprocess.PIPE

        try:
            if shell:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    executable='/bin/bash',
                    stdout=f_out if log_file else subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True
                )
            else:
                result = subprocess.run(
                    cmd,
                    stdout=f_out if log_file else subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True
                )

            if not log_file and result.stdout:
                self.log(result.stdout)

            return result.returncode

        except subprocess.CalledProcessError as e:
            self.log(f"ERROR: Command failed with return code {e.returncode}")
            if e.output:
                self.log(f"Output: {e.output}")
            raise

        finally:
            if log_file and f_out != subprocess.PIPE:
                f_out.close()

    def run_parallel_commands(self, commands, log_prefix):
        """Run multiple commands in parallel and wait for all to complete."""
        if self.dry_run:
            for i, cmd in enumerate(commands):
                cmd_str = cmd if isinstance(cmd, str) else ' '.join(str(x) for x in cmd)
                self.log(f"[DRY RUN] Parallel job {i+1}: {cmd_str}")
            return

        processes = []
        for i, cmd in enumerate(commands):
            cmd_str = cmd if isinstance(cmd, str) else ' '.join(str(x) for x in cmd)
            self.log(f"Starting parallel job {i+1}/{len(commands)}: {cmd_str}")

            log_file = self.log_dir / f"{log_prefix}_{i+1}.log"
            f_out = open(log_file, 'w')

            if isinstance(cmd, str):
                p = subprocess.Popen(
                    cmd,
                    shell=True,
                    executable='/bin/bash',
                    stdout=f_out,
                    stderr=subprocess.STDOUT
                )
            else:
                p = subprocess.Popen(
                    cmd,
                    stdout=f_out,
                    stderr=subprocess.STDOUT
                )

            processes.append((p, f_out, log_file))

        # Wait for all processes to complete
        self.log(f"Waiting for {len(processes)} parallel jobs to complete...")
        failed = []
        for i, (p, f_out, log_file) in enumerate(processes):
            returncode = p.wait()
            f_out.close()

            if returncode != 0:
                self.log(f"ERROR: Parallel job {i+1} failed with return code {returncode}")
                self.log(f"       Check log: {log_file}")
                failed.append((i+1, returncode))

        if failed:
            error_msg = f"Failed parallel jobs: {', '.join(f'job {j} (rc={rc})' for j, rc in failed)}"
            self.log(error_msg)
            raise RuntimeError(error_msg)

        self.log(f"All {len(processes)} parallel jobs completed successfully")

    def stage1_generate_splats(self):
        """Stage 1: Generate splats from checkpoints."""
        if 1 in self.skip_stages:
            self.log("Skipping Stage 1 (--skip_stages)")
            return

        self.print_stage_header(1, "Generate Splats")
        start_time = time.time()

        cmd = [
            sys.executable,
            str(self.eval_dir / "splats_to_pointcloud_via_mesh_batch_folder.py"),
            "--checkpoint_dir", str(self.checkpoint_dir),
            "--pseudo_gt_output_folder", str(self.pseudo_gt_plys),
            "--ours_output_folder", str(self.ours_plys),
            "--baseline_ood_output_folder", str(self.baseline_ood_plys),
            "--dataset_name", self.dataset_name,
            "--pretrained_ckpt", str(self.pretrained_ckpt),
        ]

        self.run_command(cmd, log_file="stage1_generate_splats.log")

        elapsed = time.time() - start_time
        self.stage_times['Stage 1'] = elapsed
        self.log(f"Stage 1 completed in {elapsed/60:.1f} minutes")

    def stage2_convert_to_pointclouds(self):
        """Stage 2: Convert PLYs to point clouds via gs2mesh (3 parallel jobs)."""
        if 2 in self.skip_stages:
            self.log("Skipping Stage 2 (--skip_stages)")
            return

        self.print_stage_header(2, "Convert to Point Clouds (gs2mesh) - 3 parallel jobs")
        start_time = time.time()

        methods = {
            "pseudo_gt": (self.pseudo_gt_plys, self.pseudo_gt_pcs),
            "ours": (self.ours_plys, self.ours_pcs),
            "baseline_ood": (self.baseline_ood_plys, self.baseline_ood_pcs),
        }

        commands = []
        for method, (input_folder, output_folder) in methods.items():
            cmd = f"""
cd {self.gs2mesh_root} && \
source ~/.bashrc && \
mamba activate {self.conda_env} && \
python run_batch_plys_to_pointclouds.py \
  --input_plys_folder {input_folder} \
  --output_folder {output_folder} \
  --colmap_name diffae_encoder_{method} \
  --dataset_name custom \
  --skip_video_extraction --skip_colmap --skip_GS \
  --GS_iterations 30000 --skip_masking --no-TSDF_use_occlusion_mask
""".strip()
            commands.append(cmd)

        self.run_parallel_commands(commands, log_prefix="stage2_gs2mesh")

        elapsed = time.time() - start_time
        self.stage_times['Stage 2'] = elapsed
        self.log(f"Stage 2 completed in {elapsed/60:.1f} minutes")

    def stage3_generate_lists(self):
        """Stage 3: Generate point cloud lists (3 parallel jobs)."""
        if 3 in self.skip_stages:
            self.log("Skipping Stage 3 (--skip_stages)")
            return

        self.print_stage_header(3, "Generate Point Cloud Lists - 3 parallel jobs")
        start_time = time.time()

        methods = {
            "pseudo_gt": (self.pseudo_gt_pcs, self.pseudo_gt_list),
            "ours": (self.ours_pcs, self.ours_list),
            "baseline_ood": (self.baseline_ood_pcs, self.baseline_ood_list),
        }

        commands = []
        for method, (input_folder, output_file) in methods.items():
            cmd = [
                sys.executable,
                str(self.eval_dir / "generate_pointcloud_lists_for_eval.py"),
                "--input_folder", str(input_folder),
                "--output_file", str(output_file),
            ]
            commands.append(cmd)

        self.run_parallel_commands(commands, log_prefix="stage3_lists")

        elapsed = time.time() - start_time
        self.stage_times['Stage 3'] = elapsed
        self.log(f"Stage 3 completed in {elapsed/60:.1f} minutes")

    def stage4_evaluate_with_icp(self):
        """Stage 4: Evaluate with ICP alignment (2 parallel jobs)."""
        if 4 in self.skip_stages:
            self.log("Skipping Stage 4 (--skip_stages)")
            return

        self.print_stage_header(4, "Evaluate with ICP Alignment - 2 parallel jobs")
        start_time = time.time()

        wandb_project = self.wandb_project or f"stylegan3-ours-co3d-{self.dataset_name}-eval"

        methods = {
            "ours": (self.ours_list, self.ours_csv),
            "baseline_ood": (self.baseline_ood_list, self.baseline_ood_csv),
        }

        commands = []
        for method, (pred_list, results_csv) in methods.items():
            cmd = [
                sys.executable,
                str(self.eval_dir / "eval_renderings_with_icp_batch.py"),
                "--checkpoint_dir", str(self.checkpoint_dir),
                "--pred_list", str(pred_list),
                "--gt_list", str(self.pseudo_gt_list),
                "--pred_method", method,
                "--results_csv", str(results_csv),
                "--num_points", "100000",
                "--max_iterations", "100",
                "--output_dir", str(self.videos_dir),
                "--wandb_project", wandb_project,
                "--wandb_run_name", f"{self.dataset_name}_{method}_rendering_eval_with_icp",
                "--dataset_name", self.dataset_name,
                "--pretrained_ckpt", str(self.pretrained_ckpt),
            ]
            commands.append(cmd)
        self.run_parallel_commands(commands, log_prefix="stage4_eval")

        elapsed = time.time() - start_time
        self.stage_times['Stage 4'] = elapsed
        self.log(f"Stage 4 completed in {elapsed/60:.1f} minutes")

    def print_summary(self, total_time):
        """Print final summary."""
        summary = f"""
{'='*80}
EVALUATION PIPELINE COMPLETE
{'='*80}
Checkpoint Dir: {self.checkpoint_dir}
Output Dir:     {self.base_dir / self.checkpoint_dir.name}

Results:
  - Splats:        {self.splats_dir}
  - Point Clouds:  {self.gs2mesh_output}
  - Lists:         {self.lists_dir}
  - CSVs:          {self.results_dir}
  - Videos:        {self.videos_dir}
  - Logs:          {self.log_dir}

CSV Files:
  - Ours:          {self.ours_csv}
  - Baseline OOD:  {self.baseline_ood_csv}

Stage Timings:
"""
        for stage, elapsed in self.stage_times.items():
            summary += f"  - {stage}: {elapsed/60:.1f} minutes\n"

        summary += f"\nTotal time: {total_time/60:.1f} minutes\n"
        summary += "="*80

        self.log(summary)

    def run(self):
        """Run the full pipeline."""
        overall_start = time.time()

        try:
            self.create_directories()

            self.log(f"\n{'='*80}")
            self.log("STARTING EVALUATION PIPELINE")
            self.log(f"{'='*80}")
            self.log(f"Checkpoint Dir: {self.checkpoint_dir}")
            self.log(f"Output Base Dir: {self.base_dir}")
            self.log(f"Dry Run: {self.dry_run}")
            if self.skip_stages:
                self.log(f"Skipping Stages: {sorted(self.skip_stages)}")

            # Run stages sequentially
            self.stage1_generate_splats()
            self.stage2_convert_to_pointclouds()
            self.stage3_generate_lists()
            self.stage4_evaluate_with_icp()

            # Print summary
            total_time = time.time() - overall_start
            self.print_summary(total_time)

            return 0

        except Exception as e:
            self.log(f"\n{'='*80}")
            self.log("PIPELINE FAILED")
            self.log(f"{'='*80}")
            self.log(f"Error: {e}")
            self.log(f"Check logs in: {self.log_dir}")
            return 1


def main():
    parser = argparse.ArgumentParser(
        description="Automated evaluation pipeline from checkpoint_dir",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_eval_pipeline.py \\
    --checkpoint_dir /path/to/checkpoints

  python run_eval_pipeline.py \\
    --checkpoint_dir /path/to/checkpoints \\
    --output_base_dir /custom/output \\
    --wandb_project "my-project" \\
    --dry_run
        """
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing checkpoint subdirectories"
    )

    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=None,
        help="Base directory for all outputs (default: {checkpoint_dir}/../eval_output)"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project name (default: stylegan3-ours-co3d-category-name-eval)"
    )

    parser.add_argument(
        "--skip_stages",
        type=str,
        default=None,
        help="Comma-separated list of stages to skip (e.g., '1,2')"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str,
        default="hydrants",
        help="Dataset name (default: hydrants)"
    )

    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default=_cfg.LIFTER_CKPTS["hydrants"],
        help="Pretrained checkpoint path (required for evaluation)"
    )

    args = parser.parse_args()
    
    if args.pretrained_ckpt is None:
        raise ValueError("Pretrained checkpoint path is required for evaluation")

    # Parse skip_stages
    skip_stages = None
    if args.skip_stages:
        skip_stages = [int(s.strip()) for s in args.skip_stages.split(',')]

    # Create and run orchestrator
    orchestrator = PipelineOrchestrator(
        checkpoint_dir=args.checkpoint_dir,
        output_base_dir=args.output_base_dir,
        wandb_project=args.wandb_project,
        skip_stages=skip_stages,
        dry_run=args.dry_run,
        dataset_name=args.dataset_name,
        pretrained_ckpt=args.pretrained_ckpt
    )

    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())
