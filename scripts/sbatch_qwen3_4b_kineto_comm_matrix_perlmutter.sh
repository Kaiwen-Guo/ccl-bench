#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --qos=interactive
#SBATCH --time=02:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --account=m4999
#SBATCH --job-name=qwen4b-kineto
#SBATCH --output=/pscratch/sd/k/kg597/ccl-bench-traces/qwen3-4b-kineto-comm-matrix/slurm-%j.out
#SBATCH --error=/pscratch/sd/k/kg597/ccl-bench-traces/qwen3-4b-kineto-comm-matrix/slurm-%j.err

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /pscratch/sd/k/kg597/session1/cvllm

cd "$HOME/ccl-bench"
mkdir -p /pscratch/sd/k/kg597/ccl-bench-traces/qwen3-4b-kineto-comm-matrix

bash scripts/run_qwen3_4b_kineto_comm_matrix_perlmutter.sh
bash scripts/make_qwen3_4b_kineto_comm_bundles_perlmutter.sh
