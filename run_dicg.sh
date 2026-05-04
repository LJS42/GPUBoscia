#!/bin/bash
#SBATCH --job-name=dicg_cube_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=/home/htc/jili/SCRATCH/bnb_fw_tests/%j.log
#SBATCH --error=/home/htc/jili/SCRATCH/bnb_fw_tests/%j.err

# ── 0. Setup ──────────────────────────────────────────────
mkdir -p "/home/htc/jili/SCRATCH/bnb_fw_tests"
echo "════════════════════════════════════════"
echo "Job started: $(date)"
echo "Node:        $(hostname)"
echo "User:        $USER"
echo "Job ID:      $SLURM_JOB_ID"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "════════════════════════════════════════"

# ── 1. Internet proxy ─────────────────────────────────────
export https_proxy=http://squid.zib.de:3128
export http_proxy=http://squid.zib.de:3128

# ── 2. Load profile ───────────────────────────────────────
source ~/.profile
source ~/.bashrc
conda activate julia_cuda
export CUDA_HOME=/scratch/htc/jili/conda/envs/julia_cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ── 3. GPU info ───────────────────────────────────────────
echo ""
echo "── nvidia-smi ──────────────────────────"
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader
echo "────────────────────────────────────────"

# ── 4. Navigate to project ────────────────────────────────
PROJECT_DIR="/home/htc/jili/project/Boscia_FWonGPU.jl"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
fi
cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"

# ── 5. Instantiate packages ───────────────────────────────
echo ""
echo "── Instantiating Julia project ─────────"
julia --project=. -e "
    using Pkg
    Pkg.instantiate()
    Pkg.status()
"

# ── 6. Setting up CUDA ───────────────────────────────────
echo "── Setting up CUDA ──────────────────────"
julia --project=. -e "
    using CUDA
    CUDA.set_runtime_version!(v\"12.6\")
"
julia --project=. -e "
    using CUDA
    println(\"CUDA functional: \", CUDA.functional())
    if CUDA.functional()
        println(\"Device: \", CUDA.name(CUDA.device()))
        println(\"Memory: \", CUDA.totalmem(CUDA.device()) ÷ 1024^3, \" GB\")
    end
"

# ── 7. Run DICG_Cube ─────────────────────────────────────
echo ""
echo "── Running DICG_Cube.jl ─────────────────"
julia --project=. \
      --threads=4 \
      examples/DICG_Cube.jl
TEST_EXIT=$?

echo ""
echo "════════════════════════════════════════"
if [ $TEST_EXIT -eq 0 ]; then
    echo "✓ DICG_Cube PASSED (exit 0)"
else
    echo "✗ DICG_Cube FAILED (exit $TEST_EXIT)"
fi
echo "Job finished: $(date)"
echo "════════════════════════════════════════"
exit $TEST_EXIT
