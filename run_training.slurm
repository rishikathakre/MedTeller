#!/bin/bash
#SBATCH --job-name=medteller
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=msml612-fa25-class
#SBATCH --output=logs/medteller_%j.out
#SBATCH --error=logs/medteller_%j.err
#SBATCH --chdir=/home/rranka/scratch.msml612-fa25/medteller

# ============================================================================
# MedTeller Training - No venv, packages in scratch
# ============================================================================

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "======================================================================"

# Load PyTorch module
module purge
module load pytorch/2.0.1

# Set paths for packages and cache on scratch
export PYTHONPATH=$(pwd)/packages:$PYTHONPATH
export NLTK_DATA=$(pwd)/nltk_data
export HF_HOME=$(pwd)/hf_cache
export TRANSFORMERS_CACHE=$(pwd)/hf_cache
export HF_DATASETS_CACHE=$(pwd)/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Show GPU
nvidia-smi

# Create output directory
mkdir -p outputs

# Check data
echo ""
echo "Data:"
ls -la data/

# Run training
echo ""
echo "======================================================================"
echo "Starting training..."
echo "======================================================================"

python train.py \
    --data_dir "$(pwd)/data" \
    --output_dir "$(pwd)/outputs" \
    --experiment_name "medteller_${SLURM_JOB_ID}" \
    --epochs 30 \
    --batch_size 8 \
    --lr 5e-5 \
    --fp16 \
    --seed 42

echo ""
echo "======================================================================"
echo "Job finished: $(date)"
echo "======================================================================"
