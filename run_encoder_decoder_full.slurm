#!/bin/bash
#SBATCH --job-name=medteller_enc_dec
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=msml612-fa25-class
#SBATCH --output=logs/enc_dec_%j.out
#SBATCH --error=logs/enc_dec_%j.err
#SBATCH --chdir=/scratch/zt1/project/msml612-fa25/user/rranka/medteller

# ============================================================================
# Encoder-Decoder Integration - Training & Evaluation
# Uses module system (no venv), packages in scratch
# ============================================================================

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "======================================================================"

# Load PyTorch module (includes torch, transformers, etc.)
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

echo ""
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Show GPU
echo ""
nvidia-smi

# Create directories
mkdir -p outputs
mkdir -p logs
mkdir -p outputs/plots
mkdir -p outputs/generations

# Check data files
echo ""
echo "Checking data files..."
ls -la data/

echo ""
echo "Checking for required files:"
[ -f "data/multimodal_dataset_full.pt" ] && echo "  ✓ multimodal_dataset_full.pt" || echo "  ✗ multimodal_dataset_full.pt NOT FOUND"
[ -d "data/decoder_tokenizer" ] && echo "  ✓ decoder_tokenizer/" || echo "  ✗ decoder_tokenizer/ NOT FOUND"
[ -d "data/decoder_pretrained" ] && echo "  ✓ decoder_pretrained/" || echo "  ✗ decoder_pretrained/ NOT FOUND"
[ -d "data/full_multimodal_decoder" ] && echo "  ✓ full_multimodal_decoder/" || echo "  ⚠ full_multimodal_decoder/ not found (will train from scratch)"

# Run training and evaluation
echo ""
echo "======================================================================"
echo "Starting encoder-decoder training and evaluation..."
echo "======================================================================"

# Python script is in code/ directory
python code/zaratan_encoder_decoder_full.py

echo ""
echo "======================================================================"
echo "Job finished: $(date)"
echo "======================================================================"
