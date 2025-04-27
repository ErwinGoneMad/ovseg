#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=out.log
#SBATCH --partition=ird_gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=250G





# Validation fold:
# - Use 0-4 for 5-fold cross validation (80% training, 20% validation)
# - Use 5 or above for training on 100% of the data (no validation)
VALIDATION_FOLD=4

# Path to save logs
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"


# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "Running on: $SLURM_NODELIST"

# Preprocess the data
echo "Started prepo"
python mypreprocess.py 

echo "Started training"
# Run training
python mytraining.py
