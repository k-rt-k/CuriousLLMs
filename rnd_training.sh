#!/usr/bin/zsh
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00 #hour-minute-second
#SBATCH --mem=48G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email at job start, end, and failure
#SBATCH --mail-user=hpoonia@andrew.cmu.edu # Replace with your email address
#SBATCH --output=/home/hpoonia/CuriousLLMs/logs/output-%j.out

conda init zsh
. ~/.zshrc
conda activate llm
cd /home/hpoonia/CuriousLLMs/
TOKENIZERS_PARALLELISM=true python math_train.py log_path="logs/rnd_training/Llama-3B"
