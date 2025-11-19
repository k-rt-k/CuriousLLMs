#!/usr/bin/zsh
#SBATCH --partition=cpu
#SBATCH --time=15:00:00 #hour-minute-second
#SBATCH --mem=8G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email at job start, end, and failure
#SBATCH --mail-user=hpoonia@andrew.cmu.edu # Replace with your email address
#SBATCH --output=/home/hpoonia/CuriousLLMs/logs/output-%j.out

conda init zsh
. ~/.zshrc
conda activate llm
cd /home/hpoonia/CuriousLLMs/tinker-cookbook/tinker_cookbook/recipes/math_rl
python train.py
