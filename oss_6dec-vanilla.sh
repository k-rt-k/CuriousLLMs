#!/usr/bin/zsh
#SBATCH --partition=general
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --time=16:00:00 #hour-minute-second
#SBATCH --mem=48G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email at job start, end, and failure
#SBATCH --mail-user=ksnair@andrew.cmu.edu # Replace with your email address
#SBATCH --output=/home/ksnair/CuriousLLMs/logs/output-%A_%a.out


source ~/micromamba/etc/profile.d/mamba.sh
micromamba activate llm
cd /home/ksnair/CuriousLLMs/
source .env
warmup=12
TOKENIZERS_PARALLELISM=true \
    python math_train.py \
    dataset_schedule="e-h" curiosity_warmup_batches=$warmup \
    learning_rate=5e-6 use_rnd_curiosity=False wandb_name="oss-20B-vanilla-curriculum-$warmup" \
    log_path="logs/rnd_training/oss-20B-vanilla-curric-wm$warmup" model_name="openai/gpt-oss-20b" \
    behavior_if_log_dir_exists=resume