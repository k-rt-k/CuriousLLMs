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
coeff_val=0.3
warmup=12
TOKENIZERS_PARALLELISM=true \
    python math_train.py target_layers=512,256,64 predictor_layers=1024,512,64 \
    dataset_schedule="e-h" curiosity_warmup_batches=$warmup \
    learning_rate=5e-6 use_rnd_curiosity=True wandb_name="oss-20B-deep-curriculum-$warmup" \
    log_path="logs/rnd_training/oss-20B-curric-wm$warmup-$coeff_val" model_name="openai/gpt-oss-20b" \
    curiosity_reward_coef=$coeff_val,-$coeff_val  behavior_if_log_dir_exists=resume
