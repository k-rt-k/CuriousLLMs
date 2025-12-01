import pickle
import time
from pathlib import Path
from datasets import load_dataset
import torch
import sys

# Add parent directory to path to import modules
sys.path.append('..')
from rnd import *

# Model name mapping
ENCODER_MODELS = {
    'granite': 'ibm-granite/granite-embedding-english-r2',
    'granite-small': 'ibm-granite/granite-embedding-small-english-r2',
    'gte': 'Alibaba-NLP/gte-modernbert-base'
}

def get_encoder_name(model_key):
    """Convert short model key to full model name."""
    if model_key in ENCODER_MODELS:
        return ENCODER_MODELS[model_key]
    return model_key

def precompute_embeddings_for_encoder(
    encoder_name: str,
    dataset,
    num_samples: int,
    max_length: int = 2048,
    batch_size: int = 64,
    device: str = 'cuda',
    save_dir: str = 'embeddings'
):
    """
    Pre-compute embeddings for all problems and solutions using a specific encoder.
    
    Args:
        encoder_name: Short name ('gte', 'granite', 'granite-small') or full model name
        dataset: HuggingFace dataset with 'question', 'r1_solution_1', 'r1_solution_2', 'r1_solution_3'
        num_samples: Number of samples to encode
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for encoding
        device: 'cuda' or 'cpu'
        save_dir: Directory to save embeddings
    
    Returns:
        dict: Contains 'problem_embs', 'solution1_embs', 'solution2_embs', 'solution3_embs',
              'embedding_dim', 'encoder_name', 'num_samples'
    """
    full_encoder_name = get_encoder_name(encoder_name)
    print(f"\n{'='*70}")
    print(f"Pre-computing embeddings for: {encoder_name} -> {full_encoder_name}")
    print(f"{'='*70}")
    
    # Initialize encoder
    print(f"Loading encoder...")
    encoder = Encoder(model_name=full_encoder_name, device=device, max_length=max_length)
    embedding_dim = encoder.embedding_dim
    print(f"✓ Encoder loaded (embedding dim: {embedding_dim})")
    
    # Prepare data
    problems = list(dataset['question'][:num_samples])
    solution1 = list(dataset['r1_solution_1'][:num_samples])
    solution2 = list(dataset['r1_solution_2'][:num_samples])
    solution3 = list(dataset['r1_solution_3'][:num_samples])
    
    print(f"\nEncoding {num_samples} samples in batches of {batch_size}...")
    start_time = time.time()
    
    # Encode in batches
    problem_embs = []
    solution1_embs = []
    solution2_embs = []
    solution3_embs = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        print("Encoding batch {}/{}...".format(batch_idx + 1, num_batches))
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Encode batch
        prob_batch = encoder.encode(problems[start_idx:end_idx])
        sol1_batch = encoder.encode(solution1[start_idx:end_idx])
        sol2_batch = encoder.encode(solution2[start_idx:end_idx])
        sol3_batch = encoder.encode(solution3[start_idx:end_idx])
        
        problem_embs.append(prob_batch.cpu())
        solution1_embs.append(sol1_batch.cpu())
        solution2_embs.append(sol2_batch.cpu())
        solution3_embs.append(sol3_batch.cpu())
        
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(f"  Processed {end_idx}/{num_samples} samples...")
    
    # Concatenate all batches
    problem_embs = torch.cat(problem_embs, dim=0)
    solution1_embs = torch.cat(solution1_embs, dim=0)
    solution2_embs = torch.cat(solution2_embs, dim=0)
    solution3_embs = torch.cat(solution3_embs, dim=0)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Encoding complete in {elapsed:.1f}s ({elapsed/num_samples:.3f}s per sample)")
    
    # Package results
    embeddings_data = {
        'problem_embs': problem_embs,
        'solution1_embs': solution1_embs,
        'solution2_embs': solution2_embs,
        'solution3_embs': solution3_embs,
        'embedding_dim': embedding_dim,
        'encoder_name': encoder_name,
        'encoder_full_name': full_encoder_name,
        'num_samples': num_samples,
        'max_length': max_length
    }
    
    # Save to disk
    Path(save_dir).mkdir(exist_ok=True)
    save_path = f"{save_dir}/embeddings_{encoder_name}_{num_samples}.pkl"
    
    print(f"\nSaving embeddings to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(embeddings_data, f)
    
    # Calculate file size
    file_size_mb = Path(save_path).stat().st_size / (1024 * 1024)
    print(f"✓ Saved {file_size_mb:.2f} MB to disk")
    
    print(f"\n📊 Embedding shapes:")
    print(f"  Problems: {problem_embs.shape}")
    print(f"  Solution 1: {solution1_embs.shape}")
    print(f"  Solution 2: {solution2_embs.shape}")
    print(f"  Solution 3: {solution3_embs.shape}")
    
    return embeddings_data

# Load dataset
print("Loading DeepMath-103K dataset...")
dataset = load_dataset("zwhe99/DeepMath-103K", split="train")
dataset = dataset.shuffle(seed=42)
print(f"✓ Dataset loaded: {len(dataset)} total samples\n")

# Configuration
NUM_SAMPLES = 12000  # Number of samples to encode
MAX_LENGTH = 512
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = 'embeddings'

print(f"Configuration:")
print(f"  Samples: {NUM_SAMPLES}")
print(f"  Max length: {MAX_LENGTH}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Device: {DEVICE}")
print(f"  Save directory: {SAVE_DIR}")

# Pre-compute embeddings for all 3 encoders
all_embeddings = {}

for encoder_key in ['gte']:
    embeddings = precompute_embeddings_for_encoder(
        encoder_name=encoder_key,
        dataset=dataset,
        num_samples=NUM_SAMPLES,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        save_dir=SAVE_DIR
    )
    all_embeddings[encoder_key] = embeddings
    print(f"\n{'='*70}\n")

print("\n🎉 All embeddings pre-computed and saved!")
print(f"\nSaved files:")
for encoder_key in all_embeddings.keys():
    filepath = f"{SAVE_DIR}/embeddings_{encoder_key}_{NUM_SAMPLES}.pkl"
    print(f"  {filepath}")