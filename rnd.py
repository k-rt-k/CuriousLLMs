"""rnd.py

Lightweight utilities for semantic novelty estimation using Random Network
Distillation (RND). This file provides three main components:

- RunningMeanStd: Online running mean/variance tracker for reward normalization.
- RNDModule: Core RND implementation (target + predictor networks).
- SemanticRND: Integration of a frozen encoder with the RND module to compute
  intrinsic rewards for (problem, answer) text pairs.

The implementations are intentionally simple and self-contained so they can be
used as building blocks in RL/GRPO training loops.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Iterator
from encoder import Encoder


class RunningMeanStd(nn.Module):
    """
    Running mean and variance tracker using Welford's algorithm.

    Keeps running statistics (mean, variance, count) for streaming data. This
    is used to normalize intrinsic rewards produced by the RND module.

    Args:
        shape: Expected shape of incoming observations (excluding batch dim).
        eps: Small epsilon to ensure numerical stability when computing std.
    """

    def __init__(self, shape: Tuple[int, ...] = (), eps: float = 1e-8):
        super().__init__()
        self.shape = shape
        self.eps = float(eps)

        # Use float32 for efficiency on GPU
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(1e-8, dtype=torch.float32))

    def update(self, x: torch.Tensor):
        """
        Update running statistics with a new batch of observations.

        Uses Welford's online algorithm for numerically stable computation
        of mean and variance from streaming data.

        Args:
            x: Input tensor of shape [batch_size, *shape] where shape matches
               the shape specified during initialization.
        """
        if x.numel() == 0:
            return

        # Ensure consistent dtype and device
        x = x.to(self.mean.device).to(self.mean.dtype)

        # Validate input shape
        if x.dim() == 1 and self.shape == ():
            # Scalar case: shape [B] is acceptable
            pass
        else:
            # Vector case: validate dimensions match
            assert x.shape[1:] == self.shape, f"Input shape {x.shape[1:]} != expected {self.shape}"

        # Compute batch statistics
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = float(x.size(0))

        # Welford's algorithm for combining statistics
        count = self.count
        tot_count = count + batch_count

        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / tot_count)

        # Combine variances using Welford's method
        m_a = self.var * count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + (delta * delta) * (count * batch_count / tot_count)
        new_var = m_2 / tot_count

        # Clamp variance for numerical stability
        new_var = torch.clamp(new_var, min=self.eps)

        # Update buffers in-place
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.fill_(tot_count)

    def normalize(self, x: torch.Tensor, clip_val: float = 5.0) -> torch.Tensor:
        """
        Normalize input using running statistics and clip to prevent outliers.

        Args:
            x: Input tensor to normalize.
            clip_val: Maximum absolute value after normalization (default: 5.0).

        Returns:
            Normalized and clipped tensor with same shape as input.
        """
        x = x.to(self.mean.device).to(self.mean.dtype)
        running_std = torch.sqrt(self.var + self.eps)
        normalized = (x - self.mean) / running_std
        return torch.clamp(normalized, -clip_val, clip_val).to(torch.float32)

    @property
    def std(self):
        """Compute standard deviation from variance."""
        return torch.sqrt(self.var + self.eps)


class RNDModule(nn.Module):
    """
    Random Network Distillation (RND) module for computing intrinsic rewards.

    This implementation follows the RND approach where a randomly initialized target
    network provides a learning signal for a predictor network. The prediction error
    serves as an intrinsic reward signal, encouraging exploration of novel states.

    Key Features:
        - No observation normalization (operates directly on embeddings)
        - Internal reward normalization using running statistics
        - Supports arbitrary batch shapes (e.g., [B, G, E] for GRPO)
        - Exposes predictor parameters for external optimization

    Architecture:
        - Target Network: Frozen MLP that transforms embeddings to feature space
        - Predictor Network: Trainable MLP that learns to match target outputs
        - Reward Normalizer: Running statistics for normalizing intrinsic rewards

    Reference:
        Burda et al. "Exploration by Random Network Distillation" (2018)
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int = 512, 
                 hidden_dim: int = 512,
                 device: str = 'cpu'):
        """
        Initializes the RND module.
        
        Args:
            input_dim (int): Dimension of the input embedding (e.g., 384).
            output_dim (int): Projection size for the RND networks.
            hidden_dim (int): Hidden layer size for the MLPs.
            device (str): The device to run the module on ('cpu' or 'cuda').
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        
        # Constraint 2: Internal reward normalizer
        self.reward_normalizer = RunningMeanStd(shape=()) 
        
        self.target_network = self._build_network(input_dim, output_dim, hidden_dim).to(device)
        self.predictor_network = self._build_network(input_dim, output_dim, hidden_dim).to(device)
        
        # Freeze the Target network (it is never trained)
        for param in self.target_network.parameters():
            param.requires_grad = False

    def _build_network(self, input_dim, output_dim, hidden_dim) -> nn.Sequential:
        """Helper to create the MLP architecture."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def predictor_parameters(self) -> Iterator[nn.Parameter]:
        """
        Returns an iterator over the trainable parameters (predictor network only).
        Use this to initialize your external optimizer.
        """
        return self.predictor_network.parameters()

    def forward(self, embeddings_batch: torch.Tensor, 
                update_stats: bool = True,
                train_encoder: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single forward pass that computes both normalized rewards and predictor loss efficiently.
        
        The key efficiency insight:
        - Target network output is computed once and detached (no gradients needed)
        - Predictor network output is used for both:
          1. Raw rewards (detached for reward computation)
          2. Loss computation (with gradients flowing back to predictor)
        - Embeddings are detached by default to prevent unnecessary backprop through encoder
        
        Args:
            embeddings_batch (torch.Tensor): A batch of embeddings [..., D].
            update_stats (bool): Whether to update the reward normalizer statistics.
                                 Set to False during evaluation/rollout if you don't
                                 want to update running stats.
            train_encoder (bool): Whether to allow gradients to flow back through the embeddings
                                  to train the encoder. Default is False (encoder frozen/separate).
                                  Set to True for end-to-end training of encoder with RND loss.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - normalized_rewards: Normalized intrinsic rewards with original batch shape
                - predictor_loss: Scalar MSE loss for training the predictor
        """
        embeddings_batch = embeddings_batch.to(self.device)
        
        # Detach embeddings by default to prevent unnecessary backprop through encoder
        # Only keep gradients if explicitly training the encoder end-to-end
        if not train_encoder:
            embeddings_batch = embeddings_batch.detach()
        
        original_shape = embeddings_batch.shape
        
        # 1. Flatten to [N, E]
        flat_embeddings = embeddings_batch.view(-1, self.input_dim)
        
        # 2. Forward pass through both networks (single pass each)
        # Target network: frozen, no gradients needed
        target_output = self.target_network(flat_embeddings).detach()
        
        # Predictor network: trainable, keep gradients
        predictor_output = self.predictor_network(flat_embeddings)
        
        # 3. Compute raw rewards (per-sample MSE between target and predictor)
        # Detach predictor output for reward computation to avoid gradients
        raw_rewards = (target_output - predictor_output.detach()).pow(2).mean(dim=1)  # [N]
        
        # 4. Update reward normalizer statistics if requested
        if update_stats:
            self.reward_normalizer.update(raw_rewards)
        
        # 5. Normalize rewards
        normalized_rewards = self.reward_normalizer.normalize(raw_rewards)
        
        # 6. Reshape rewards back to original batch shape (e.g., [B, G])
        output_shape = original_shape[:-1]
        normalized_rewards = normalized_rewards.view(output_shape)
        
        # 7. Compute predictor loss (scalar MSE)
        # Gradients flow from this loss back to predictor_output and predictor network
        predictor_loss = (target_output - predictor_output).pow(2).mean()
        
        return normalized_rewards, predictor_loss


class SemanticRND(nn.Module):
    """
    Integrated Semantic Novelty module combining Encoder + RND.
    
    This module takes (problem, answer) pairs, encodes them using a transformer-based
    sentence encoder, and computes intrinsic rewards using Random Network Distillation.
    
    Input Structure:
        - List[List[Tuple[str, str]]]: Outer list = batches, inner list = groups (GRPO),
          each tuple = (problem, answer) pair
    
    Output:
        - normalized_rewards: Tensor matching the [batch, group] structure
        - predictor_loss: Scalar MSE loss for training the RND predictor network
    """
    
    def __init__(
        self,
        encoder_model_name: str = "Alibaba-NLP/gte-modernbert-base",
        rnd_output_dim: int = 512,
        rnd_hidden_dim: int = 512,
        device: str = None,
        max_length: int = 8192,
        concat_problem_answer: bool = True,
        separator: str = " [SEP] "
    ):
        """
        Initialize the Semantic RND module.
        
        Args:
            encoder_model_name: Hugging Face model identifier for the sentence encoder
            rnd_output_dim: Output dimension of RND networks (projection size)
            rnd_hidden_dim: Hidden layer size for RND MLPs
            device: Device to run on ('cpu' or 'cuda'). If None, auto-detects.
            max_length: Maximum token length for encoder
            concat_problem_answer: If True, concatenate problem and answer before encoding.
                                   If False, encode them separately and concatenate embeddings.
            separator: Separator to use when concatenating problem and answer texts
        """
        super().__init__()
        
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        self.concat_problem_answer = concat_problem_answer
        self.separator = separator
        
        # 1. Initialize the encoder
        print(f"[SemanticRND] Initializing encoder: {encoder_model_name}")
        self.encoder = Encoder(
            model_name=encoder_model_name,
            device=self.device,
            max_length=max_length
        )
        
        # Freeze the encoder - we don't train it with RND
        for param in self.encoder.model.parameters():
            param.requires_grad = False
        self.encoder.model.eval()  # Set to eval mode permanently
        print(f"[SemanticRND] Encoder frozen (requires_grad=False)")
        
        # Get embedding dimension from encoder
        self.embedding_dim = self.encoder.embedding_dim
        
        # If encoding problem and answer separately, embedding dim doubles
        if not concat_problem_answer:
            rnd_input_dim = self.embedding_dim * 2
            print(f"[SemanticRND] Encoding problem and answer separately (input_dim={rnd_input_dim})")
        else:
            rnd_input_dim = self.embedding_dim
            print(f"[SemanticRND] Encoding concatenated problem+answer (input_dim={rnd_input_dim})")
        
        # 2. Initialize the RND module
        print(f"[SemanticRND] Initializing RND module")
        self.rnd = RNDModule(
            input_dim=rnd_input_dim,
            output_dim=rnd_output_dim,
            hidden_dim=rnd_hidden_dim,
            device=self.device
        )
        
        print(f"[SemanticRND] Initialization complete!")
    
    def predictor_parameters(self) -> Iterator[nn.Parameter]:
        """
        Returns an iterator over the RND predictor parameters for optimization.
        Note: The encoder is frozen and not trainable.
        """
        return self.rnd.predictor_parameters()
    
    def encode(
        self,
        problem_answer_pairs: List[List[Tuple[str, str]]]
    ) -> Tuple[torch.Tensor, List[int], List[int]]:
        """
        Encode all (problem, answer) pairs into embeddings.
        
        Args:
            problem_answer_pairs: List[List[Tuple[str, str]]]
                Outer list: batches (B)
                Inner list: groups within each batch (G_i, can vary)
                Tuple: (problem, answer) pair
        
        Returns:
            Tuple containing:
                - embeddings: Tensor of shape [total_samples, embedding_dim]
                - batch_sizes: List of group counts per batch
                - total_samples: Total number of (problem, answer) pairs
        """
        # Flatten the nested structure and track sizes
        all_texts = []
        batch_sizes = []  # Number of groups in each batch
        
        for batch in problem_answer_pairs:
            batch_sizes.append(len(batch))
            for problem, answer in batch:
                if self.concat_problem_answer:
                    # Concatenate problem and answer with separator
                    text = problem + self.separator + answer
                    all_texts.append(text)
                else:
                    # Keep them separate for now (will encode separately)
                    all_texts.append((problem, answer))
        
        total_samples = sum(batch_sizes)
        
        # Encode all texts with no_grad since encoder is frozen
        with torch.no_grad():
            if self.concat_problem_answer:
                # Single batch encoding of all concatenated texts
                embeddings = self.encoder.encode(
                    all_texts,
                    pooling="auto",
                    normalize=True,
                    return_numpy=False
                )  # Shape: [total_samples, embedding_dim]
            else:
                # Encode problems and answers separately, then concatenate
                problems = [pair[0] for pair in all_texts]
                answers = [pair[1] for pair in all_texts]
                
                problem_embeds = self.encoder.encode(
                    problems,
                    pooling="auto",
                    normalize=True,
                    return_numpy=False
                )  # [total_samples, embedding_dim]
                
                answer_embeds = self.encoder.encode(
                    answers,
                    pooling="auto",
                    normalize=True,
                    return_numpy=False
                )  # [total_samples, embedding_dim]
                
                # Concatenate along feature dimension
                embeddings = torch.cat([problem_embeds, answer_embeds], dim=1)  # [total_samples, 2*embedding_dim]
        
        return embeddings, batch_sizes, total_samples
    
    def forward(
        self,
        problem_answer_pairs: List[List[Tuple[str, str]]],
        update_stats: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        End-to-end forward pass: encode texts and compute RND rewards + loss.
        
        Args:
            problem_answer_pairs: List[List[Tuple[str, str]]]
                Nested structure: batches -> groups -> (problem, answer) pairs
            update_stats: Whether to update RND reward normalizer statistics
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - normalized_rewards: Tensor of shape matching input structure
                  If input is [B][G_i] structure, output will be flattened [sum(G_i)]
                  Or can be reshaped based on batch_sizes
                - predictor_loss: Scalar MSE loss for RND predictor training
        """
        # 1. Encode all texts into embeddings (encoder is frozen, wrapped in no_grad)
        embeddings, batch_sizes, total_samples = self.encode(problem_answer_pairs)
        # embeddings shape: [total_samples, embedding_dim or 2*embedding_dim]
        
        # 2. Pass through RND module (train_encoder=False since encoder is frozen)
        normalized_rewards, predictor_loss = self.rnd(
            embeddings,
            update_stats=update_stats,
            train_encoder=False
        )
        # normalized_rewards shape: [total_samples]
        
        # 3. Reshape rewards to match batch structure [B, G_i]
        # For simplicity, we can return as list of tensors per batch
        rewards_per_batch = []
        start_idx = 0
        for batch_size in batch_sizes:
            end_idx = start_idx + batch_size
            batch_rewards = normalized_rewards[start_idx:end_idx]
            rewards_per_batch.append(batch_rewards)
            start_idx = end_idx
        
        # Stack into a single tensor if all batches have same size
        # Otherwise, return as list
        if len(set(batch_sizes)) == 1:
            # All batches have same number of groups - can stack
            num_batches = len(batch_sizes)
            groups_per_batch = batch_sizes[0]
            normalized_rewards = normalized_rewards.view(num_batches, groups_per_batch)
        # else: keep as 1D tensor [total_samples]
        
        return normalized_rewards, predictor_loss
    
    def get_rewards_and_loss(
        self,
        problem_answer_pairs: List[List[Tuple[str, str]]],
        update_stats: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """
        Convenience method for GRPO integration that returns loss as a float.
        
        Args:
            problem_answer_pairs: List[List[Tuple[str, str]]]
            update_stats: Whether to update reward normalizer stats
        
        Returns:
            Tuple[torch.Tensor, float]:
                - normalized_rewards: Reward tensor
                - predictor_loss: Loss value as Python float
        """
        # Encoder is already frozen and wrapped in no_grad in encode()
        # Just ensure predictor loss computation doesn't accumulate gradients
        with torch.no_grad():
            rewards, loss = self.forward(
                problem_answer_pairs,
                update_stats=update_stats
            )
        
        return rewards, loss.item()


if __name__ == "__main__":
    """
    Simple example demonstrating how to use SemanticRND for computing
    intrinsic rewards based on semantic novelty.
    """
    
    # 1. Initialize the module
    print("\n1. Initializing SemanticRND...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    
    semantic_rnd = SemanticRND(
        encoder_model_name="Alibaba-NLP/gte-modernbert-base",
        rnd_output_dim=256,
        rnd_hidden_dim=256,
        device=device,
        max_length=512,
        concat_problem_answer=False,
        separator=" [SEP] "
    )
    
    # 2. Prepare sample data (GRPO-style: batches of groups of problem-answer pairs)
    print("\n2. Preparing sample problem-answer pairs...")
    problem_answer_pairs = [
        [  # Batch 1: Math problems
            ("What is 15 + 27?", "42"),
            ("Calculate 8 × 9", "72"),
            ("What is the square root of 144?", "12"),
        ],
        [  # Batch 2: Science questions
            ("What is the chemical formula for water?", "H2O"),
            ("What gas do plants absorb?", "Carbon dioxide (CO2)"),
        ]
    ]
    print(f"   ✓ Created {len(problem_answer_pairs)} batches")
    print(f"   ✓ Batch 1: {len(problem_answer_pairs[0])} pairs")
    print(f"   ✓ Batch 2: {len(problem_answer_pairs[1])} pairs")
    
    # 3. Compute intrinsic rewards and loss
    print("\n3. Computing intrinsic rewards...")
    rewards, loss = semantic_rnd(
        problem_answer_pairs,
        update_stats=False
    )
    
    print(f"   ✓ Rewards shape: {rewards.shape}")
    print(f"   ✓ Predictor loss: {loss.item():.6f}")
    print(f"\n   Normalized intrinsic rewards:")
    print(f"   {rewards}")
    
    # 4. Train the RND predictor (optional)
    print("\n4. Training RND predictor for 5 steps...")
    optimizer = torch.optim.Adam(semantic_rnd.predictor_parameters(), lr=1e-4)
    
    for step in range(5):
        optimizer.zero_grad()
        
        # Generate varied training data
        train_pairs = [
            [
                (f"Question {step}-{i}: What is {i+1} × {i+2}?", 
                 f"Answer: {(i+1)*(i+2)}")
                for i in range(4)
            ]
        ]
        
        _, loss = semantic_rnd(train_pairs, update_stats=True)
        loss.backward()
        optimizer.step()
        
        print(f"   Step {step+1}/5: Loss = {loss.item():.6f}")
    print("Example completed successfully!")
