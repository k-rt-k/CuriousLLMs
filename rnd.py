"""rnd.py

Lightweight utilities for semantic novelty estimation using Random Network
Distillation (RND). This file provides four main components:

- RunningMeanStd: Online running mean/variance tracker for reward normalization.
- RNDModule: Core RND implementation (target + predictor networks).
- RNDBuffer: Circular FIFO buffer for storing embeddings and sampling minibatches.
- SemanticRND: Integration of a frozen encoder with the RND module to compute
  intrinsic rewards for (problem, answer) text pairs.

The implementations are intentionally simple and self-contained so they can be
used as building blocks in RL/GRPO training loops.
"""

import torch
import torch.nn as nn
import pickle
from pathlib import Path
from typing import Tuple, List, Iterator, Optional, Dict
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
        - Optional FIXED embedding normalization (stats never updated after initialization)
        - Internal reward normalization using running statistics
        - Supports arbitrary batch shapes (e.g., [B, G, E] for GRPO)
        - Exposes predictor parameters for external optimization

    Architecture:
        - Embedding Normalizer: Fixed statistics for consistent input normalization (optional)
        - Target Network: Frozen MLP that transforms embeddings to feature space
        - Predictor Network: Trainable MLP that learns to match target outputs
        - Reward Normalizer: Running statistics for normalizing intrinsic rewards

    Important: Embedding normalization uses FIXED statistics to avoid the problem where
    updating normalization stats would make the "same" embedding appear different over time,
    breaking RND's ability to recognize previously seen states.

    Reference:
        Burda et al. "Exploration by Random Network Distillation" (2018)
    """
    
    def __init__(self, 
                 input_dim: int, 
                 target_layers: Tuple[int, ...] = (512, 512, 512),
                 predictor_layers: Tuple[int, ...] = (512, 512, 512),
                 device: str = 'cpu',
                 normalize_embeddings: bool = False):
        """
        Initializes the RND module.
        
        Args:
            input_dim (int): Dimension of the input embedding (e.g., 384).
            target_layers (Tuple[int, ...]): Layer sizes for target network including output dim.
                                             Example: (256, 128, 64) creates a 3-layer network
                                             with hidden layers of 256, 128 and output of 64.
            predictor_layers (Tuple[int, ...]): Layer sizes for predictor network including output dim.
                                                Must have same output dimension as target_layers.
            device (str): The device to run the module on ('cpu' or 'cuda').
            normalize_embeddings (bool): If True, normalize input embeddings using FIXED
                                        statistics (computed once and frozen). This ensures
                                        the "same" embedding always normalizes to the same value.
                                        Default is False (no normalization, safest option).
        
        Raises:
            ValueError: If target and predictor networks have different output dimensions.
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.normalize_embeddings = normalize_embeddings
        
        # Validate that both networks have same output dimension
        if target_layers[-1] != predictor_layers[-1]:
            raise ValueError(
                f"Target and predictor networks must have same output dimension. "
                f"Got target={target_layers[-1]}, predictor={predictor_layers[-1]}"
            )
        
        self.output_dim = target_layers[-1]
        
        # Optional: FIXED embedding normalizer (stats will be frozen after initialization)
        # This is crucial - we NEVER update these stats during training to avoid the problem
        # where updating stats would make the same embedding normalize to different values
        if normalize_embeddings:
            self.embedding_normalizer = RunningMeanStd(shape=(input_dim,)).to(device)
            self.embedding_stats_frozen = False  # Will be set to True after initialization
        else:
            self.embedding_normalizer = None
            self.embedding_stats_frozen = True
        
        # Reward normalizer: Updated during training (this is standard RND)
        self.reward_normalizer = RunningMeanStd(shape=()).to(device)
        
        self.target_network = self._build_network(input_dim, target_layers).to(device)
        self.predictor_network = self._build_network(input_dim, predictor_layers).to(device)
        
        # Freeze the Target network (it is never trained)
        for param in self.target_network.parameters():
            param.requires_grad = False

    def _build_network(self, input_dim: int, layer_dims: Tuple[int, ...]) -> nn.Sequential:
        """
        Helper to create the MLP architecture with custom layer sizes.
        
        Args:
            input_dim (int): Input dimension
            layer_dims (Tuple[int, ...]): Dimensions for each layer including output
                                         Example: (256, 128, 64) creates:
                                         Linear(input_dim, 256) -> ReLU ->
                                         Linear(256, 128) -> ReLU ->
                                         Linear(128, 64)
        
        Returns:
            nn.Sequential: The constructed MLP
        """
        layers = []
        prev_dim = input_dim
        
        for i, dim in enumerate(layer_dims):
            layers.append(nn.Linear(prev_dim, dim))
            # Add ReLU for all layers except the last one
            if i < len(layer_dims) - 1:
                layers.append(nn.ReLU())
            prev_dim = dim
        
        return nn.Sequential(*layers)

    def predictor_parameters(self) -> Iterator[nn.Parameter]:
        """
        Returns an iterator over the trainable parameters (predictor network only).
        Use this to initialize your external optimizer.
        """
        return self.predictor_network.parameters()
    
    def initialize_embedding_normalizer(self, initial_embeddings: torch.Tensor):
        """
        Initialize embedding normalization statistics from a fixed dataset.
        
        CRITICAL: This must be called ONCE with a representative sample of embeddings
        before training begins. After this call, the embedding normalization stats
        are FROZEN and will never be updated again.
        
        This ensures that the "same" embedding always normalizes to the same value,
        which is essential for RND to recognize previously seen states.
        
        Args:
            initial_embeddings: Tensor of shape [N, input_dim] containing a representative
                               sample of embeddings from your dataset. Should be large enough
                               to capture the distribution (e.g., 1000+ samples).
        
        Raises:
            RuntimeError: If embedding normalization is disabled or already frozen.
        
        Example:
            >>> # Collect initial embeddings
            >>> initial_embeds = semantic_rnd.encode(initial_problem_answer_pairs)[0]
            >>> # Freeze normalization stats
            >>> semantic_rnd.rnd.initialize_embedding_normalizer(initial_embeds)
        """
        if not self.normalize_embeddings:
            raise RuntimeError("Cannot initialize embedding normalizer when normalize_embeddings=False")
        
        if self.embedding_stats_frozen:
            return
        
        # Update stats with initial data
        self.embedding_normalizer.update(initial_embeddings)
        
        # FREEZE the stats - they will never be updated again
        self.embedding_stats_frozen = True

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
        - If embedding normalization is enabled, uses FIXED stats (never updated)
        
        Args:
            embeddings_batch (torch.Tensor): A batch of embeddings [..., D].
            update_stats (bool): Whether to update the reward normalizer statistics.
                                 Set to False during evaluation/rollout if you don't
                                 want to update running stats.
                                 NOTE: This does NOT affect embedding normalization (always frozen).
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
        
        # 1.5. OPTIONAL: Normalize embeddings using FIXED statistics
        # This is safe because stats are frozen - same embedding always gives same normalized value
        if self.normalize_embeddings:
            if not self.embedding_stats_frozen:
                raise RuntimeError(
                    "Embedding normalizer not initialized! Call initialize_embedding_normalizer() "
                    "with initial data before training."
                )
            # Use frozen stats to normalize
            flat_embeddings = self.embedding_normalizer.normalize(flat_embeddings)
        
        # 2. Forward pass through both networks (single pass each)
        # Target network: frozen, no gradients needed
        target_output = self.target_network(flat_embeddings).detach()
        
        # Predictor network: trainable, keep gradients
        predictor_output = self.predictor_network(flat_embeddings)
        
        # 3. Compute raw rewards (per-sample MSE between target and predictor)
        # Detach predictor output for reward computation to avoid gradients
        raw_rewards = (target_output - predictor_output.detach()).pow(2).mean(dim=1)  # [N]
        
        # 4. Update reward normalizer statistics if requested
        # (This is separate from embedding normalization and is standard RND behavior)
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

    def train_step(self, embeddings_batch: torch.Tensor) -> torch.Tensor:
        """
        Perform a single training step on the predictor network.
        
        This method is used for buffer-based training where we sample from a replay
        buffer and train the predictor to match the target network outputs.
        
        Args:
            embeddings_batch: Tensor of shape [N, input_dim] - concatenated embeddings
        
        Returns:
            loss: Scalar MSE loss tensor (with gradients for backprop)
        """
        embeddings_batch = embeddings_batch.to(self.device).detach()
        
        # Flatten to [N, E]
        flat_embeddings = embeddings_batch.view(-1, self.input_dim)
        
        # Apply embedding normalization if enabled
        if self.normalize_embeddings:
            if not self.embedding_stats_frozen:
                raise RuntimeError(
                    "Embedding normalizer not initialized! Call initialize_embedding_normalizer() "
                    "with initial data before training."
                )
            flat_embeddings = self.embedding_normalizer.normalize(flat_embeddings)
        
        # Forward through both networks
        target_output = self.target_network(flat_embeddings).detach()
        predictor_output = self.predictor_network(flat_embeddings)
        
        # Compute MSE loss
        loss = (target_output - predictor_output).pow(2).mean()
        
        return loss


class RNDBuffer:
    """
    Circular FIFO buffer for RND training with separate problem/response storage.
    
    This buffer stores problem and response embeddings separately, along with
    correctness labels. When sampling, it reconstructs (problem, response) pairs
    by using the implicit mapping: response[i] belongs to problem[i // group_size].
    
    The buffer automatically handles FIFO eviction when full - oldest batches are
    overwritten by new ones.
    
    Memory Layout:
        - problem_embs: [max_batches * batch_size, embedding_dim]
        - response_embs: [max_batches * batch_size * group_size, embedding_dim]
        - correctness: [max_batches * batch_size * group_size]
    
    Example:
        >>> buffer = RNDBuffer(max_batches=5, batch_size=128, group_size=16, embedding_dim=768)
        >>> buffer.add_batch(problem_embs, response_embs, correctness)  # Add B=128 problems, B*G=2048 responses
        >>> sampled = buffer.sample(2048)  # Sample 2048 concatenated (problem, response) pairs
    """
    
    def __init__(
        self,
        max_batches: int,
        batch_size: int,
        group_size: int,
        embedding_dim: int,
        device: str = 'cuda'
    ):
        """
        Initialize the RND buffer.
        
        Args:
            max_batches: S = buffer_size_multiplier (how many batches to store)
            batch_size: B = number of problems per batch
            group_size: G = number of rollouts per problem
            embedding_dim: D = dimension of each embedding (problem and response separately)
            device: Device to store tensors on
        """
        self.max_batches = max_batches
        self.batch_size = batch_size
        self.group_size = group_size
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Compute sizes
        self.problems_per_batch = batch_size
        self.responses_per_batch = batch_size * group_size
        self.max_problems = max_batches * batch_size
        self.max_responses = max_batches * batch_size * group_size
        
        # Pre-allocate tensors (circular buffer storage)
        self.problem_embs = torch.zeros(self.max_problems, embedding_dim, device=device)
        self.response_embs = torch.zeros(self.max_responses, embedding_dim, device=device)
        self.correctness = torch.zeros(self.max_responses, dtype=torch.bool, device=device)
        
        # Buffer state
        self.num_batches_stored = 0  # Current number of batches in buffer (0 to max_batches)
        self.write_batch_idx = 0     # Next batch slot to write to (circular: 0 to max_batches-1)
        
        print(f"[RNDBuffer] Initialized: max_batches={max_batches}, batch_size={batch_size}, "
              f"group_size={group_size}, embedding_dim={embedding_dim}")
        print(f"[RNDBuffer] Capacity: {self.max_problems} problems, {self.max_responses} responses")
    
    def add_batch(
        self,
        problem_embs: torch.Tensor,
        response_embs: torch.Tensor,
        correctness: torch.Tensor
    ) -> None:
        """
        Add a batch of problems and responses to the buffer.
        
        If the buffer is full, the oldest batch is overwritten (FIFO).
        
        Args:
            problem_embs: Tensor of shape [B, embedding_dim] - one embedding per problem
            response_embs: Tensor of shape [B*G, embedding_dim] - one embedding per response
            correctness: Tensor of shape [B*G] (bool) - True if response is correct
        
        Raises:
            ValueError: If tensor shapes don't match expected dimensions
        """
        B = self.batch_size
        G = self.group_size
        
        # Validate shapes
        if problem_embs.shape != (B, self.embedding_dim):
            raise ValueError(f"Expected problem_embs shape ({B}, {self.embedding_dim}), "
                           f"got {problem_embs.shape}")
        if response_embs.shape != (B * G, self.embedding_dim):
            raise ValueError(f"Expected response_embs shape ({B * G}, {self.embedding_dim}), "
                           f"got {response_embs.shape}")
        if correctness.shape != (B * G,):
            raise ValueError(f"Expected correctness shape ({B * G},), got {correctness.shape}")
        
        # Compute write positions for this batch slot
        prob_start = self.write_batch_idx * B
        resp_start = self.write_batch_idx * B * G
        
        # Write to buffer (overwrites oldest if full)
        self.problem_embs[prob_start:prob_start + B] = problem_embs.to(self.device)
        self.response_embs[resp_start:resp_start + B * G] = response_embs.to(self.device)
        self.correctness[resp_start:resp_start + B * G] = correctness.to(self.device)
        
        # Update circular buffer state
        self.write_batch_idx = (self.write_batch_idx + 1) % self.max_batches
        self.num_batches_stored = min(self.num_batches_stored + 1, self.max_batches)
    
    def sample(self, n: int) -> torch.Tensor:
        """
        Sample n (problem, response) pairs from the buffer.
        
        Samples response indices uniformly, then looks up the corresponding problem
        embedding based on the implicit mapping: response[i] belongs to problem[i // G].
        
        Args:
            n: Number of pairs to sample
        
        Returns:
            concatenated: Tensor of shape [n, 2 * embedding_dim] containing
                         concatenated (problem_emb, response_emb) pairs
        
        Raises:
            RuntimeError: If buffer is empty
        """
        if self.num_batches_stored == 0:
            raise RuntimeError("Cannot sample from empty buffer")
        
        # Compute how many responses are currently in the buffer
        total_responses = self.num_batches_stored * self.responses_per_batch
        
        # Sample response indices uniformly with replacement
        response_indices = torch.randint(0, total_responses, (n,), device=self.device)
        
        # Get response embeddings
        sampled_responses = self.response_embs[response_indices]
        
        # Compute corresponding problem indices
        # Each batch of B*G responses maps to B problems
        # response[i] in batch b belongs to problem (b * B) + (i % (B*G)) // G
        batch_idx = response_indices // self.responses_per_batch
        within_batch_response_idx = response_indices % self.responses_per_batch
        within_batch_problem_idx = within_batch_response_idx // self.group_size
        problem_indices = batch_idx * self.batch_size + within_batch_problem_idx
        
        # Get problem embeddings
        sampled_problems = self.problem_embs[problem_indices]
        
        # Concatenate for RND input: [problem_emb, response_emb]
        concatenated = torch.cat([sampled_problems, sampled_responses], dim=1)
        
        return concatenated
    
    def get_current_size(self) -> Dict[str, int]:
        """Get current buffer statistics."""
        return {
            'num_batches': self.num_batches_stored,
            'num_problems': self.num_batches_stored * self.batch_size,
            'num_responses': self.num_batches_stored * self.responses_per_batch,
            'capacity_batches': self.max_batches,
            'capacity_responses': self.max_responses
        }
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.num_batches_stored == 0
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return self.num_batches_stored >= self.max_batches


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
        target_layers: Tuple[int, ...] = (512, 512, 512),
        predictor_layers: Tuple[int, ...] = (512, 512, 512),
        device: str = None,
        max_length: int = 8192,
        concat_before_emb: bool = False,
        separator: str = " [SEP] ",
        normalize_embeddings: bool = False,
        embedding_load_file: str = None
    ):
        """
        Initialize the Semantic RND module.
        
        Args:
            encoder_model_name: Hugging Face model identifier for the sentence encoder
            target_layers: Layer dimensions for target network (including output).
                          Example: (256, 128, 64) creates a 3-layer network.
            predictor_layers: Layer dimensions for predictor network (including output).
                             Must have same final dimension as target_layers.
            device: Device to run on ('cpu' or 'cuda'). If None, auto-detects.
            max_length: Maximum token length for encoder
            concat_before_emb: MUST be False for buffer-based training.
                                   Problem and answer are encoded separately and concatenated.
            separator: Separator to use when concatenating problem and answer texts (unused when concat=False)
            normalize_embeddings: If True, normalize embeddings with FIXED statistics.
                                 Requires embedding_load_file to be provided.
            embedding_load_file: Path to precomputed embeddings file (.pkl). Required when
                                normalize_embeddings=True. Used to initialize fixed normalization stats.
        
        Raises:
            ValueError: If concat_before_emb=True (not supported for buffer-based training)
            ValueError: If normalize_embeddings=True but embedding_load_file is not provided
        """
        super().__init__()
        
        # For buffer-based training, we MUST encode problem and answer separately
        # so we can store them separately and sample efficiently
        if concat_before_emb:
            raise ValueError(
                "concat_before_emb=True is not supported for buffer-based training. "
                "Set concat_before_emb=False to encode problem and answer separately, "
                "allowing efficient storage and sampling from the RND buffer."
            )
        
        # Validate embedding_load_file requirement
        if normalize_embeddings and embedding_load_file is None:
            raise ValueError(
                "embedding_load_file must be provided when normalize_embeddings=True. "
                "Provide the path to a precomputed embeddings .pkl file."
            )
        
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        self.concat_before_emb = concat_before_emb
        self.separator = separator
        self.normalize_embeddings = normalize_embeddings
        
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
        
        # Get embedding dimension from encoder
        self.embedding_dim = self.encoder.embedding_dim
        
        # If encoding problem and answer separately, embedding dim doubles
        if not concat_before_emb:
            rnd_input_dim = self.embedding_dim * 2
        else:
            rnd_input_dim = self.embedding_dim
        
        # 2. Initialize the RND module
        self.rnd = RNDModule(
            input_dim=rnd_input_dim,
            target_layers=target_layers,
            predictor_layers=predictor_layers,
            device=self.device,
            normalize_embeddings=normalize_embeddings
        )
        
        # Internal optimizer for buffer-based training
        self._optimizer = torch.optim.Adam(self.rnd.predictor_parameters(), lr=1e-3)
        
        # 3. If normalize_embeddings=True, load precomputed embeddings and initialize normalizer
        if normalize_embeddings:
            initial_embeddings = self._load_and_prepare_embeddings_for_normalizer(
                embedding_load_file
            )
            self.initialize_embedding_normalizer(initial_embeddings)
        
        print(f"[SemanticRND] Initialization complete!")
    
    def _load_and_prepare_embeddings_for_normalizer(
        self, 
        filepath: str
    ) -> torch.Tensor:
        """
        Load precomputed embeddings and prepare them for initializing the embedding normalizer.
        
        Args:
            filepath: Path to the precomputed embeddings .pkl file
        
        Returns:
            torch.Tensor: Concatenated embeddings suitable for initializing the normalizer
        """
        # Load embeddings from disk
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")
        
        print(f"[SemanticRND] Loading embeddings from {filepath}...")
        with open(filepath, 'rb') as f:
            embeddings_data = pickle.load(f)
        
        # Extract embeddings
        problem_embs = embeddings_data['problem_embs']
        solution1_embs = embeddings_data['solution1_embs']
        solution2_embs = embeddings_data['solution2_embs']
        solution3_embs = embeddings_data['solution3_embs']
        
        if not self.concat_before_emb:
            # Do not concatenate problem and solution strings but after encoding
            train_sol1_embs = torch.cat([problem_embs, solution1_embs], dim=1)
            train_sol2_embs = torch.cat([problem_embs, solution2_embs], dim=1)
            train_sol3_embs = torch.cat([problem_embs, solution3_embs], dim=1)
        else:
            raise NotImplementedError("concat_before_emb=True is not supported for normalization.")

        # Combine all training embeddings for normalization stats
        all_embeddings = torch.cat([train_sol1_embs, train_sol2_embs, train_sol3_embs], dim=0)

        print(f"[SemanticRND] Loaded {all_embeddings.shape[0]} embeddings for normalizer initialization")
        return all_embeddings
    
    def predictor_parameters(self) -> Iterator[nn.Parameter]:
        """
        Returns an iterator over the RND predictor parameters for optimization.
        Note: The encoder is frozen and not trainable.
        """
        return self.rnd.predictor_parameters()
    
    
    def initialize_embedding_normalizer(
        self,
        embeddings: torch.Tensor
    ):
        """
        Initialize embedding normalization with FIXED statistics from initial data.
        
        This is called ONCE before training begins if normalize_embeddings=True.
        The statistics computed from this data will be frozen and never updated.
        
        Args:
            embeddings: Initial dataset to compute normalization statistics.
                                 Should be large enough to be representative (e.g., 1000+ samples).
        
        """
        if not self.normalize_embeddings:
            print("[SemanticRND] Embedding normalization is disabled; skipping initialization.")
            return
        
        # Initialize RND's embedding normalizer with these embeddings
        self.rnd.initialize_embedding_normalizer(embeddings)
    
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
                if self.concat_before_emb:
                    # Concatenate problem and answer with separator
                    text = problem + self.separator + answer
                    all_texts.append(text)
                else:
                    # Keep them separate for now (will encode separately)
                    all_texts.append((problem, answer))
        
        total_samples = sum(batch_sizes)
        
        # Encode all texts with no_grad since encoder is frozen
        with torch.no_grad():
            if self.concat_before_emb:
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

    # =========================================================================
    # BUFFER-BASED TRAINING METHODS
    # =========================================================================
    
    def encode_batch_separately(
        self,
        problems: List[str],
        responses: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode problems and responses separately for buffer-based training.
        
        This method is used to prepare embeddings for the RND buffer, where
        problem and response embeddings are stored separately.
        
        Args:
            problems: List of B problem strings (one per problem in batch)
            responses: List of B*G response strings (G responses per problem)
        
        Returns:
            problem_embs: Tensor of shape [B, embedding_dim]
            response_embs: Tensor of shape [B*G, embedding_dim]
        """
        with torch.no_grad():
            problem_embs = self.encoder.encode(
                problems,
                pooling="auto",
                normalize=True,
                return_numpy=False
            )  # [B, embedding_dim]
            
            response_embs = self.encoder.encode(
                responses,
                pooling="auto",
                normalize=True,
                return_numpy=False
            )  # [B*G, embedding_dim]
        
        return problem_embs, response_embs
    
    def compute_normalized_rewards_from_embeddings(
        self,
        problem_embs: torch.Tensor,
        response_embs: torch.Tensor,
        group_size: int,
        update_stats: bool = True
    ) -> torch.Tensor:
        """
        Compute RND rewards WITH running mean/std normalization from pre-computed embeddings.
        
        This method applies the standard RND running mean/std normalization. The returned
        rewards have been normalized by the running statistics maintained by the RND module.
        
        Args:
            problem_embs: Tensor of shape [B, embedding_dim] - one per problem
            response_embs: Tensor of shape [B*G, embedding_dim] - G per problem
            group_size: G = number of responses per problem
            update_stats: Whether to update the running mean/std statistics
        
        Returns:
            normalized_rewards: Tensor of shape [B*G] with running mean/std normalized rewards
        """
        B = problem_embs.shape[0]
        G = group_size
        
        # Expand problem embeddings to match responses: each problem is repeated G times
        expanded_problems = problem_embs.unsqueeze(1).expand(-1, G, -1).reshape(B * G, -1)
        
        # Concatenate: [problem_emb, response_emb]
        concatenated = torch.cat([expanded_problems, response_embs], dim=1)  # [B*G, 2*D]
        
        # Use RNDModule's forward method which applies running mean/std normalization
        # This returns (normalized_rewards, predictor_loss) - we only need rewards
        normalized_rewards, _ = self.rnd(
            concatenated,
            update_stats=update_stats,
            train_encoder=False
        )
        
        return normalized_rewards.flatten()  # Ensure [B*G] shape
    
    def train_step_from_buffer(
        self,
        buffer: 'RNDBuffer',
        minibatch_size: int,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Perform a single training step by sampling from the buffer.
        
        Args:
            buffer: RNDBuffer instance to sample from
            minibatch_size: Number of (problem, response) pairs to sample
            optimizer: Optimizer for the RND predictor network
        
        Returns:
            loss_value: The MSE loss value as a float
        """
        # Sample from buffer
        sampled_embeddings = buffer.sample(minibatch_size)  # [N, 2*embedding_dim]
        
        # Compute loss
        loss = self.rnd.train_step(sampled_embeddings)
        
        # Backward and step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def encode_problems_responses_separately(
        self,
        problems: List[str],
        responses: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode problems and responses into separate embeddings.
        
        Alias for encode_batch_separately for API consistency with curiosity_train.py.
        
        Args:
            problems: List of B problem strings (one per problem in batch)
            responses: List of B*G response strings (G responses per problem)
        
        Returns:
            problem_embs: Tensor of shape [B, embedding_dim]
            response_embs: Tensor of shape [B*G, embedding_dim]
        """
        return self.encode_batch_separately(problems, responses)
    
    def compute_rewards_stratified_normalized(
        self,
        problem_embeddings: torch.Tensor,
        response_embeddings: torch.Tensor,
        extrinsic_rewards: torch.Tensor,
        group_size: int,
        curiosity_coef: float, #TODO: different coefficients for correct and incorrect?
        correctness_threshold: float = 0.5,
        penalize_incorrect: bool = True,
        update_rnd_stats: bool = True
    ) -> torch.Tensor:
        """
        Compute stratified-normalized signed curiosity rewards.
        
        Full pipeline for buffer-based training:
        1. Compute RND rewards WITH running mean/std normalization (standard RND behavior)
        2. Determine correctness from extrinsic rewards
        3. Apply stratified min-max normalization within each group's correct/incorrect subsets
        4. Apply signed curiosity coefficient
        
        The two-stage normalization:
        - Stage 1: RND running mean/std normalization (global, accumulated over time)
        - Stage 2: Stratified group-wise min-max normalization (local, per-group, per-correctness)
        
        Args:
            problem_embeddings: Tensor of shape [B, embedding_dim]
            response_embeddings: Tensor of shape [B*G, embedding_dim]
            extrinsic_rewards: Tensor of shape [B*G] - original task rewards
            group_size: G = number of responses per problem
            curiosity_coef: Coefficient for curiosity rewards
            correctness_threshold: Threshold for determining correctness (>= threshold)
            penalize_incorrect: Whether to apply negative reward for incorrect novel responses
            update_rnd_stats: Whether to update RND's running mean/std statistics
        
        Returns:
            signed_rewards: Tensor of shape [B*G] with signed curiosity rewards
        """
        # 1. Compute RND rewards WITH running mean/std normalization
        # This uses the standard RND normalization (running mean/std)
        rnd_normalized_rewards = self.compute_normalized_rewards_from_embeddings(
            problem_embeddings, response_embeddings, group_size,
            update_stats=update_rnd_stats
        )
        
        # 2. Determine correctness
        correctness = extrinsic_rewards >= correctness_threshold
        
        # 3. Stratified min-max normalization within groups
        # This normalizes the already RND-normalized rewards within each group's 
        # correct/incorrect subsets to [0, 1]
        stratified_normalized_rewards = normalize_rewards_stratified_groupwise(
            rnd_normalized_rewards, correctness, group_size
        )
        
        # 4. Apply signed coefficient
        if penalize_incorrect:
            signed_rewards = apply_signed_curiosity_rewards(
                stratified_normalized_rewards, correctness, curiosity_coef
            )
        else:
            # Only reward correct, no penalty for incorrect
            signed_rewards = torch.where(
                correctness,
                curiosity_coef * stratified_normalized_rewards,
                torch.zeros_like(stratified_normalized_rewards)
            )
        
        return signed_rewards
    
    def train_step_from_embeddings(
        self,
        concatenated_embeddings: torch.Tensor
    ) -> float:
        """
        Perform a single training step from pre-concatenated embeddings.
        
        Uses the internal optimizer to update the predictor network.
        
        Args:
            concatenated_embeddings: Tensor of shape [N, 2*embedding_dim] 
                                    containing concatenated (problem, response) embeddings
        
        Returns:
            loss_value: The MSE loss value as a float
        """
        # Zero gradients
        self._optimizer.zero_grad()
        
        # Compute loss
        loss = self.rnd.train_step(concatenated_embeddings)
        
        # Backward and step
        loss.backward()
        self._optimizer.step()
        
        return loss.item()


def normalize_rewards_stratified_groupwise(
    rewards: torch.Tensor,
    correctness: torch.Tensor,
    group_size: int,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Apply stratified min-max normalization within each group.
    
    This is the SECOND stage of normalization, applied AFTER the RND's running
    mean/std normalization. It normalizes rewards within each group's correct
    and incorrect subsets separately to [0, 1].
    
    For each group of G rollouts:
    - Normalize correct rewards among themselves to [0, 1]
    - Normalize incorrect rewards among themselves to [0, 1]
    
    This ensures that novelty is compared within the same correctness stratum,
    preventing incorrect answers from dominating the reward scale.
    
    Args:
        rewards: Tensor of shape [B*G] with RND-normalized intrinsic rewards
                (already normalized by running mean/std)
        correctness: Tensor of shape [B*G] (bool) - True if response is correct (reward >= threshold)
        group_size: G = number of responses per problem
        eps: Small epsilon for numerical stability
    
    Returns:
        stratified_normalized: Tensor of shape [B*G] with values in [0, 1] within each stratum
    """
    device = rewards.device
    B = rewards.shape[0] // group_size
    G = group_size
    
    # Reshape to [B, G] for group-wise operations
    rewards_bg = rewards.view(B, G)
    correct_bg = correctness.view(B, G)
    
    # Output tensor
    normalized_bg = torch.zeros_like(rewards_bg)
    
    for b in range(B):
        group_rewards = rewards_bg[b]  # [G]
        group_correct = correct_bg[b]  # [G] bool
        
        # Normalize correct subset
        correct_mask = group_correct
        num_correct = correct_mask.sum().item()
        if num_correct > 0:
            correct_rewards = group_rewards[correct_mask]
            if num_correct > 1:
                r_min = correct_rewards.min()
                r_max = correct_rewards.max()
                normalized = (correct_rewards - r_min) / (r_max - r_min + eps)
            else:
                # Single element: assign 0.5 (middle of [0, 1])
                normalized = torch.tensor([0.5], device=device) #TODO: maybe do 1?
            normalized_bg[b, correct_mask] = normalized
        
        # Normalize incorrect subset
        incorrect_mask = ~group_correct
        num_incorrect = incorrect_mask.sum().item()
        if num_incorrect > 0:
            incorrect_rewards = group_rewards[incorrect_mask]
            if num_incorrect > 1:
                r_min = incorrect_rewards.min()
                r_max = incorrect_rewards.max()
                normalized = (incorrect_rewards - r_min) / (r_max - r_min + eps)
            else:
                # Single element: assign 0.5
                normalized = torch.tensor([0.5], device=device)
            normalized_bg[b, incorrect_mask] = normalized
    
    return normalized_bg.view(-1)  # Flatten back to [B*G]


def apply_signed_curiosity_rewards(
    normalized_rewards: torch.Tensor,
    correctness: torch.Tensor,
    curiosity_coef: float
) -> torch.Tensor:
    """
    Apply signed curiosity rewards based on correctness.
    
    - Correct answers: receive +curiosity_coef * normalized_reward (reward novelty)
    - Incorrect answers: receive -curiosity_coef * normalized_reward (penalize creative failures)
    
    Args:
        normalized_rewards: Tensor of shape [B*G] with values in [0, 1]
        correctness: Tensor of shape [B*G] (bool) - True if response is correct
        curiosity_coef: Coefficient for curiosity rewards
    
    Returns:
        signed_rewards: Tensor of shape [B*G] with signed curiosity rewards
    """
    signed_rewards = torch.where(
        correctness,
        curiosity_coef * normalized_rewards,    # Positive for correct
        -curiosity_coef * normalized_rewards    # Negative for incorrect
    )
    return signed_rewards


if __name__ == "__main__":
    """
    Simple example demonstrating how to use SemanticRND for computing
    intrinsic rewards based on semantic novelty.
    
    This example shows both modes:
    1. Without embedding normalization (default, safest)
    2. With FIXED embedding normalization (advanced)
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"EXAMPLE: SemanticRND without embedding normalization (default)")
    print(f"{'='*70}")
    
    # 1. Initialize the module (default: no embedding normalization)
    print("\n1. Initializing SemanticRND...")
    print(f"   Using device: {device}")
    
    semantic_rnd = SemanticRND(
        encoder_model_name="Alibaba-NLP/gte-modernbert-base",
        target_layers=(256, 128, 64),  # Target: 3 layers with decreasing size
        predictor_layers=(512, 256, 64),  # Predictor: different architecture, same output
        device=device,
        max_length=512,
        concat_before_emb=False,
        separator=" [SEP] ",
        normalize_embeddings=True,  # Default: no embedding normalization (safest)
        embedding_load_file="RND_experimentation/embeddings/embeddings_gte_12000.pkl"
    )
    
    # 2. Prepare sample data
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
    
    # 3. Compute intrinsic rewards
    print("\n3. Computing intrinsic rewards...")
    rewards, loss = semantic_rnd(problem_answer_pairs, update_stats=False)
    print(f"   ✓ Rewards shape: {rewards.shape}")
    print(f"   ✓ Predictor loss: {loss.item():.6f}")
    
    # 4. Train briefly
    print("\n4. Training RND predictor for 3 steps...")
    optimizer = torch.optim.Adam(semantic_rnd.predictor_parameters(), lr=1e-4)
    
    for step in range(3):
        optimizer.zero_grad()
        train_pairs = [
            [(f"Q{step}-{i}: What is {i+1} × {i+2}?", f"A: {(i+1)*(i+2)}") for i in range(4)]
        ]
        _, loss = semantic_rnd(train_pairs, update_stats=True)
        loss.backward()
        optimizer.step()
        print(f"   Step {step+1}/3: Loss = {loss.item():.6f}")
    
    print("\n" + "="*70)
    print("EXAMPLE: Buffer-based RND training with stratified rewards")
    print("="*70)
    
    # 5. Demonstrate buffer-based training
    print("\n5. Creating RND buffer...")
    
    # Simulate a small batch: B=4 problems, G=3 responses each
    B, G = 4, 3
    embedding_dim = semantic_rnd.embedding_dim
    
    buffer = RNDBuffer(
        max_batches=5,  # S = 5
        batch_size=B,
        group_size=G,
        embedding_dim=embedding_dim,
        device=device
    )
    
    print(f"   Buffer created: {buffer.get_current_size()}")
    
    # 6. Encode a batch and add to buffer
    print("\n6. Encoding batch and adding to buffer...")
    
    problems = ["What is 2+2?", "What is 3×3?", "What is 5-1?", "What is 6÷2?"]
    responses = [
        # Problem 0: 3 responses
        "4", "The answer is 4", "2+2=4",
        # Problem 1: 3 responses  
        "9", "It's nine", "Wrong answer",
        # Problem 2: 3 responses
        "4", "5-1=4", "The answer is 5",
        # Problem 3: 3 responses
        "3", "6/2=3", "Two"
    ]
    # Correctness: first 2 of each group correct, last one incorrect
    correctness = torch.tensor([
        True, True, True,    # Problem 0: all correct
        True, True, False,   # Problem 1: 2 correct, 1 wrong
        True, True, False,   # Problem 2: 2 correct, 1 wrong
        True, True, False    # Problem 3: 2 correct, 1 wrong
    ], device=device)
    
    problem_embs, response_embs = semantic_rnd.encode_batch_separately(problems, responses)
    print(f"   Problem embeddings: {problem_embs.shape}")
    print(f"   Response embeddings: {response_embs.shape}")
    
    # Compute rewards using proper two-stage normalization:
    # Stage 1: RND running mean/std normalization (global)
    # Stage 2: Stratified group-wise min-max normalization (local)
    # Then apply signed coefficient based on correctness
    extrinsic_rewards = correctness.float()  # Simulate extrinsic rewards (1.0 for correct, 0.0 for incorrect)
    signed_rewards = semantic_rnd.compute_rewards_stratified_normalized(
        problem_embeddings=problem_embs,
        response_embeddings=response_embs,
        extrinsic_rewards=extrinsic_rewards,
        group_size=G,
        curiosity_coef=0.1,
        correctness_threshold=0.5,
        penalize_incorrect=True,
        update_rnd_stats=True
    )
    print(f"   Signed rewards: min={signed_rewards.min():.4f}, max={signed_rewards.max():.4f}")
    
    # Add to buffer
    buffer.add_batch(problem_embs, response_embs, correctness)
    print(f"   Buffer after add: {buffer.get_current_size()}")
    
    # 7. Train from buffer
    print("\n7. Training RND from buffer for 3 steps...")
    for step in range(3):
        loss = semantic_rnd.train_step_from_buffer(buffer, minibatch_size=B*G, optimizer=optimizer)
        print(f"   Step {step+1}/3: Loss = {loss:.6f}")
    
    print("\n" + "="*70)
    print("KEY POINTS:")
    print("  • Use concat_before_emb=False for buffer-based training")
    print("  • Store problem and response embeddings separately in RNDBuffer")
    print("  • Compute rewards BEFORE adding to buffer (for accurate novelty)")
    print("  • Use stratified normalization within correct/incorrect groups")
    print("  • Apply +coef for correct answers, -coef for incorrect")
    print("="*70)
    
    print("\nExample completed successfully!")
