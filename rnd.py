import torch
import torch.nn as nn
from typing import Tuple, List


class SemanticNoveltyModule(nn.Module):
    """
    A skeleton class for the combined semantic novelty module.
    
    This class will eventually encapsulate:
    1. A sentence encoder.
    2. An RND module.
    3. Reward and observation normalizers.
    """
    
    def __init__(self, embedding_dim: int = 384, device: str = 'cpu', **kwargs):
        """
        Initializes the components.
        
        Args:
            embedding_dim (int): The output dimension of the sentence encoder.
            device (str): 'cpu' or 'cuda'
            **kwargs: Placeholder for other hyperparameters (like lr, hidden_dims).
        """
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        
        # --- Skeletons for internal components ---
        # In the real code, these will be initialized modules.
        # self.encoder = ...
        # self.rnd_module = ...
        # self.reward_normalizer = ...
        
        print(f"Skeleton SemanticNoveltyModule initialized on {self.device} "
              f"with embedding_dim={self.embedding_dim}.")

    def get_reward_and_loss(self, 
                            problem_answer_pairs: List[Tuple[str, str]]
                           ) -> Tuple[torch.Tensor, float]:
        """
        The main public method.
        
        Takes a batch of (problem, answer) pairs and performs all steps.
        
        Args:
            problem_answer_pairs: List[Tuple[str, str]]
            
        Returns:
            Tuple[torch.Tensor, float]:
                - normalized_rewards: A 1D tensor [B] for your GRPO agent.
                - predictor_loss: A scalar float for logging.
        """
        
        batch_size = len(problem_answer_pairs)

        
        # Dummy normalized rewards (one per item in batch)
        normalized_rewards = torch.rand(batch_size).to(self.device)
        
        # Dummy predictor loss (a single float)
        predictor_loss = 0.5 # A simple float value

        print(f"(Skeleton get_reward_and_loss: Processed {batch_size} pairs.)")
        
        return normalized_rewards, predictor_loss