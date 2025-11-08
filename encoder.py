"""
encoder.py

Contains an encoder that is responsible for text semantic embedding.
Supported Models:
    - "ibm-granite/granite-embedding-english-r2"
    - "ibm-granite/granite-embedding-small-english-r2"
    - "Alibaba-NLP/gte-modernbert-base"

Dependencies:
    pip install transformers torch
"""

from typing import List, Union, Callable, Optional
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ---------------------------
# Helpers: pooling functions
# ---------------------------
def cls_pool(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    CLS pooling: return the first token embedding.
    hidden_states: [B, T, D]
    returns: [B, D]
    """
    return hidden_states[:, 0]


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Attention-mask aware mean pooling.
    hidden_states: [B, T, D]
    attention_mask: [B, T]
    returns: [B, D]
    """
    # mask = attention_mask.unsqueeze(-1).type_as(hidden_states)  # [B, T, 1]
    # summed = (hidden_states * mask).sum(dim=1)
    # counts = mask.sum(dim=1).clamp(min=1e-9)
    # return summed / counts
    return NotImplementedError("Mean Pooling has not been implemented yet.")


# ---------------------------
# Encoder class
# ---------------------------
class Encoder:
    """
    Unified transformer embedding encoder.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier (e.g., "ibm-granite/granite-embedding-english-r2").
    device : str | None
        "cuda" or "cpu". If None, auto-selects CUDA when available.
    max_length : int
        Maximum tokens to send to model (tokenizer truncation length). Default 8192.
    dtype : torch.dtype
        dtype used for model (torch.float32 or torch.float16 for GPU).
    """

    # Models we know to prefer CLS pooling by default
    _DEFAULT_CLS_MODELS = ("granite", "gte", "ibm-granite", "Alibaba-NLP", "gte-")

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_length: int = 8192,
        dtype: torch.dtype = torch.float32,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.dtype = dtype

        # Load tokenizer and model
        print(f"[Encoder] loading model '{model_name}' on device={self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        # Ensure pad token exists (some models don't have one)
        if self.tokenizer.pad_token is None:
            # try to set pad token to eos or add one
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.model.resize_token_embeddings(len(self.tokenizer))

        # Move model to device and dtype
        self.model.to(self.device)
        if self.device.startswith("cuda") and self.dtype == torch.float16:
            # convert to half precision if requested
            self.model.half()
        self.model.eval()

        # dimension
        self.embedding_dim = getattr(self.model.config, "hidden_size", None)
        print(f"[Encoder] loaded (embedding_dim={self.embedding_dim})")

    # ---------------------------
    # Preprocess hook
    # ---------------------------
    def _preprocess(self, text: str) -> str:
        """
        Preprocess input string before tokenization.
        Default: identity (strip whitespace). Extend for LaTeX/code normalization as needed.
        """
        return text.strip()

    # ---------------------------
    # Encode method for both single and batched inputs
    # ---------------------------

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        pooling: Union[str, Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]] = "auto",
        normalize: bool = True,
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, "np.ndarray"]:
        """
        Encode one or more text inputs into embeddings (CUDA by default).

        Parameters
        ----------
        texts : str or List[str]
            Text(s) to encode. Can be a single string or list.
        pooling : 'auto' | 'cls' | 'mean' | callable
            Pooling strategy. 'auto' picks default based on model_name.
        normalize : bool
            Whether to L2-normalize the embeddings.
        return_numpy : bool
            If True, return numpy array(s); otherwise torch.Tensor (default).

        Returns
        -------
        torch.Tensor (or numpy.ndarray)
            [D] if single input, [N, D] if multiple.
        """

        # Handle single string case
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Preprocess
        processed_texts = [self._preprocess(t) for t in texts]

        # Tokenize
        token_batch = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)  # keep everything on same device

        # Forward
        outputs = self.model(**token_batch, return_dict=True)
        last_hidden = outputs.last_hidden_state  # [B, T, D]

        # Pooling
        pooling_fn = self._resolve_pooling(pooling)
        try:
            embeddings = pooling_fn(last_hidden, token_batch.get("attention_mask", None))
        except TypeError:
            embeddings = pooling_fn(last_hidden)

        # Normalize if requested
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Handle single input return
        if single_input:
            embeddings = embeddings.squeeze(0)  # [D]

        if return_numpy:
            embeddings = embeddings.detach().cpu().numpy()

        return embeddings

    # ---------------------------
    # Pooling resolution
    # ---------------------------
    def _resolve_pooling(self, pooling) -> Callable:
        """
        Resolve pooling strategy string or return given callable.
        'auto' chooses default based on model name heuristics.
        """
        if callable(pooling):
            return pooling

        p = pooling.lower() if isinstance(pooling, str) else None
        if p == "auto" or p is None:
            # heuristic: if model name contains known tokens, use CLS pooling
            name_lower = self.model_name.lower()
            for token in self._DEFAULT_CLS_MODELS:
                if token.lower() in name_lower:
                    return lambda hs, am: cls_pool(hs, am)
            # fallback to mean_pool if we have attention mask available
            return lambda hs, am: mean_pool(hs, am)
        if p == "cls":
            return lambda hs, am: cls_pool(hs, am)
        if p == "mean":
            return lambda hs, am: mean_pool(hs, am)
        raise ValueError(f"Unsupported pooling option: {pooling}")


# ---------------------------
# Quick demo when run as script
# ---------------------------
if __name__ == "__main__":
    # Example usage (replace with your preferred model names)
    MODEL = "ibm-granite/granite-embedding-english-r2"  
    MODEL = "Alibaba-NLP/gte-modernbert-base"
    enc = Encoder(MODEL, max_length=8192)

    single = "Explain why x^2 + y^2 = 25 represents a circle."
    emb = enc.encode(single)  # returns torch.Tensor of shape [D]
    print("Single embedding:", emb.shape, emb.dtype)

    # Define three sentences
    sentences = [
        "Find the derivative of x² + 3x + 2.",  # A
        "Compute the slope function for x squared plus three x plus two.",  # B (similar)
        "Explain why the sky appears blue during the day."  # C (different)
    ]

    # Encode all at once
    embs = enc.encode(sentences)  # shape [3, D]

    # Compute cosine similarities
    sim_ab = F.cosine_similarity(embs[0], embs[1], dim=0).item()
    sim_ac = F.cosine_similarity(embs[0], embs[2], dim=0).item()
    sim_bc = F.cosine_similarity(embs[1], embs[2], dim=0).item()

    print(f"Cosine similarity (A, B): {sim_ab:.4f}  <-- semantically similar")
    print(f"Cosine similarity (A, C): {sim_ac:.4f}  <-- unrelated")
    print(f"Cosine similarity (B, C): {sim_bc:.4f}")
