"""encoder

Utilities for producing semantic text embeddings from transformer
models. The module exposes a small, focused :class:`Encoder` wrapper
around Hugging Face ``transformers`` models and a couple of pooling
helpers that convert per-token hidden states into fixed-size vector
embeddings.

Supported example models:
    - "ibm-granite/granite-embedding-english-r2"
    - "ibm-granite/granite-embedding-small-english-r2"
    - "Alibaba-NLP/gte-modernbert-base"

The implementation is intentionally minimal and synchronous. It keeps
the model in evaluation mode and performs operations on the device
selected at construction time (CPU or CUDA).

Dependencies
----------
transformers, torch, numpy

Example
-------
>>> enc = Encoder("Alibaba-NLP/gte-modernbert-base")
>>> emb = enc.encode("hello world")  # returns a torch.Tensor or numpy array

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
    """CLS-token pooling.

    Return the embedding of the first token in the sequence (commonly
    the [CLS] token for BERT-like models). This is a simple pooling
    strategy that many embedding models expect.

    Args:
        hidden_states: Tensor of shape ``[B, T, D]`` containing the
            last-layer hidden states from the model.
        attention_mask: Optional tensor of shape ``[B, T]`` (ignored
            for CLS pooling) but accepted for a consistent signature.

    Returns:
        Tensor of shape ``[B, D]``.
    """
    return hidden_states[:, 0]


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean (average) pooling with attention-mask support.

    This computes a masked mean across the sequence dimension. When an
    attention mask is provided, only the tokens with a mask value of 1
    are included in the average; otherwise the plain average across
    tokens is returned.

    Args:
        hidden_states: Tensor with shape ``[B, T, D]``.
        attention_mask: Optional tensor with shape ``[B, T]`` where
            ``1`` indicates a valid token and ``0`` a padding token.

    Returns:
        Tensor with shape ``[B, D]`` containing the mean-pooled
        representations.
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
    """Lightweight encoder wrapper for transformer models.

    The :class:`Encoder` class wraps a Hugging Face tokenizer + model and
    exposes a small, convenient :meth:`encode` method which returns an
    L2-normalized embedding by default. The class keeps the model in
    evaluation mode and will attempt to use CUDA automatically when
    available.

    Attributes:
        model_name: The HF model identifier used to load tokenizer and
            model.
        device: Device string used for model execution (``"cuda"`` or
            ``"cpu"``).
        tokenizer: Loaded HF tokenizer instance.
        model: Loaded HF model instance.
        embedding_dim: Expected dimensionality of output embeddings (if
            available from model config).

    Args:
        model_name: Hugging Face model identifier (e.g.,
            ``"ibm-granite/granite-embedding-english-r2"``).
        device: Optional device string. If ``None`` the code will prefer
            CUDA when available.
        max_length: Maximum token length used for tokenization and
            truncation.
        dtype: Torch dtype to move the model to (use ``torch.float16`` for
            CUDA half precision when supported).
    """

    # Models we know to prefer CLS pooling by default
    _DEFAULT_CLS_MODELS = ("ibm-granite/granite-embedding-english-r2",
    "ibm-granite/granite-embedding-small-english-r2",
    "Alibaba-NLP/gte-modernbert-base")

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
        """Preprocess a single input string before tokenization.

        The default implementation performs a simple ``str.strip()`` to
        remove leading/trailing whitespace. Override this method in a
        subclass to implement additional normalization (for example,
        LaTeX cleanup, code formatting, or language-specific
        canonicalization).

        Args:
            text: Raw input text.

        Returns:
            Preprocessed text string.
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
        """Encode input text(s) into fixed-size embeddings.

        The method accepts a single string or a list of strings. Tokenization
        and model-forwarding happens on the configured device. By default
        embeddings are L2-normalized along the feature dimension.

        Args:
            texts: A single string or a list of strings to encode.
            pooling: Pooling strategy; one of ``"auto"``, ``"cls"``,
                ``"mean"`` or a callable with signature ``(hidden_states,
                attention_mask) -> Tensor``.
            normalize: If True, L2-normalize the resulting embeddings.
            return_numpy: If True, return a NumPy ``ndarray``; otherwise
                return a ``torch.Tensor``.

        Returns:
            If ``texts`` is a single string, returns a vector of shape
            ``[D]``. If ``texts`` is a list, returns a tensor/array of
            shape ``[N, D]``.
        """

        # Handle single string case
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Preprocess
        processed_texts = [self._preprocess(t) for t in texts]

        # Tokenize and move tensors to the selected device
        token_batch = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        # Forward pass
        outputs = self.model(**token_batch, return_dict=True)
        last_hidden = outputs.last_hidden_state  # [B, T, D]

        # Resolve and apply pooling
        pooling_fn = self._resolve_pooling(pooling)
        try:
            embeddings = pooling_fn(last_hidden, token_batch.get("attention_mask", None))
        except TypeError:
            # Support older callables that only accept hidden_states
            embeddings = pooling_fn(last_hidden)

        # Normalize if requested
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # If user passed a single string, return a single vector
        if single_input:
            embeddings = embeddings.squeeze(0)

        if return_numpy:
            embeddings = embeddings.detach().cpu().numpy()

        return embeddings

    # ---------------------------
    # Pooling resolution
    # ---------------------------
    def _resolve_pooling(self, pooling) -> Callable:
        """Resolve pooling strategy from a string or return a callable.

        The function accepts either a callable or one of the strings
        ``"auto"``, ``"cls"``, or ``"mean"``. When ``"auto"`` is
        selected, a tiny heuristic based on the model name determines the
        default (some models are known to expect CLS pooling).

        Args:
            pooling: Strategy specifier or callable.

        Returns:
            A callable with signature ``(hidden_states, attention_mask)``.
        """
        if callable(pooling):
            return pooling

        p = pooling.lower() if isinstance(pooling, str) else None
        if p == "auto" or p is None:
            for model in self._DEFAULT_CLS_MODELS:
                if model.lower() == self.model_name.lower():
                    print(f"[Encoder] using CLS pooling (by auto) for model '{self.model_name}'")
                    return lambda hs, am: cls_pool(hs, am)
            # fallback to mean_pool if we have attention mask available
            print(f"[Encoder] using mean pooling (by auto) for model '{self.model_name}'")
            return lambda hs, am: mean_pool(hs, am)
        if p == "cls":
            print(f"[Encoder] using CLS pooling (by cls) for model '{self.model_name}'")
            return lambda hs, am: cls_pool(hs, am)
        if p == "mean":
            print(f"[Encoder] using mean pooling (by mean) for model '{self.model_name}'")
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
