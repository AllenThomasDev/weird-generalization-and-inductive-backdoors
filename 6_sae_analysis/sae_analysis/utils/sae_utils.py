"""SAE utilities: SAE loading and projection operations."""

import torch
import einops
from pathlib import Path
import os
from typing import Any
from huggingface_hub import snapshot_download
from dictionary_learning.utils import load_dictionary


def load_sae_from_path(sae_path: str, device: str = "cpu"):
    """Load SAE from either local path or HuggingFace repository."""
    local_path = Path(sae_path)
    if local_path.exists() and local_path.is_dir():
        final_path = str(local_path)
    else:
        parts = sae_path.split('/')
        if len(parts) < 2:
            raise ValueError(f"Invalid path format: {sae_path}")

        repo_id = f"{parts[0]}/{parts[1]}"
        subfolder = "/".join(parts[2:]) if len(parts) > 2 else None

        try:
            local_dir = snapshot_download(
                repo_id=repo_id,
                allow_patterns=f"{subfolder}/*" if subfolder else None,
                local_dir_use_symlinks=False,
                cache_dir=None
            )
            final_path = os.path.join(local_dir, subfolder) if subfolder else local_dir
        except Exception as e:
            raise RuntimeError(f"Failed to download SAE from {sae_path}: {e}")

    try:
        sae, config = load_dictionary(final_path, device=device)
        sae.eval()
        return sae
    except Exception as e:
        raise RuntimeError(f"Failed to load SAE from {final_path}: {e}")


def load_sae_for_layer(sae_repo: str, layer: int, trainer: int = 1, device: str = "cuda"):
    """Load SAE for a specific layer."""
    sae_path = f"{sae_repo}/resid_post_layer_{layer}/trainer_{trainer}"
    return load_sae_from_path(sae_path, device=device)


def compute_feature_cosine_similarities(
    activation_diff: torch.Tensor,
    sae: Any,
    *,
    device: str = "cuda"
) -> torch.Tensor:
    """Compute cosine similarity between activation difference and SAE features."""
    decoder_weight = sae.decoder.weight
    feature_directions = decoder_weight.T if decoder_weight.shape[0] == activation_diff.shape[0] else decoder_weight
    
    diff_normalized = (activation_diff / activation_diff.norm()).to(torch.float32)
    features_normalized = (feature_directions / feature_directions.norm(dim=1, keepdim=True)).to(torch.float32)
    
    return einops.einsum(features_normalized, diff_normalized, "f d, d -> f")
