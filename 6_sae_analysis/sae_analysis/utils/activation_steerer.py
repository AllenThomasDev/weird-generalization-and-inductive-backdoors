"""Minimal activation steering implementation for ablation."""

import torch
import einops
from typing import Union, Iterable, List


class ActivationIntervention:
    def __init__(
        self,
        vector: torch.Tensor,
        coefficient: float,
        layer_indices: Union[int, List[int]],
        intervention_type: str = "ablation",
    ):
        self.vector = vector
        self.coefficient = float(coefficient)
        self.layer_indices = [layer_indices] if isinstance(layer_indices, int) else list(layer_indices)
        self.intervention_type = intervention_type.lower()
        if self.intervention_type not in {"addition", "ablation"}:
            raise ValueError(f"intervention_type must be 'addition' or 'ablation', got '{intervention_type}'")


class ActivationSteerer:
    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "model.layers",
        "model.model.layers",
    )

    def __init__(
        self,
        model: torch.nn.Module,
        interventions: List[ActivationIntervention],
        *,
        positions: str = "all",
    ):
        self.model = model
        self.positions = positions.lower()
        self._handles = []

        if self.positions not in {"all", "last"}:
            raise ValueError("positions must be 'all' or 'last'")

        p = next(self.model.parameters())
        hidden_size = getattr(self.model.config, "hidden_size", None)
        self.interventions_by_layer = {}

        for intervention in interventions:
            vector = torch.as_tensor(intervention.vector, dtype=p.dtype, device=p.device)
            if vector.ndim != 1:
                raise ValueError(f"Vector must be 1-D, got shape {vector.shape}")
            if hidden_size and vector.numel() != hidden_size:
                raise ValueError(f"Vector length {vector.numel()} ≠ model hidden_size {hidden_size}")

            for layer_idx in intervention.layer_indices:
                if layer_idx not in self.interventions_by_layer:
                    self.interventions_by_layer[layer_idx] = []
                self.interventions_by_layer[layer_idx].append((vector, intervention.coefficient, intervention.intervention_type))


    def _locate_layer_list(self):
        """Find the layer list in the model."""
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:
                if hasattr(cur, "__getitem__"):
                    return cur, path
        raise ValueError("Could not find layer list on the model. Add the attribute name to _POSSIBLE_LAYER_ATTRS.")

    def _get_layer_module(self, layer_idx):
        """Get the module for a specific layer index."""
        layer_list, path = self._locate_layer_list()
        if not (-len(layer_list) <= layer_idx < len(layer_list)):
            raise IndexError(f"layer_idx {layer_idx} out of range for {len(layer_list)} layers")
        return layer_list[layer_idx]

    def _create_hook_fn(self, layer_idx):
        """Create a hook function for a specific layer."""
        def hook_fn(module, ins, out):
            return self._apply_layer_interventions(out, layer_idx)
        return hook_fn

    def _apply_layer_interventions(self, activations, layer_idx):
        """Apply only the interventions assigned to this specific layer."""
        if layer_idx not in self.interventions_by_layer:
            return activations

        if torch.is_tensor(activations):
            tensor_out, was_tuple = activations, False
        elif isinstance(activations, (tuple, list)) and torch.is_tensor(activations[0]):
            tensor_out, was_tuple = activations[0], True
        else:
            return activations

        modified_out = tensor_out
        for vector, coeff, intervention_type in self.interventions_by_layer[layer_idx]:
            fn = self._apply_addition if intervention_type == "addition" else self._apply_ablation
            modified_out = fn(modified_out, vector, coeff)

        return (modified_out, *activations[1:]) if was_tuple else modified_out

    def _apply_addition(self, activations, vector, coeff):
        """Apply standard activation addition: x + coeff * vector"""
        steer = coeff * vector
        if self.positions == "all":
            return activations + steer
        else:
            result = activations.clone()
            result[:, -1, :] += steer
            return result

    def _apply_ablation(self, activations, vector, coeff):
        """Apply ablation: project out direction, then add back with coefficient."""
        vector_norm = vector / (vector.norm() + 1e-8)
        
        if self.positions == "all":
            projections = einops.einsum(activations, vector_norm, "b l d, d -> b l")
            return activations - einops.einsum(projections, vector_norm, "b l, d -> b l d") + coeff * vector
        else:
            result = activations.clone()
            last_pos = result[:, -1, :]
            projection = einops.einsum(last_pos, vector_norm, "b d, d -> b")
            result[:, -1, :] = last_pos - einops.einsum(projection, vector_norm, "b, d -> b d") + coeff * vector
            return result

    def __enter__(self):
        """Register hooks on all unique layers."""
        for layer_idx in self.interventions_by_layer.keys():
            layer_module = self._get_layer_module(layer_idx)
            hook_fn = self._create_hook_fn(layer_idx)
            handle = layer_module.register_forward_hook(hook_fn)
            self._handles.append(handle)
        return self

    def __exit__(self, *exc):
        """Remove all hooks."""
        self.remove()

    def remove(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            if handle:
                handle.remove()
        self._handles = []
