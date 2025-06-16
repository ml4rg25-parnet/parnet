import os
from typing import Literal

import numpy as np
import torch

from .models import RBPNet
from .utils import sequence_to_onehot


def load_parnet_model_for_prediction(
    model_weigth_path: os.PathLike,
    device: torch.device,
    dtype: torch.dtype,
    is_old_model: bool = False,
) -> RBPNet:
    """Load a parnet model for prediction."""
    model = torch.load(
        model_weigth_path,
        map_location=device,
        weights_only=False,
    ).to(dtype)

    # disable dropout for deterministic results
    model.eval()

    # Parnet is still under heavy dev. and some "older" models are lacking
    # components in their architectures. For prediction it is enough to
    # simply set these components to None.
    if is_old_model:
        model.projection = None
        model.head.control_nograd = False


    return model


def print_basic_info_parnet_model(
    model: RBPNet,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Display basic information about the model."""
    # Display number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of trainable parameters: {params:_}")

    # Display available keys in output from output heads.
    tmp_seq = torch.stack([sequence_to_onehot("A").float()]).to(
        device=device, dtype=dtype
    )
    print("Available output data: ", model({"sequence": tmp_seq}).keys())

PARNET_OUTPUT_KEYS = Literal[
    "target",
    "control",
    "total",
    "mix_coeff",
]

def get_predictions(
    sequence: list[str] | str,
    model: RBPNet,
    device: torch.device,
    dtype: torch.dtype,
    is_old_model: bool = False,
    apply_softmax_transformation_to: list[PARNET_OUTPUT_KEYS] | None = None,
) -> dict[str, torch.Tensor]:
    """Predict using a parnet model."""
    if isinstance(sequence, str):
        sequence = [sequence]

    # Convert sequence to one-hot encoding
    oh_seq = torch.stack(
        [sequence_to_onehot(seq).float() for seq in sequence]
    ).to(device=device, dtype=dtype)

    # Make prediction
    predictions = model({"sequence": oh_seq})

    # Apply softmax transformation if specified
    if apply_softmax_transformation_to is not None:
        for k in apply_softmax_transformation_to:
            if k == "mix_coeff":
                dim = 1
            elif k in ["target", "control", "total"]:
                dim = 2
            else:
                raise ValueError(f"Unknown key {k} for softmax transformation.")
            predictions[k] = torch.nn.functional.softmax(predictions[k], dim=dim)

    return predictions
