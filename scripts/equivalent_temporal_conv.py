"""Equivalent FP32 lowering for the selected models' small temporal Conv2d.

Preserves Conv2d modules, parameters and state_dict keys. For eligible kernels,
cross-correlation is evaluated as a sum of linear projections over shifted time
slices. Other convolutions retain their original implementation. This is an
explicit opt-in execution backend, not a new model or a precision conversion.
"""
import hashlib
from pathlib import Path
import types

import torch
import torch.nn.functional as F


def linear_temporal_forward(conv, value):
    batch, channels, nodes, _ = value.shape
    kernel, dilation, padding = conv.kernel_size[1], conv.dilation[1], conv.padding[1]
    if padding:
        value = F.pad(value, (padding, padding))
    steps = value.shape[-1] - dilation * (kernel - 1)
    sequence = value.permute(0, 2, 3, 1).contiguous()
    result = None
    for index in range(kernel):
        piece = sequence[:, :, index * dilation:index * dilation + steps, :].contiguous()
        projected = F.linear(piece.reshape(-1, channels), conv.weight[:, :, 0, index].contiguous())
        projected = projected.reshape(batch, nodes, steps, conv.out_channels)
        result = projected if result is None else result + projected
    if conv.bias is not None:
        result = result + conv.bias
    return result.permute(0, 3, 1, 2)


def apply_backend(model):
    paths = []
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Conv2d)
                and module.kernel_size[0] == 1 and module.kernel_size[1] <= 3
                and module.stride == (1, 1) and module.dilation[0] == 1
                and module.padding[0] == 0 and module.padding_mode == "zeros"
                and module.groups == 1 and module.in_channels >= 4):
            module.forward = types.MethodType(linear_temporal_forward, module)
            paths.append(name)
    source = Path(__file__).resolve()
    info = {"name": "linear_temporal_conv_fp32", "module_paths": paths,
            "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "precision": "unchanged", "parameter_and_state_dict_layout": "unchanged"}
    model._benchmark_execution_backend = info
    return model
