"""MatMul-related modules for otx.v2.adapters.openvino.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch

from .builder import OPS
from .op import Attribute, Operation


@dataclass
class MatMulV0Attribute(Attribute):
    """MatMulV0Attribute class."""

    transpose_a: bool = field(default=False)
    transpose_b: bool = field(default=False)


@OPS.register()
class MatMulV0(Operation[MatMulV0Attribute]):
    """MatMulV0 class."""

    TYPE = "MatMul"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = MatMulV0Attribute
    attrs: MatMulV0Attribute

    def forward(self, input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
        """MatMulV0's forward function."""
        if self.attrs.transpose_a:
            input_a = torch.transpose(input_a, -1, -2)
        if self.attrs.transpose_b:
            input_b = torch.transpose(input_b, -1, -2)
        return torch.matmul(input_a, input_b)


@dataclass
class EinsumV7Attribute(Attribute):
    """EinsumV7Attribute class."""

    equation: str


@OPS.register()
class EinsumV7(Operation[EinsumV7Attribute]):
    """EinsumV7 class."""

    TYPE = "Einsum"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = EinsumV7Attribute
    attrs: EinsumV7Attribute

    def forward(self, *inputs) -> torch.Tensor:
        """EinsumV7's forward function."""
        return torch.einsum(self.attrs.equation, *inputs)