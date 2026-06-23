# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""coreai-torch: Convert PyTorch models to Core AI format."""

# Re-export MetalParameter so users don't need a separate coreai import.
from coreai.authoring import MetalParameter

from .__version__ import __version__
from ._composite_declaration import generate_composite_decl
from ._decomp import get_decomp_table
from ._torch_metal_kernel import TorchMetalKernel
from .converter import TorchConverter
from .externalize import (
    ExternalizeMarkers,
    ExternalizeSpec,
    mark_for_externalization,
)

__all__ = [
    "__version__",
    "ExternalizeMarkers",
    "ExternalizeSpec",
    "MetalParameter",
    "TorchConverter",
    "TorchMetalKernel",
    "get_decomp_table",
    "generate_composite_decl",
    "mark_for_externalization",
]
