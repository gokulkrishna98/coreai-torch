# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for debug_infos from CompiledLibrary and AIModel."""

import sys
import tempfile
from pathlib import Path

import pytest
import torch
from coreai.authoring import AIProgram
from coreai.runtime import AIModel

from coreai_torch.converter import TorchConverter, _DebugInfoRecorder
from coreai_torch.debugging.debug_info import DebugInfoRecord, parse_debug_infos

from .test_model import LinearMulAddModel, get_example_inputs


@pytest.fixture
async def simple_coreai_program() -> AIProgram:
    """Create a coreai_program program with debug info from a simple torch model."""
    model = LinearMulAddModel().eval()
    example_inputs = get_example_inputs(LinearMulAddModel)
    exported_program = torch.export.export(model, args=tuple(example_inputs.values()))
    exported_program = exported_program.run_decompositions()

    converter: TorchConverter = TorchConverter()
    converter._debug_info_recorder.config = _DebugInfoRecorder.Config(
        include_stack_trace=True,
        options=_DebugInfoRecorder.Options.DEBUGINFO,
        verify_debuginfo_locations=True,
    )
    converter.add_exported_program(exported_program, entrypoint_name="main")
    coreai_program = converter.to_coreai()
    return coreai_program


def _verify_debug_info_record(record: DebugInfoRecord) -> None:
    """Verify debug info record structure and content."""
    assert len(record.identifier) > 0, (
        "Expected non-empty identifier in debug info record"
    )
    assert len(record.operations) > 0, "Expected operations in debug info"

    # Check first operation
    op = record.operations[0]
    assert op.odix_id >= 0
    # Verify all source locations have expected fields
    for loc in op.source_locations:
        assert isinstance(loc.file_name, str)
        assert isinstance(loc.line, int)
        assert isinstance(loc.column, int)

    # Verify all metadata entries have expected fields
    for metadata in op.metadatas:
        assert isinstance(metadata.key, str)
        assert metadata.value.value_type in [
            "integer",
            "string",
            "array",
            "dictionary",
            "unit",
            "unknown",
        ]


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Requires loading a runtime asset (AIModel.load); only supported on macOS",
)
@pytest.mark.asyncio
async def test_compiled_library_debug_infos(
    simple_coreai_program: AIProgram,
) -> None:
    """Test debug_infos from CompiledLibrary returns valid metadata."""

    with tempfile.TemporaryDirectory() as tmpdir:
        asset_path = Path(tmpdir) / "test_model.aimodel"

        # Create asset from AIProgram and load model from asset
        asset = simple_coreai_program.save_asset(asset_path)
        library = await AIModel.load(asset.path)

        # Get and parse debug_infos
        debug_infos_bytes = library._debug_infos
        assert isinstance(debug_infos_bytes, bytes), "debug_infos should return bytes"

        debug_info_records = parse_debug_infos(debug_infos_bytes)
        assert len(debug_info_records) > 0, "Expected at least one debug info record"

        # Verify structure
        _verify_debug_info_record(debug_info_records[0])


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Requires loading a runtime asset (AIModel.load); only supported on macOS",
)
@pytest.mark.asyncio
async def test_aimodel_debug_infos(
    simple_coreai_program: AIProgram,
) -> None:
    """Test debug_infos from AIModel returns valid metadata."""

    with tempfile.TemporaryDirectory() as tmpdir:
        asset_path = Path(tmpdir) / "test_model.aimodel"

        # Create asset from AIProgram and load model from asset
        asset = simple_coreai_program.save_asset(asset_path)
        ai_model = await AIModel.load(asset.path)

        # Get and parse debug_infos
        debug_infos_bytes = ai_model._debug_infos
        assert isinstance(debug_infos_bytes, bytes), "debug_infos should return bytes"

        debug_info_records = parse_debug_infos(debug_infos_bytes)
        assert len(debug_info_records) > 0, "Expected at least one debug info record"

        # Verify structure
        _verify_debug_info_record(debug_info_records[0])
