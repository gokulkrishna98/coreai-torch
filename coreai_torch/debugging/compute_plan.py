# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Compute plan utilities for Core AI models.

This module provides a :class:`ComputePlan` that, for each Core AI operation,
reports the :class:`ComputeDevice` it is scheduled to run on.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TextIO

import coreai._compiler._mlir_libs._coreaiIR._bindings.mlir as _mlir
from coreai._compiler.ir import Operation
from coreai.authoring import AIProgram
from coreai.runtime import AIModel, SpecializationOptions
from typing_extensions import Self

from .annotations import _DETAIL_STYLE, _Annotation, _AnnotationLine
from .debug_info import DebugInfo, DebugInfoRecord, _Delegate, parse_debug_infos
from .source_annotator import (
    ModulePath,
    _annotate_operations,
    _operation_in_module,
)
from .utils import _walk_operations


@dataclass(frozen=True)
class _ComputeDeviceAnnotation:
    """
    :class:`~coreai_torch.debugging.annotations._Annotation` for compute info.

    Renders the operation's compute device(s) on one comment line and each
    delegate validation message on its own indented line below it.
    """

    operation_name: str
    """Name of the annotated Core AI operation."""

    devices: tuple[str, ...]
    """Sorted compute device names the operation runs on."""

    validation_messages: tuple[str, ...]
    """Sorted delegate validation messages, rendered one per line."""

    def lines(self: Self) -> Iterable[_AnnotationLine]:
        """
        Return the device line followed by any validation-message lines.

        Returns:
            The lines, device first, then one per validation message.

        """
        devices = ", ".join(self.devices)
        return (
            _AnnotationLine(f"{self.operation_name}: {devices}"),
            *(
                _AnnotationLine(f"validation: {message}", _DETAIL_STYLE, indent="  ")
                for message in self.validation_messages
            ),
        )


class ComputeDevice(Enum):
    """Compute device an operation is scheduled to run on."""

    CPU = "CPU"
    """CPU compute device."""

    GPU = "GPU"
    """GPU compute device."""

    NEURAL_ENGINE = "NEURAL_ENGINE"
    """Neural Engine compute device."""

    UNKNOWN = "UNKNOWN"
    """Unknown compute device."""

    @classmethod
    def _missing_(cls, value: object) -> "ComputeDevice":
        """Return UNKNOWN for unrecognized device values."""
        return cls.UNKNOWN

    @classmethod
    def from_string(cls, value: str) -> "ComputeDevice":
        """
        Parse a device string into a :class:`ComputeDevice`.

        Performs case-insensitive matching against the enum values.

        Args:
            value: Raw device string from debug info metadata.

        Returns:
            The matching :class:`ComputeDevice`, or ``UNKNOWN`` if unrecognized.

        """
        device_str = value.strip().upper()
        device_str = (
            ComputeDevice.NEURAL_ENGINE.value if device_str == "ANE" else device_str
        )
        return cls(device_str)


def _get_residency(operation: DebugInfo) -> str | None:
    residency = operation.get_metadata("residency")
    if residency is None or residency.value_type != "string":
        return None

    return residency.value


def _get_compute_device(operation: DebugInfo, delegate: _Delegate) -> ComputeDevice:
    if delegate == _Delegate.INTERPRETER or delegate == _Delegate.BNNS:
        return ComputeDevice.CPU
    elif delegate == _Delegate.MPS:
        residency = _get_residency(operation=operation)
        return (
            ComputeDevice.GPU
            if residency is None
            else ComputeDevice.from_string(residency)
        )
    else:
        return ComputeDevice.UNKNOWN


def _get_ane_validation_message(operation: DebugInfo) -> str | None:
    message = operation.get_metadata("ane_validation_message")
    if message is None or message.value_type != "string":
        return None

    return message.value


def _get_validation_message(operation: DebugInfo, delegate: _Delegate) -> str | None:
    if delegate == _Delegate.MPS:
        return _get_ane_validation_message(operation=operation)

    return None


def _get_operation_id(operation: Operation) -> int | None:
    """
    Extract the Core AI operation ID from an operation's location.

    Args:
        operation: Core AI operation whose location carries the operation ID.

    Returns:
        The Core AI operation ID, or ``None`` if no ID is present.

    """
    op_id_obj = _mlir.get_operation_id(operation.location, "coreai")
    if op_id_obj is None:
        return None
    return getattr(op_id_obj, "value", None)


@dataclass
class _ComputeInfo:
    """Internal record tracking compute devices and validation messages for an op."""

    devices: set[ComputeDevice] = field(default_factory=set)
    """Compute devices the operation is scheduled to run on."""

    validation_messages: set[str] = field(default_factory=set)
    """Delegate validation messages (e.g. ANE validation diagnostics)."""


def _build_coreai_id_to_compute_info_map(
    debug_info_records: list[DebugInfoRecord],
) -> dict[int, _ComputeInfo]:
    coreai_id_to_compute_info_map: dict[int, _ComputeInfo] = defaultdict(_ComputeInfo)
    odix_record = next(
        (
            record
            for record in debug_info_records
            if _Delegate.from_string(record.identifier) == _Delegate.INTERPRETER
        ),
        None,
    )
    if odix_record is None:
        raise ValueError(
            "No INTERPRETER (odix) debug info record found; cannot build compute plan."
        )

    for operation in odix_record.operations:
        op_ids = operation.get_op_ids(level="coreai")
        for op_id in op_ids:
            coreai_id_to_compute_info_map[op_id] = _ComputeInfo(
                devices={ComputeDevice.CPU}
            )

    delegate_records = (
        record
        for record in debug_info_records
        if _Delegate.from_string(record.identifier) != _Delegate.INTERPRETER
    )
    # Merge the delegate-specific device (and any validation message) into the
    # operation's existing compute info. A single operation may be scheduled
    # across more than one compute device, so devices and validation messages
    # accumulate across delegate records rather than overwriting one another.
    for record in delegate_records:
        delegate = _Delegate.from_string(value=record.identifier)
        for operation in record.operations:
            op_ids = operation.get_op_ids(level="coreai")
            for op_id in op_ids:
                compute_device = _get_compute_device(
                    operation=operation, delegate=delegate
                )
                validation_message = _get_validation_message(
                    operation=operation, delegate=delegate
                )
                compute_info = coreai_id_to_compute_info_map[op_id]
                compute_info.devices.add(compute_device)
                if validation_message is not None:
                    compute_info.validation_messages.add(validation_message)

    return coreai_id_to_compute_info_map


class ComputePlan:
    """
    Describes the set of :class:`ComputeDevice` each Core AI operation executes on.

    A single operation may be scheduled across more than one compute device, so
    the plan associates each Core AI operation with a set of devices.

    The plan is constructed from a :class:`~coreai.authoring.AIProgram` by
    loading it into the runtime (optionally with explicit
    :class:`~coreai.runtime.SpecializationOptions`) and reading the
    operation-level debug info metadata embedded in the resulting model.
    """

    def __init__(self: Self, debug_info_records: list[DebugInfoRecord]) -> None:
        """
        Initialize the compute plan.

        Args:
            debug_info_records: Parsed operation-level debug info records from a
                deployed Core AI model, used to derive the Core AI operation ID
                to compute device mapping.

        """
        self._coreai_id_to_compute_info_map = _build_coreai_id_to_compute_info_map(
            debug_info_records=debug_info_records
        )

    @classmethod
    async def from_program(
        cls,
        program: AIProgram,
        specialization_options: SpecializationOptions | None = None,
    ) -> "ComputePlan":
        """
        Build a compute plan from an AIProgram.

        Saves the program to a temporary asset, loads it into the Core AI
        runtime (applying the given specialization options), and reads the
        operation-level debug info to determine the compute device for each
        Core AI operation.

        Args:
            program: AIProgram to build the compute plan for.
            specialization_options: Options for configuring model
                specialization (e.g. preferred compute unit). When None, the
                runtime defaults are used.

        Returns:
            A :class:`ComputePlan` mapping Core AI op IDs to compute devices.

        """
        with TemporaryDirectory() as temp_dir_name:
            asset_path = Path(temp_dir_name) / "model.aimodel"
            specialization_options = (
                specialization_options.with_debug(enabled=True)
                if specialization_options is not None
                else None
            )
            asset = program.save_asset(asset_path)
            model = await AIModel.load(asset.path, specialization_options)

            debug_info_records = parse_debug_infos(model._debug_infos)
            return cls(debug_info_records=debug_info_records)

    def _get_devices_for_id(self: Self, operation_id: int) -> set[ComputeDevice]:
        """
        Get the compute devices for a Core AI operation ID.

        Args:
            operation_id: Core AI operation ID.

        Returns:
            The set of :class:`ComputeDevice` the operation runs on, or a set
            containing ``UNKNOWN`` if the operation is not present in the plan.

        """
        compute_info = self._coreai_id_to_compute_info_map.get(operation_id)
        if compute_info is None:
            return {ComputeDevice.UNKNOWN}
        return set(compute_info.devices)

    def get_devices(self: Self, operation: Operation) -> set[ComputeDevice]:
        """
        Get the compute devices for a Core AI operation.

        Args:
            operation: Core AI operation whose location carries the operation ID.

        Returns:
            The set of :class:`ComputeDevice` the operation runs on, or a set
            containing ``UNKNOWN`` if the operation is not present in the plan.

        """
        operation_id = _get_operation_id(operation)
        if operation_id is None:
            return {ComputeDevice.UNKNOWN}
        return self._get_devices_for_id(operation_id=operation_id)

    def get_validation_messages(self: Self, operation: Operation) -> set[str]:
        """
        Get the delegate validation messages for a Core AI operation.

        Args:
            operation: Core AI operation whose location carries the operation ID.

        Returns:
            The set of validation messages associated with the operation, or an
            empty set if the operation is not present in the plan or has none.

        """
        operation_id = _get_operation_id(operation)
        if operation_id is None:
            return set()
        compute_info = self._coreai_id_to_compute_info_map.get(operation_id)
        if compute_info is None:
            return set()
        return set(compute_info.validation_messages)

    def annotate_source(
        self: Self,
        program: AIProgram,
        output: TextIO | None = None,
        *,
        module: ModulePath | None = None,
        annotate_all_files: bool = False,
    ) -> None:
        """
        Annotate the program's source with each operation's compute device(s).

        Walks ``program`` and, for every operation present in this plan, writes
        the resolved :class:`ComputeDevice` set as a colored comment above the
        corresponding source line, followed by one line per delegate validation
        message.

        Args:
            program: AIProgram whose source should be annotated. Should be the
                same program the plan was built from so operation IDs line up.
            output: Text stream to write annotated source to. Defaults to
                ``sys.stdout`` when None.
            module: Optional module instance path (outermost first) to restrict
                annotation to a single module subtree. When None (default), all
                operations are annotated.
            annotate_all_files: When True, annotate every attributed source file
                (ordered by attribution count, descending). When False
                (default), only the single dominant file is annotated.

        """

        def annotate(operation: Operation) -> _Annotation | None:
            operation_id = _get_operation_id(operation)
            if operation_id is None:
                return None
            compute_info = self._coreai_id_to_compute_info_map.get(operation_id)
            if compute_info is None:
                return None

            return _ComputeDeviceAnnotation(
                operation_name=operation.name,
                devices=tuple(
                    sorted(device.value for device in compute_info.devices),
                ),
                validation_messages=tuple(sorted(compute_info.validation_messages)),
            )

        operations = _walk_operations(program)
        if module is not None:
            operations = [op for op in operations if _operation_in_module(op, module)]

        _annotate_operations(
            operations,
            annotate,
            output,
            annotate_all_files=annotate_all_files,
        )
