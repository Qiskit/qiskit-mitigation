# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for mitigation methods."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from qiskit import QuantumCircuit
from qiskit.circuit import BoxOp, CircuitInstruction
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp
from samplomatic.quantum_program import QuantumProgramResult
from samplomatic.utils import find_unique_box_instructions, undress_box

from qiskit_mitigation.trex import TREX
from qiskit_mitigation.utils.meas_qubits import derive_meas_qubits

if TYPE_CHECKING:
    from qiskit_mitigation.mitigation_task import MitigationTask


def load_tasks_from_result(
    result: QuantumProgramResult, return_trex: bool = False
) -> list[MitigationTask]:
    """Load ``MitigationTask`` objects from an execution result using the passthrough_data saved in the result.

    Args:
        result: The resul object to load the tasks from.
        return_trex: If True, include TREX calibration element in the returned list of MitigationTask objects.

    Returns:
        List of ``MitigationTask`` objects that created the program of the given result.
    """
    from qiskit_mitigation.mitigation_task import MitigationTask
    from qiskit_mitigation.pec import PEC
    from qiskit_mitigation.zne.gate_folding import GateFolding
    from qiskit_mitigation.zne.pea import PEA

    if not isinstance(passthrough := result.passthrough_data, dict):
        raise ValueError(
            "Wrong type for passthrough data: Expected a 'dict', found "
            f"'{type(result.passthrough_data)}'."
        )

    if (mitigation_passthrough := passthrough.get("qiskit_mitigation", None)) is None:
        raise ValueError("Missing 'qiskit_mitigation' in passthrough data.")
    if not isinstance(mitigation_passthrough, Sequence) or len(mitigation_passthrough) == 0:
        raise ValueError("'qiskit_mitigation' in passthrough data is empty or not a list.")

    # check if there is TREX calibration task in the program
    trex = None
    for task_passthrough in mitigation_passthrough:
        mitigation_type = task_passthrough.get("mitigation", None)
        if mitigation_type == "trex":
            trex = TREX.create_instance_from_passthrough_data(task_passthrough)
    tasks = []
    for task_passthrough in mitigation_passthrough:
        mitigation_type = task_passthrough.get("mitigation", None)
        trex_calibration = task_passthrough.get("trex_calibration", None)
        version = task_passthrough.get("version", None)
        if version is None:
            raise ValueError("missing version in one of the passthrough data items.")
        if trex_calibration and trex is None:
            raise ValueError(
                "The program contains a task that is using TREX mitigation in which the calibration "
                "circuit was part of the same program, but the TREX calibration result is not found in the program results."
            )
        match mitigation_type:
            case None:
                tasks.append(
                    MitigationTask.create_instance_from_passthrough_data(task_passthrough, trex)
                )
            case "pec":
                tasks.append(PEC.create_instance_from_passthrough_data(task_passthrough, trex))
            case "gate_folding":
                tasks.append(
                    GateFolding.create_instance_from_passthrough_data(task_passthrough, trex)
                )
            case "pea":
                tasks.append(PEA.create_instance_from_passthrough_data(task_passthrough, trex))
            case "trex":
                if return_trex:
                    tasks.append(TREX.create_instance_from_passthrough_data(task_passthrough))  # type: ignore[arg-type]
            case _:
                raise ValueError("Unknown mitigation type in one of the passthrough data items.")
    return tasks


def find_combined_unique_layers(
    circuits: list[QuantumCircuit],
    mitigation_types: list[MitigationTask] | None = None,
    custom_boxing_options: dict | list[dict] | None = None,
    box_types: Literal["gates", "measurement", "all"] = "all",
    observables: Sequence[ObservablesArray | Sequence[SparsePauliOp]] | None = None,
    trex: bool | None = None,
) -> list[CircuitInstruction]:
    """Return the unique boxed layers found across the given circuits using the given boxing options.

    The returned list contains one instance of each distinct boxed layer (represented as a :class:`~.CircuitInstruction`) appearing in the input circuits.
    Some mitigation methods enforce relevant boxing options. By supllying the wanted mitigation methods, these options are enforced when boxing the circuits to find unique layers.
    Supplying the wanted mitigation methods is recommended to ensure that the unique layers used for learning wiil be equal to those used in execution.
    The returned boxes can be filtered so only boxes containing measurements, only boxes containing solely gates or all types of boxes are returned.

    .. note::
        Unless ``trex`` is ``True``, the measurement boxes are found without the boxing options
        that ``prepare`` adds when a :class:`~qiskit_mitigation.TREX` is used, so they do not match
        the measurement boxes of a task prepared with TREX. If ``trex`` is ``True``, the
        measurement boxes of the given circuits are replaced by the measurement layer of the TREX
        calibration circuit.

    Args:
        circuits: The circuit to find the unique layers for.
        mitigation_types: List of initiated mitigation classes that the circuits will be mitigated with. Each mitigation method might enforce boxing options that will affect the unique layers.
            If given, the list must be the same length as ``circuits``.
        custom_boxing_options: The custom boxing options that will be used by :meth:`~samplomatic.transpiler.generate_boxing_pass_manager` function to find the unique layers.
            If a single dict is given, the same custom boxing option will be used for all tasks. If ``None`` is given, the default boxing option will be used.
        box_types: Can be either ``"gates"``, ``"measurements"`` or ``"all"``, corresponding to filter boxes with only gate layers, only measurement layers or all layers, respectively.
        observables: If given, one entry (an observables array or a sequence of observables) per
            circuit. The layers are then found for circuits whose measurements are trimmed to the
            active and observed qubits, exactly as when ``prepare`` is called with
            ``trim_measurements=True`` and the same observables. If ``None`` (default), all the
            qubits are measured.

            .. warning::
                Pass ``observables`` if and only if ``trim_measurements=True`` is passed to
                ``prepare`` of the tasks, together with the same observables. The measured qubits
                change the boxing, so layers learned with a different choice are not found by
                ``prepare``.
        trex: Whether to add the boxing options that ``prepare`` enforces when a
            :class:`~qiskit_mitigation.TREX` is used (measurement boxes annotated for all qubits).
            If ``True``, these options are added for all circuits, the measurement boxes of the
            given circuits are filtered out and the measurement layer of the TREX calibration
            circuit, built from all the given circuits, is returned instead (together with the
            gate layers, according to ``box_types``). If ``False``, the options are never
            added. If ``None`` (default), they are added for each circuit if and only if its
            corresponding task in ``mitigation_types`` has a TREX mitigation. A warning is issued
            when ``box_types`` is ``"measurement"`` or ``"all"`` and ``trex`` is not ``True``,
            since the measurement boxes may then not match those of tasks prepared with TREX.

    Returns:
        Unique boxed layers found across the given circuits using the given boxing options.

    Raises:
        ValueError: If ``trex`` is ``True`` and ``custom_boxing_options`` is a list of different
            options, since the single TREX calibration circuit is boxed with a single set of options.
        ValueError: If the length of ``mitigation_types``, ``custom_boxing_options`` or
            ``observables`` does not match the number of circuits.
        ValueError: If ``observables`` is given and, for some circuit, no qubit is active or
            observed.
    """
    from qiskit_mitigation.mitigation_task import MitigationTask

    if box_types in ["measurement", "all"] and not trex:
        warnings.warn(
            "The measurement boxes are found without the boxing options that `prepare` adds when "
            "a TREX is used, so they do not match the measurement boxes of tasks prepared with "
            "TREX. Pass `trex=True` to add these boxing options.",
            UserWarning,
            stacklevel=2,
        )
    if mitigation_types is None:
        mitigation_types = [MitigationTask()] * len(circuits)
    if len(mitigation_types) != len(circuits):
        raise ValueError("Number of mitigation_types does not match number of circuits.")
    if observables is not None and len(observables) != len(circuits):
        raise ValueError("Number of observables does not match number of circuits.")
    if isinstance(custom_boxing_options, list) and len(custom_boxing_options) != len(circuits):
        raise ValueError("Number of custom boxing options does not match number of circuits.")
    boxing_options_list: list[dict | None]
    if custom_boxing_options is None:
        boxing_options_list = [None] * len(circuits)
    elif isinstance(custom_boxing_options, dict):
        boxing_options_list = [custom_boxing_options] * len(circuits)
    else:
        boxing_options_list = list(custom_boxing_options)

    meas_qubits_list: Sequence[list[int] | None]
    if observables is None:
        meas_qubits_list = [None] * len(circuits)
    else:
        meas_qubits_list = [
            derive_meas_qubits(circuit, obs, boxing_options)
            for circuit, obs, boxing_options in zip(
                circuits, observables, boxing_options_list, strict=True
            )
        ]

    for i, task in enumerate(mitigation_types):
        if trex or (trex is None and task.trex is not None):
            boxing_options_list[i] = TREX._edit_boxing_options(boxing_options_list[i])

    meas_qubits_list: Sequence[list[int] | None]
    if observables is None:
        meas_qubits_list = [None] * len(circuits)
    else:
        meas_qubits_list = [
            derive_meas_qubits(circuit, obs, boxing_options)
            for circuit, obs, boxing_options in zip(
                circuits, observables, boxing_options_list, strict=True
            )
        ]

    boxed_circuits = (
        task._box_circuit(circuit, boxing_options, meas_qubits)
        for task, circuit, boxing_options, meas_qubits in zip(
            mitigation_types, circuits, boxing_options_list, meas_qubits_list, strict=True
        )
    )
    instructions = (box for boxed_circuit in boxed_circuits for box in boxed_circuit)
    unique_boxes: list[CircuitInstruction] = find_unique_box_instructions(
        instructions=instructions, normalize_annotations=None, undress_boxes=True
    )
    # add trex circuit
    if trex:
        # The measurement boxes of the given circuits are replaced by the TREX calibration layer.
        unique_boxes = [box for box in unique_boxes if _find_box_type(box) == "gates"]
        # The calibration circuit is a single circuit, so it can only use a single set of options.
        trex_boxing_options: dict | None
        if isinstance(custom_boxing_options, list):
            trex_boxing_options = custom_boxing_options[0]
            if any(options != trex_boxing_options for options in custom_boxing_options):
                raise ValueError(
                    "With trex=True, all the custom boxing options must be equal, since they are "
                    "also used for the single TREX calibration circuit."
                )
        else:
            trex_boxing_options = custom_boxing_options
        cal_qubits = TREX._calibration_qubits(circuits, meas_qubits_list)
        trex_instance = TREX()
        trex_instance._prepare_calibration_circuit(
            max(cal_qubits) + 1, trex_boxing_options, cal_qubits
        )
        unique_boxes += find_unique_box_instructions(
            instructions=trex_instance.boxed_circuit,
            normalize_annotations=None,
            undress_boxes=True,
        )
    box_types_filter = ["gates", "measurement"] if box_types == "all" else (box_types,)
    return [
        unique_box for unique_box in unique_boxes if _find_box_type(unique_box) in box_types_filter
    ]


def _find_box_type(instruction: BoxOp) -> str:
    """Find the type of :class:`~qiskit.circuit.BoxOp` that ``instruction`` contains.

    Args:
        instruction: The instruction to get the type of.

    Returns:
        The box type. Can be one of ``"gates"`` or ``"measurement"``.

    Raises:
        ValueError: If ``instruction`` does not contain a box.
    """
    box = instruction.operation
    if (name := box.name) != "box":
        raise ValueError(f"Expected a 'box' but found '{name}'.")

    undressed_box = undress_box(box)

    if len(undressed_box.body) == 0:
        return "gates"

    contain_measurement = any(op.name.startswith("measure") for op in undressed_box.body)

    if contain_measurement:
        return "measurement"

    return "gates"
