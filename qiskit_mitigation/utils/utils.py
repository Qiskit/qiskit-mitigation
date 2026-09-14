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

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from qiskit import QuantumCircuit
from qiskit.circuit import BoxOp, CircuitInstruction
from samplomatic.quantum_program import QuantumProgramResult
from samplomatic.utils import find_unique_box_instructions, undress_box

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
    from qiskit_mitigation.trex import TREX
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
) -> list[CircuitInstruction]:
    """Return the unique boxed layers found across the given circuits using the given boxing options.

    The returned list contains one instance of each distinct boxed layer (represented as a :class:`~.CircuitInstruction`) appearing in the input circuits.
    Some mitigation methods enforce relevant boxing options. By supllying the wanted mitigation methods, these options are enforced when boxing the circuits to find unique layers.
    Supplying the wanted mitigation methods is recommended to ensure that the unique layers used for learning wiil be equal to those used in execution.
    The returned boxes can be filtered so only boxes containing measurements, only boxes containing solely gates or all types of boxes are returned.

    Args:
        circuits: The circuit to find the unique layers for.
        mitigation_types: List of initiated mitigation classes that the circuits will be mitigated with. Each mitigation method might enforce boxing options that will affect the unique layers.
            If given, the list must be the same length as ``circuits``.
        custom_boxing_options: The custom boxing options that will be used by :meth:`~samplomatic.transpiler.generate_boxing_pass_manager` function to find the unique layers.
            If a single dict is given, the same custom boxing option will be used for all tasks. If ``None`` is given, the default boxing option will be used.
        box_types: Can be either ``"gates"``, ``"measurements"`` or ``"all"``, corresponding to filter boxes with only gate layers, only measurement layers or all layers, respectively.

    Returns:
        Unique boxed layers found across the given circuits using the given boxing options.
    """
    from qiskit_mitigation.mitigation_task import MitigationTask

    if mitigation_types is None:
        mitigation_types = [MitigationTask()] * len(circuits)
    if len(mitigation_types) != len(circuits):
        raise ValueError("Number of mitigation_types does not match number of circuits.")
    if isinstance(custom_boxing_options, list) and len(custom_boxing_options) != len(circuits):
        raise ValueError("Number of custom boxing options does not match number of circuits.")
    boxing_options_list: Sequence[dict | None]
    if custom_boxing_options is None:
        boxing_options_list = [None] * len(circuits)
    elif isinstance(custom_boxing_options, dict):
        boxing_options_list = [custom_boxing_options] * len(circuits)
    else:
        boxing_options_list = custom_boxing_options

    boxed_circuits = (
        task._box_circuit(circuit, boxing_options)
        for task, circuit, boxing_options in zip(
            mitigation_types, circuits, boxing_options_list, strict=True
        )
    )
    instructions = (box for boxed_circuit in boxed_circuits for box in boxed_circuit)
    unique_boxes: list[CircuitInstruction] = find_unique_box_instructions(
        instructions=instructions, normalize_annotations=None, undress_boxes=True
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
