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

"""Helpers for the choice and validation of the qubits to measure."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag


def get_active_qubits(circuit: QuantumCircuit) -> list[int]:
    """Return the sorted indices of the qubits that have non-trivial instructions on them.

    A qubit is active if any instruction other than a barrier or a delay acts on it. Final
    measurements are not taken into account, since they are replaced when the circuit is boxed.

    Args:
        circuit: The circuit to find the active qubits of.

    Returns:
        Sorted list of active qubit indices.
    """
    stripped_circuit = circuit.remove_final_measurements(inplace=False)
    idle_wires = set(circuit_to_dag(stripped_circuit).idle_wires(ignore=["barrier", "delay"]))
    return [index for index, qubit in enumerate(stripped_circuit.qubits) if qubit not in idle_wires]


def validate_meas_qubits(
    circuit: QuantumCircuit,
    meas_qubits: Sequence[int],
    boxing_options: dict | None = None,
    active_qubits: Sequence[int] | None = None,
) -> list[int]:
    """Validate that measuring ``meas_qubits`` is possible for ``circuit`` and return them sorted.

    Args:
        circuit: The circuit that is going to be measured.
        meas_qubits: The indices of the qubits to measure.
        boxing_options: The boxing options that are going to be used to box the circuit.
        active_qubits: The result of :func:`get_active_qubits` for ``circuit``, if it was already
            computed. If ``None``, it is computed here.

    Returns:
        The sorted qubit indices, without duplicates.

    Raises:
        ValueError: If ``meas_qubits`` is empty.
        ValueError: If an index is out of range for the circuit.
        ValueError: If a qubit with non-trivial instructions is not measured.
        ValueError: If ``boxing_options["twirling_strategy"]`` is ``"all"`` and not all qubits are measured.
    """
    normalized = sorted({int(qubit) for qubit in meas_qubits})
    if not normalized:
        raise ValueError("``meas_qubits`` must contain at least one qubit.")
    if out_of_range := [q for q in normalized if q < 0 or q >= circuit.num_qubits]:
        raise ValueError(
            f"``meas_qubits`` contains qubits {out_of_range}, which are out of range for a circuit "
            f"with {circuit.num_qubits} qubits."
        )
    if boxing_options is not None and boxing_options.get("twirling_strategy") == "all":
        if len(normalized) != circuit.num_qubits:
            raise ValueError(
                'All qubits must be measured when ``twirling_strategy`` is "all", but ``meas_qubits`` '
                f"has {len(normalized)} of the {circuit.num_qubits} qubits of the circuit."
            )
    elif unmeasured := sorted(
        set(get_active_qubits(circuit) if active_qubits is None else active_qubits)
        - set(normalized)
    ):
        raise ValueError(
            f"Qubits {unmeasured} have instructions on them but are not in ``meas_qubits``. "
            "Every qubit with non-trivial instructions must be measured. "
            "Use ``get_measured_qubits`` to compute a sufficient set of qubits."
        )
    return normalized


def expand_meas_data(
    data: np.ndarray,
    meas_qubits: Sequence[int] | None,
    circuit_num_qubits: int | None,
) -> np.ndarray:
    """Expand compressed measurement data to full-circuit width by padding unmeasured qubits with zeros.

    The last axis of ``data`` contains one column per measured qubit (in the order given by
    ``meas_qubits``). This function returns an array with last-axis length ``circuit_num_qubits``
    where column ``meas_qubits[i]`` holds the original column ``i`` and all other columns are zero.
    If neither ``meas_qubits`` nor ``circuit_num_qubits`` is given, all the qubits were measured and
    the input ``data`` is returned unchanged.

    Args:
        data: Array with last axis of length ``len(meas_qubits)``.
        meas_qubits: Sorted list of circuit qubit indices that were measured.
        circuit_num_qubits: Total number of qubits in the original circuit.

    Returns:
        Array of the same dtype and all leading dimensions, with last axis length
        ``circuit_num_qubits``.

    Raises:
        ValueError: If only one of ``meas_qubits`` and ``circuit_num_qubits`` is given.
    """
    if (meas_qubits is None) != (circuit_num_qubits is None):
        raise ValueError(
            "``meas_qubits`` and ``circuit_num_qubits`` must be provided together, "
            "but only one was given."
        )
    if meas_qubits is None or circuit_num_qubits is None:
        return data
    expanded = np.zeros((*data.shape[:-1], circuit_num_qubits), dtype=data.dtype)
    expanded[..., list(meas_qubits)] = data
    return expanded
