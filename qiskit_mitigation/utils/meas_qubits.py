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

"""Helpers for deriving the qubits to measure and for expanding partial measurement data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp


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


def get_observable_qubit_indices(
    observables: ObservablesArray | Sequence[SparsePauliOp],
) -> list[int]:
    """Return the sorted circuit qubit indices in the support of the observables.

    A qubit is in the support of the observables if at least one term in any observable acts on
    it with a non-identity operator, including both Pauli operators (X, Y, Z) and projectors
    (0, 1, +, -, r, l).

    Args:
        observables: The observables whose support is to be determined.

    Returns:
        Sorted list of the circuit qubit indices in the support of the observables.
    """
    obs_array = ObservablesArray.coerce(observables)
    obs_support: set[int] = set()
    for sparse_obs in obs_array.ravel().sparse_observables_array():
        obs_support.update(int(q) for q in sparse_obs.indices)
    return sorted(obs_support)


def derive_meas_qubits(
    circuit: QuantumCircuit,
    observables: ObservablesArray | Sequence[SparsePauliOp],
    boxing_options: dict | None = None,
) -> list[int]:
    """Derive the qubits that must be measured when trimming the measurements of a circuit.

    The derived set is the union of the qubits with non-trivial instructions on them and the
    qubits in the support of the observables. This is the single place that decides which qubits
    are measured when measurements are trimmed, so that task preparation and the layer-finding
    functions always agree.

    Args:
        circuit: The circuit that is going to be measured.
        observables: The observables whose expectation values are estimated on ``circuit``.
        boxing_options: The boxing options given by the user. Used only to reject
            ``twirling_strategy="all"``, which boxes every qubit and therefore cannot be combined
            with trimmed measurements.

    Returns:
        Sorted list of the qubit indices to measure.

    Raises:
        ValueError: If ``boxing_options["twirling_strategy"]`` is ``"all"``.
        ValueError: If no qubit is active or observed.
    """
    if boxing_options is not None and boxing_options.get("twirling_strategy") == "all":
        raise ValueError(
            'Trimming the measurements cannot be combined with ``twirling_strategy="all"``, '
            "since this strategy boxes all the qubits. Turn off the trimming "
            "(``trim_measurements=False`` or no ``observables`` for the layer-finding functions) "
            "or use another ``twirling_strategy``."
        )
    meas_qubits = sorted(
        set(get_active_qubits(circuit)) | set(get_observable_qubit_indices(observables))
    )
    if not meas_qubits:
        raise ValueError(
            "Cannot trim the measurements: no qubit is active or observed, that is, the circuit "
            "has no instructions and all the observables are identities. Turn off the trimming."
        )
    return meas_qubits


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


def expand_meas_data_and_flips(
    data: np.ndarray,
    meas_flips: np.ndarray | None,
    meas_qubits: Sequence[int] | None,
    circuit_num_qubits: int | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Expand compressed measurement data and its measurement flips to full-circuit width.

    Both arrays are expanded with :func:`expand_meas_data`. If ``meas_qubits`` and
    ``circuit_num_qubits`` are ``None`` the inputs are returned unchanged.

    Args:
        data: The measurement data, with last axis of length ``len(meas_qubits)``.
        meas_flips: The measurement flips of ``data``, or ``None`` if there are none.
        meas_qubits: Sorted list of circuit qubit indices that were measured.
        circuit_num_qubits: Total number of qubits in the original circuit.

    Returns:
        The expanded data and the expanded flips (``None`` if ``meas_flips`` is ``None``).

    Raises:
        ValueError: If only one of ``meas_qubits`` and ``circuit_num_qubits`` is given.
    """
    data = expand_meas_data(data, meas_qubits, circuit_num_qubits)
    if meas_flips is not None:
        meas_flips = expand_meas_data(meas_flips, meas_qubits, circuit_num_qubits)
    return data, meas_flips
