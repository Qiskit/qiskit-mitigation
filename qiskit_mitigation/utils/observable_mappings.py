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

"""Utility functions for mapping observables between qubit-indexing definitions."""

from __future__ import annotations

from collections.abc import Sequence

from qiskit.quantum_info import Pauli, SparseObservable, SparsePauliOp


def _permute_observable(
    observable: Pauli | SparsePauliOp | SparseObservable, qubits: Sequence[int]
) -> Pauli | SparsePauliOp | SparseObservable:
    """Permute an observable so that output qubit ``i`` acts as input qubit ``qubits[i]``."""
    if isinstance(observable, Pauli):
        permuted = observable[list(qubits)]
        permuted.phase = observable.phase
        return permuted
    if isinstance(observable, (SparsePauliOp, SparseObservable)):
        new_indices = {old: new for new, old in enumerate(qubits)}
        return type(observable).from_sparse_list(
            [
                (paulis, [new_indices[qubit] for qubit in term_qubits], coeff)
                for paulis, term_qubits, coeff in observable.to_sparse_list()
            ],
            num_qubits=len(qubits),
        )
    raise ValueError(
        f"Observable of type {type(observable)} is not supported, try casting to a "
        "Pauli, SparsePauliOp, or SparseObservable."
    )


def map_observable_isa_to_canonical(
    isa_observable: Pauli | SparsePauliOp | SparseObservable, canonical_qubits: Sequence[int]
) -> Pauli | SparsePauliOp | SparseObservable:
    """Map an observable defined relative to the transpiled circuit to canonical box-order.

    In the transpiled (or ISA) ordering, the qubits are indexed based on the "physical"
    layout of qubits in the device.

    For info on canonical qubit ordering conventions see the `Samplomatic docs <https://qiskit.github.io/samplomatic/guides/samplex_io.html#qubit-ordering-convention>`_).

    Args:
        isa_observable: A `Pauli`, `SparsePauliOp`, or `SparseObservable` object.
        canonical_qubits: A sequence specifying the physical qubit for each canonical qubit.

    Return:
        A mapped operator of the same type as ``isa_observable``
    """
    return _permute_observable(isa_observable, canonical_qubits)


def map_observable_virtual_to_canonical(
    virt_observable: Pauli | SparsePauliOp | SparseObservable,
    layout: Sequence[int],
    canonical_qubits: Sequence[int],
) -> Pauli | SparsePauliOp | SparseObservable:
    """Map an observable with virtual qubit ordering to canonical box-order.

    For info on canonical qubit ordering conventions see the `Samplomatic docs <https://qiskit.github.io/samplomatic/guides/samplex_io.html#qubit-ordering-convention>`_).

    Args:
        virt_observable: A `Pauli`, `SparsePauliOp`, or `SparseObservable` object.
        layout: The list of physical qubits used for the isa circuit.
        canonical_qubits: A sequence specifying the physical qubit for each canonical qubit.

    Return:
        A mapped operator of the same type as ``virt_observable``
    """
    virtual_qubits = {phys: virt for virt, phys in enumerate(layout)}
    return _permute_observable(virt_observable, [virtual_qubits[phys] for phys in canonical_qubits])


def map_observable_isa_to_virtual(
    isa_observable: Pauli | SparsePauliOp | SparseObservable, layout: Sequence[int]
) -> Pauli | SparsePauliOp | SparseObservable:
    """Map an observable defined relative to the transpiled circuit to virtual order.

    In the transpiled (or ISA) ordering, the qubits are indexed based on the "physical"
    layout of qubits in the device.

    Args:
        isa_observable: A `Pauli`, `SparsePauliOp`, or `SparseObservable` object.
        layout: The list of physical qubits used for the isa circuit.

    Return:
        A mapped operator of the same type as ``isa_observable``
    """
    return _permute_observable(isa_observable, layout)
