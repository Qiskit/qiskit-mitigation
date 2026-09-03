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

"""Utility functions for selecting efficient bases for measurement of observables."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp

# Mapping for projecting observable terms to Z computational basis
CHAR_TO_Z_CHARS = (
    dict.fromkeys(["Z", "X", "Y"], "Z")
    | dict.fromkeys(["0", "+", "r"], "0")
    | dict.fromkeys(["1", "-", "l"], "1")
    | {"I": "I"}
)

# Lookup table for converting Pauli characters to samplomatic integers
PAULI_TO_INT_LOOKUP_TABLE = {"I": 0, "Z": 1, "X": 2, "Y": 3}
INT_TO_PAULI_LOOKUP = ["I", "Z", "X", "Y"]


def get_measurement_bases(
    observables: SparsePauliOp | list[SparsePauliOp],
) -> tuple[np.typing.NDArray[np.uint8], dict[Pauli, list[SparsePauliOp]]]:
    """Choose bases to sample in order to calculate expectation values for all given observables.

    Here a "basis" refers to measurement of a full-weight or high-weight Pauli, from which multiple qubit-wise commuting Paulis may be estimated.

    The bases are chosen by grouping commuting Paulis across the different observables.

    Args:
        observables: The observables to calculate using the quantum computer.

    Returns:
        * List of Pauli bases to sample encoded in a list of uint8 where 0=I,1=Z,2=X,3=Y.
        * Dict that maps each measured basis to the relevant Paulis and their coefficients for each observable.
          With the measured bases as keys, for each observable there is a SparsePauliOp representing it.
    """
    if isinstance(observables, SparsePauliOp):
        observables = [observables]
    combined_paulis = sum([obs.paulis for obs in observables]).unique()
    pauli_groups = combined_paulis.group_commuting(qubit_wise=True)
    bases = PauliList([_meas_basis_for_pauli_group(group) for group in pauli_groups])

    observables_as_dicts = [dict(obs.label_iter()) for obs in observables]
    reverser: dict[Pauli, list[SparsePauliOp]] = {}
    for basis, group in zip(bases, pauli_groups, strict=True):
        reverser[basis] = [[] for _ in range(len(observables))]
        current_basis_weight = np.complex128(0)
        for i, observable in enumerate(observables_as_dicts):
            coeffs = []
            paulis = []
            for pauli in set(group):
                coeff = observable.get(pauli.to_label(), None)
                if coeff:
                    coeffs.append(coeff)
                    paulis.append(pauli)
                    current_basis_weight += coeff
            reverser[basis][i] = SparsePauliOp(paulis, coeffs) if paulis else None

    bases = _convert_basis_to_uint_representation(bases)

    return bases, reverser


def _meas_basis_for_pauli_group(group: PauliList) -> Pauli:
    """Find the collective measurement basis of a given commutative Pauli group.

    Args:
        group: The Pauli group to find the collective measurement basis to.

    Returns:
        The Pauli basis to measure that represents the given Pauli group.
    """
    sum_z = group.z.sum(axis=0, dtype=bool)
    sum_x = group.x.sum(axis=0, dtype=bool)
    return Pauli((sum_z, sum_x))


def _convert_basis_to_uint_representation(bases: PauliList) -> np.typing.NDArray[np.uint8]:
    """Converts list of Paulis in PauliList format into an array of integers representing those Paulis.

    The representation of the Paulis as integers is:
    I=0, Z=1, X=2, Y=3

    Args:
        bases: The bases in PauliList format to convert.

    Returns:
        The bases represented as an array of integers.
    """
    bases_uint8 = np.array([np.array(_pauli_to_ints(pauli), dtype=np.uint8) for pauli in bases])
    return bases_uint8


def _convert_pauli_basis(basis: str) -> Pauli:
    """Convert computational basis to Pauli measurement basis.

    Converts basis strings like "000", "++0", "rl1" to Pauli operators.
    - 0, 1 → Z
    - +, - → X
    - r, l → Y
    - I → I

    Args:
        basis: Basis string to convert.

    Returns:
        Pauli operator representing the measurement basis.
    """
    basis = (
        basis.replace("0", "Z")
        .replace("1", "Z")
        .replace("+", "X")
        .replace("-", "X")
        .replace("r", "Y")
        .replace("l", "Y")
    )
    return Pauli(basis)


def _pauli_to_ints(pauli: Pauli) -> list[int]:
    """Convert Pauli to list of ints following samplomatic convention.

    I→0, Z→1, X→2, Y→3

    Args:
        pauli: Pauli operator to convert.

    Returns:
        List of integers representing the Pauli. Index ``i`` of the list represents qubit ``i``.
    """
    return [PAULI_TO_INT_LOOKUP_TABLE[p] for p in pauli.to_label()][::-1]


def _ints_to_pauli(basis: list[int]) -> Pauli:
    """Convert list of ints to Pauli following samplomatic convention.

    0→I, 1→Z, 2→X, 3→Y

    Args:
        basis: List of integers representing the Pauli basis. Index ``i`` of the list represents qubit ``i``.

    Returns:
        Pauli operator which the list of integers represents.
    """
    return Pauli("".join([INT_TO_PAULI_LOOKUP[p] for p in basis][::-1]))


def _identify_measure_basis(pauli: Pauli, measure_bases: list[tuple[Pauli, int]]) -> int:
    """Find which measurement basis can measure the given Pauli.

    A basis is compatible if, on every qubit where ``pauli`` is non-identity,
    the basis measures the exact same Pauli axis. Identity positions in
    ``pauli`` may correspond to any axis in the measurement basis.

    Args:
        pauli: Pauli operator to measure.
        measure_bases: List of (measurement_basis, config_idx) tuples.

    Returns:
        The config_idx of the compatible basis.

    Raises:
        ValueError: If no compatible basis found.
    """
    pauli_support = np.logical_or(pauli.z, pauli.x)

    for basis, config_idx in measure_bases:
        # On all non-identity positions of ``pauli``, the measurement basis
        # must match both the z/x symplectic components exactly.
        if np.array_equal(pauli.z[pauli_support], basis.z[pauli_support]) and np.array_equal(
            pauli.x[pauli_support], basis.x[pauli_support]
        ):
            return config_idx

    raise ValueError(f"Cannot compute eval of {pauli} from the given bases elements.")
