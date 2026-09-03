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

# Warning: this module is not documented and it does not have an RST file.
# If we ever publicly expose interfaces users can import from this module,
# we should set up its RST file.

"""Functionality for specifying and transforming quantum noise."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, Pauli, PauliLindbladMap, PauliList
from samplomatic.annotations import InjectionSite, InjectNoise
from samplomatic.utils import get_annotation

# Instructions which do not affect commutation checks
_UNMOVED = frozenset({"barrier", "delay", "id", "x", "y", "z"})

# Rotation axes of common non-Clifford gates
_ROTATION_AXES = {
    "rx": "X",
    "ry": "Y",
    "rz": "Z",
    "p": "Z",
    "t": "Z",
    "tdg": "Z",
    "rxx": "XX",
    "ryy": "YY",
    "rzz": "ZZ",
}


def create_postselected_noise_mask(
    noisy_circuit: QuantumCircuit,
    noise_maps: dict[str, PauliLindbladMap],
    detectors: PauliList | Sequence[Pauli | str],
) -> tuple[dict[str, np.ndarray], float]:
    r"""Identify generators in ``noise_maps`` that anticommute with one or more ``detectors``.

    Each detector, given as a Pauli operator at the end of ``noisy_circuit`` is evolved backward
    through the circuit. At each :class:`~samplomatic.InjectNoise` site, a generator that
    anticommutes with some evolved detector gets scale ``0``; a surviving generator gets ``1``.

    Args:
        noisy_circuit: A circuit of box instructions. Box instructions may contain Clifford
            instructions, and any non-Clifford instructions they contain must commute with
            all ``detectors``. ``InjectNoise`` annotations should be used to denote noisy layers
            in the circuit, and each ``InjectNoise`` reference ID should be a key in ``noise_maps``.
        detectors: Pauli detectors defined at the end of ``noisy_circuit`` on all qubits.
        noise_maps: A mapping from each unique ``InjectNoise`` reference ID to the Pauli-Lindblad
            noise associated with that circuit layer.

    Returns:
        - A ``dict`` mapping each noisy box's ``InjectNoise.modifier_ref`` to a ``0``/``1`` array aligned with
          that box's ``noise_maps`` generators, ready to use as samplomatic ``local_scales``
        - The sampling cost :math:`\gamma^2 = e^{(4 \sum \lambda \cdot \text{mask})}` of
          inverting only the surviving generators with PEC.
    """
    detectors = PauliList(detectors)
    if detectors.num_qubits != (n := noisy_circuit.num_qubits):
        raise ValueError(
            f"detectors act on {detectors.num_qubits} qubits but noisy_circuit has "
            f"{noisy_circuit.num_qubits}."
        )
    # Signs never affect commutation, so track only the symplectic bits, as one [x | z] block.
    bits = np.hstack([detectors.x, detectors.z]).astype(np.float32)
    scales: dict[str, np.ndarray] = {}
    cache: dict = {}
    kept_rates = 0.0
    for instruction in reversed(noisy_circuit.data):
        operation = instruction.operation
        qargs = [noisy_circuit.find_bit(q).index for q in instruction.qubits]
        inject = get_annotation(operation, InjectNoise) if operation.name == "box" else None
        if inject is None or not inject.ref:
            _back(bits, operation, qargs, cache, n)
            continue
        if (ref := inject.ref) not in noise_maps:
            raise ValueError(
                f"noise_maps has no entry for {ref!r}, a layer of noisy_circuit. "
                f"Its keys are {sorted(noise_maps)}."
            )
        # Shade exactly where the channel acts -- before or after the box's gates.
        if inject.site == InjectionSite.AFTER:
            scales[inject.modifier_ref] = mask = _mask(
                noise_maps[ref], sorted(qargs), bits, n, cache
            )
            kept_rates += noise_maps[ref].rates @ mask
        _back(bits, operation, qargs, cache, n)
        if inject.site != InjectionSite.AFTER:
            scales[inject.modifier_ref] = mask = _mask(
                noise_maps[ref], sorted(qargs), bits, n, cache
            )
            kept_rates += noise_maps[ref].rates @ mask
    return scales, float(np.exp(4 * kept_rates))


def _back(bits: np.ndarray, operation, qargs: list[int], cache: dict, n: int) -> None:
    """Conjugate the detectors' symplectic bits in place backward through one instruction."""
    if operation.name in _UNMOVED:
        return
    if operation.name in ("measure", "reset"):
        # Z support passes a Z-basis measurement exactly; X/Y support is destroyed by either
        # instruction. A reset also erases all earlier errors on its qubit, so Z support drops.
        if bits[:, qargs].any():
            raise ValueError(
                f"a detector has X or Y support on qubit(s) {qargs} where the circuit "
                f"applies a {operation.name!r}, so its value there is not deterministic and "
                "post-selection on it is undefined."
            )
        if operation.name == "reset":
            bits[:, [n + q for q in qargs]] = 0
        return
    body = operation.blocks[0] if operation.name == "box" else None
    key = (
        (operation.name, tuple(map(str, operation.params)))
        if body is None
        else tuple(
            (
                s.operation.name,
                tuple(map(str, s.operation.params)),
                tuple(body.find_bit(q).index for q in s.qubits),
            )
            for s in body.data
        )
    )
    if key not in cache:
        try:
            source = operation if body is None else body
            cache[key] = Clifford(source).adjoint().symplectic_matrix.astype(np.float32)
        except QiskitError:
            cache[key] = None
    matrix = cache[key]
    cols = qargs + [n + q for q in qargs]
    if matrix is not None:
        bits[:, cols] = bits[:, cols] @ matrix % 2
    elif body is not None:
        for sub in reversed(body.data):
            _back(
                bits, sub.operation, [qargs[body.find_bit(q).index] for q in sub.qubits], cache, n
            )
    else:
        # Anticommutation with the axis is the symplectic product against its swapped-halves bits.
        axis = _ROTATION_AXES.get(operation.name, "")
        partner = np.array([c != "X" for c in axis] + [c != "Z" for c in axis], dtype=np.float32)
        if not axis or (bits[:, cols] @ partner % 2).any():
            raise ValueError(
                f"cannot back-propagate through non-Clifford {operation.name!r} on qubits "
                f"{qargs}: a detector anticommutes with its rotation axis, so the first-order "
                "Clifford analysis does not apply there."
            )


def _mask(
    rates: PauliLindbladMap, qubits: list[int], bits: np.ndarray, n: int, cache: dict
) -> np.ndarray:
    """0/1 over ``rates``' generators: 0 where some detector anticommutes at this point.

    ``qubits`` maps the map's qubit indices to circuit qubits; a box's Pauli-Lindblad map
    indexes its qubits in outer-circuit order, so pass the box's sorted qubit indices.
    """
    k = len(qubits)
    if (partners := cache.get(id(rates))) is None:
        partners = cache[id(rates)] = np.zeros((rates.num_terms, 2 * k), dtype=np.float32)
        for j, (pauli, indices, _rate) in enumerate(rates.to_sparse_list()):
            for char, q in zip(pauli, indices, strict=True):
                partners[j, q], partners[j, k + q] = char != "X", char != "Z"
    products = partners @ bits[:, qubits + [n + q for q in qubits]].T % 2
    return np.asarray(products.any(axis=1) == 0, dtype=float)
