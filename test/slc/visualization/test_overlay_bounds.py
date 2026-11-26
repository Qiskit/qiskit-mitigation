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

"""Tests for overlaying bounds on quantum circuits."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import BoxOp
from qiskit_mitigation.slc.visualization.overlay_bounds import (
    _group_qb_pairs_into_layers,
    _sort_support_keys_for_drawing,
    overlay_bounds_onto_circuit,
)
from samplomatic import InjectNoise


def test_overlay_bounds_unboxed_and_empty_boxes():
    """Test overlay_bounds with unboxed gates and empty boxes."""
    circuit = QuantumCircuit(3)
    # unboxed gate
    circuit.h(0)
    # empty box
    circuit.append(
        BoxOp(QuantumCircuit(3), annotations=[InjectNoise(ref="noise_0", modifier_ref="box_0")]),
        [0, 1, 2],
    )
    # non-empty box
    body = QuantumCircuit(3)
    body.cx(0, 1)
    circuit.append(
        BoxOp(body, annotations=[InjectNoise(ref="noise_1", modifier_ref="box_1")]), [0, 1, 2]
    )

    pauli_bounds = {
        "box_0": {},
        "box_1": {(0,): 0.1, (0, 1): 0.2},
    }

    bounds_circuit = overlay_bounds_onto_circuit(pauli_bounds, circuit, include_empty_boxes=True)
    assert "max_bound" in bounds_circuit.metadata


def test_sort_support_keys_for_drawing_heavy_weights():
    """Test _sort_support_keys_for_drawing with heavy weights (>2 qubits)."""
    supports = [(0,), (1, 2), (0, 1, 2)]
    sorted_supports = _sort_support_keys_for_drawing(supports)
    assert (0, 1, 2) in sorted_supports


def test_group_qb_pairs_non_bipartite():
    """Test _group_qb_pairs_into_layers with non-bipartite graph (e.g. triangle)."""
    triangle_edges = [(0, 1), (1, 2), (0, 2)]
    grouped = _group_qb_pairs_into_layers(triangle_edges)
    assert len(grouped) == 3
