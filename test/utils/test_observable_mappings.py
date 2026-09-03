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

"""Tests for the observable_mappings module."""

import pytest
from qiskit.quantum_info import Pauli, SparseObservable, SparsePauliOp
from qiskit_mitigation.utils.observable_mappings import (
    map_observable_isa_to_canonical,
    map_observable_isa_to_virtual,
    map_observable_virtual_to_canonical,
)

# Each mapper is bound to fixed index arguments; the labels are the expected
# outputs for inputs "XYZ" and "IIZ" respectively.
MAPPERS = [
    pytest.param(
        lambda obs: map_observable_isa_to_canonical(obs, [2, 0, 1]),
        "YZX",
        "IZI",
        id="isa_to_canonical",
    ),
    pytest.param(
        lambda obs: map_observable_isa_to_virtual(obs, [2, 0, 1]),
        "YZX",
        "IZI",
        id="isa_to_virtual",
    ),
    pytest.param(
        lambda obs: map_observable_virtual_to_canonical(obs, [5, 3, 4], [3, 4, 5]),
        "ZXY",
        "ZII",
        id="virtual_to_canonical",
    ),
]


@pytest.mark.parametrize("mapper,xyz,iiz", MAPPERS)
class TestObservableMappings:
    """Tests shared by all three mapping functions."""

    # pylint: disable=unused-argument

    @pytest.mark.parametrize("cls", [Pauli, SparsePauliOp, SparseObservable])
    def test_basic_mapping(self, mapper, xyz, iiz, cls):
        """Each supported observable type is remapped onto the new qubit indices."""
        assert mapper(cls("XYZ")) == cls(xyz)

    def test_multiple_terms_and_coefficients(self, mapper, xyz, iiz):
        """Term order, coefficients, and identities are preserved."""
        obs = SparsePauliOp(["XYZ", "IIZ"], [0.5, 1.5j])
        assert mapper(obs) == SparsePauliOp([xyz, iiz], [0.5, 1.5j])

    def test_invalid_type(self, mapper, xyz, iiz):
        """Unsupported observable types raise a ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            mapper("XYZ")

    def test_pauli_phase_preserved(self, mapper, xyz, iiz):
        """The phase of a Pauli observable survives the mapping."""
        assert mapper(Pauli("-XYZ")) == Pauli(f"-{xyz}")
