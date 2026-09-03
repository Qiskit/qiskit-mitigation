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

"""Tests for the measurement_bases module."""

import unittest

import numpy as np
import pytest
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from qiskit_mitigation.utils.measurement_bases import (
    PAULI_TO_INT_LOOKUP_TABLE,
    _convert_basis_to_uint_representation,
    _convert_pauli_basis,
    _identify_measure_basis,
    _ints_to_pauli,
    _meas_basis_for_pauli_group,
    _pauli_to_ints,
    get_measurement_bases,
)


class TestGetMeasurementBases(unittest.TestCase):
    """Tests for get_measurement_bases function."""

    def test_single_observable_single_pauli(self):
        """Test with a single observable containing a single Pauli term."""
        obs = SparsePauliOp("ZZZ", 1.0)
        bases, reverser = get_measurement_bases(obs)

        self.assertEqual(len(bases), 1)
        self.assertEqual(len(reverser), 1)
        np.testing.assert_array_equal(bases[0], np.array([1, 1, 1], dtype=np.uint8))

        # Check reverser structure
        basis_pauli = next(iter(reverser.keys()))
        self.assertEqual(basis_pauli, Pauli("ZZZ"))
        self.assertEqual(len(reverser[basis_pauli]), 1)
        self.assertIsInstance(reverser[basis_pauli][0], SparsePauliOp)

    def test_single_observable_multiple_paulis(self):
        """Test with a single observable containing multiple Pauli terms."""
        obs = SparsePauliOp(["ZZI", "IZZ", "ZIZ"], [1.0, 2.0, 3.0])
        bases, _ = get_measurement_bases(obs)

        # All Z-type Paulis should commute and be in one basis
        self.assertEqual(len(bases), 1)
        np.testing.assert_array_equal(bases[0], np.array([1, 1, 1], dtype=np.uint8))

    def test_multiple_observables(self):
        """Test with multiple observables."""
        obs1 = SparsePauliOp("ZZZ", 1.0)
        obs2 = SparsePauliOp("XXX", 2.0)
        bases, reverser = get_measurement_bases([obs1, obs2])

        # Z and X don't commute qubit-wise, so we need 2 bases
        self.assertEqual(len(bases), 2)
        self.assertEqual(len(reverser), 2)

        # Check that each basis maps to a list with 2 elements (one per observable)
        for _, obs_list in reverser.items():
            self.assertEqual(len(obs_list), 2)

    def test_commuting_paulis_grouped(self):
        """Test that commuting Paulis are grouped into the same basis."""
        obs = SparsePauliOp(["ZII", "IZI", "IIZ"], [1.0, 1.0, 1.0])
        bases, _ = get_measurement_bases(obs)

        # All should be in one basis since they commute qubit-wise
        self.assertEqual(len(bases), 1)

    def test_non_commuting_paulis_separate_bases(self):
        """Test that non-commuting Paulis get separate bases."""
        obs = SparsePauliOp(["ZI", "XI"], [1.0, 1.0])
        bases, _ = get_measurement_bases(obs)

        # These don't commute qubit-wise, so need separate bases
        self.assertEqual(len(bases), 2)

    def test_identity_terms(self):
        """Test handling of identity terms."""
        obs = SparsePauliOp(["III", "ZZZ"], [1.0, 2.0])
        bases, _ = get_measurement_bases(obs)

        # Identity commutes with everything
        self.assertGreaterEqual(len(bases), 1)

    def test_empty_observable_list(self):
        """Test with an empty list of observables."""
        # Empty list causes sum() to return 0, which doesn't have .unique() method
        # This is expected behavior - function requires at least one observable
        with pytest.raises(AttributeError):
            _, _ = get_measurement_bases([])

    def test_reverser_none_values(self):
        """Test that reverser contains None for observables without terms in a basis."""
        obs1 = SparsePauliOp("ZZ", 1.0)
        obs2 = SparsePauliOp("XX", 2.0)
        _, reverser = get_measurement_bases([obs1, obs2])

        # Each basis should have one observable with terms and one with None
        for _, obs_list in reverser.items():
            non_none_count = sum(1 for obs in obs_list if obs is not None)
            self.assertEqual(non_none_count, 1)


class TestMeasBasisForPauliGroup(unittest.TestCase):
    """Tests for _meas_basis_for_pauli_group function."""

    def test_single_z_pauli(self):
        """Test with a single Z Pauli."""
        group = PauliList(["ZII"])
        basis = _meas_basis_for_pauli_group(group)
        self.assertEqual(basis, Pauli("ZII"))

    def test_single_x_pauli(self):
        """Test with a single X Pauli."""
        group = PauliList(["XII"])
        basis = _meas_basis_for_pauli_group(group)
        self.assertEqual(basis, Pauli("XII"))

    def test_single_y_pauli(self):
        """Test with a single Y Pauli."""
        group = PauliList(["YII"])
        basis = _meas_basis_for_pauli_group(group)
        self.assertEqual(basis, Pauli("YII"))

    def test_multiple_z_paulis(self):
        """Test with multiple Z Paulis."""
        group = PauliList(["ZII", "IZI", "IIZ"])
        basis = _meas_basis_for_pauli_group(group)
        self.assertEqual(basis, Pauli("ZZZ"))

    def test_mixed_paulis(self):
        """Test with mixed Pauli types."""
        group = PauliList(["ZI", "IX"])
        basis = _meas_basis_for_pauli_group(group)
        self.assertEqual(basis, Pauli("ZX"))

    def test_identity_in_group(self):
        """Test with identity in the group."""
        group = PauliList(["III", "ZII"])
        basis = _meas_basis_for_pauli_group(group)
        self.assertEqual(basis, Pauli("ZII"))

    def test_overlapping_paulis(self):
        """Test with overlapping Pauli positions."""
        group = PauliList(["ZZI", "ZIZ"])
        basis = _meas_basis_for_pauli_group(group)
        self.assertEqual(basis, Pauli("ZZZ"))


class TestConvertBasisToUintRepresentation(unittest.TestCase):
    """Tests for _convert_basis_to_uint_representation function."""

    def test_single_identity(self):
        """Test conversion of identity."""
        bases = PauliList(["I"])
        result = _convert_basis_to_uint_representation(bases)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], np.array([0], dtype=np.uint8))

    def test_single_z(self):
        """Test conversion of Z."""
        bases = PauliList(["Z"])
        result = _convert_basis_to_uint_representation(bases)
        np.testing.assert_array_equal(result[0], np.array([1], dtype=np.uint8))

    def test_single_x(self):
        """Test conversion of X."""
        bases = PauliList(["X"])
        result = _convert_basis_to_uint_representation(bases)
        np.testing.assert_array_equal(result[0], np.array([2], dtype=np.uint8))

    def test_single_y(self):
        """Test conversion of Y."""
        bases = PauliList(["Y"])
        result = _convert_basis_to_uint_representation(bases)
        np.testing.assert_array_equal(result[0], np.array([3], dtype=np.uint8))

    def test_multi_qubit_pauli(self):
        """Test conversion of multi-qubit Pauli."""
        bases = PauliList(["IXYZ"])
        result = _convert_basis_to_uint_representation(bases)
        # Note: reversed order (little-endian) - IXYZ becomes Z,Y,X,I
        np.testing.assert_array_equal(result[0], np.array([1, 3, 2, 0], dtype=np.uint8))

    def test_multiple_bases(self):
        """Test conversion of multiple bases."""
        bases = PauliList(["ZZ", "XX", "YY"])
        result = _convert_basis_to_uint_representation(bases)
        self.assertEqual(len(result), 3)
        np.testing.assert_array_equal(result[0], np.array([1, 1], dtype=np.uint8))
        np.testing.assert_array_equal(result[1], np.array([2, 2], dtype=np.uint8))
        np.testing.assert_array_equal(result[2], np.array([3, 3], dtype=np.uint8))

    def test_dtype_is_uint8(self):
        """Test that output dtype is uint8."""
        bases = PauliList(["XYZ"])
        result = _convert_basis_to_uint_representation(bases)
        self.assertEqual(result[0].dtype, np.uint8)


class TestIdentifyMeasureBasis(unittest.TestCase):
    """Tests for :func:`_identify_measure_basis`."""

    def _bases(self, *labels):
        """Build a list of (Pauli, config_idx) from label strings."""
        return [(Pauli(lbl), idx) for idx, lbl in enumerate(labels)]

    def test_exact_single_qubit_match(self):
        """A single-qubit Z Pauli is found in a Z-basis list."""
        bases = self._bases("Z")
        self.assertEqual(_identify_measure_basis(Pauli("Z"), bases), 0)

    def test_identity_matches_any_basis(self):
        """An all-identity Pauli has no support, so the first basis always matches."""
        bases = self._bases("X", "Z")
        self.assertEqual(_identify_measure_basis(Pauli("I"), bases), 0)

    def test_correct_index_returned_for_second_basis(self):
        """When only the second basis matches, config_idx 1 must be returned."""
        bases = self._bases("X", "Z")
        self.assertEqual(_identify_measure_basis(Pauli("Z"), bases), 1)

    def test_multi_qubit_partial_support(self):
        """ZI Pauli (support only on qubit 1) matches a ZZ basis."""
        bases = self._bases("ZZ")
        # ZI: Z on qubit-1, I on qubit-0  → support only at qubit-1, matches ZZ basis there
        self.assertEqual(_identify_measure_basis(Pauli("ZI"), bases), 0)

    def test_raises_when_no_basis_matches(self):
        """A Z Pauli against an X-only basis list must raise ValueError."""
        bases = self._bases("X")
        with self.assertRaises(ValueError):
            _identify_measure_basis(Pauli("Z"), bases)

    def test_raises_empty_basis_list(self):
        """An empty basis list always raises ValueError."""
        with self.assertRaises(ValueError):
            _identify_measure_basis(Pauli("Z"), [])

    def test_first_matching_index_is_returned(self):
        """When multiple bases match, the first one's config_idx is returned."""
        bases = self._bases("Z", "Z")
        self.assertEqual(_identify_measure_basis(Pauli("Z"), bases), 0)

    def test_x_pauli_does_not_match_z_basis(self):
        """An X Pauli does not match a Z measurement basis."""
        bases = self._bases("Z")
        with self.assertRaises(ValueError):
            _identify_measure_basis(Pauli("X"), bases)

    def test_y_pauli_matches_y_basis(self):
        """A Y Pauli matches a Y measurement basis."""
        bases = self._bases("Y")
        self.assertEqual(_identify_measure_basis(Pauli("Y"), bases), 0)


class TestGetPauliBasis(unittest.TestCase):
    """Tests for :func:`_convert_pauli_basis`."""

    def test_zeros_map_to_z(self):
        self.assertEqual(_convert_pauli_basis("0"), Pauli("Z"))

    def test_ones_map_to_z(self):
        self.assertEqual(_convert_pauli_basis("1"), Pauli("Z"))

    def test_plus_maps_to_x(self):
        self.assertEqual(_convert_pauli_basis("+"), Pauli("X"))

    def test_minus_maps_to_x(self):
        self.assertEqual(_convert_pauli_basis("-"), Pauli("X"))

    def test_r_maps_to_y(self):
        self.assertEqual(_convert_pauli_basis("r"), Pauli("Y"))

    def test_l_maps_to_y(self):
        self.assertEqual(_convert_pauli_basis("l"), Pauli("Y"))

    def test_mixed_basis_string(self):
        """'0+r' → 'ZXY'."""
        self.assertEqual(_convert_pauli_basis("0+r"), Pauli("ZXY"))

    def test_multi_qubit_all_z(self):
        self.assertEqual(_convert_pauli_basis("000"), Pauli("ZZZ"))

    def test_returns_pauli_instance(self):
        result = _convert_pauli_basis("0")
        self.assertIsInstance(result, Pauli)


class TestPauliToInts(unittest.TestCase):
    """Tests for :func:`_pauli_to_ints`."""

    def test_identity_maps_to_zero(self):
        self.assertEqual(_pauli_to_ints(Pauli("I")), [0])

    def test_z_maps_to_one(self):
        self.assertEqual(_pauli_to_ints(Pauli("Z")), [1])

    def test_x_maps_to_two(self):
        self.assertEqual(_pauli_to_ints(Pauli("X")), [2])

    def test_y_maps_to_three(self):
        self.assertEqual(_pauli_to_ints(Pauli("Y")), [3])

    def test_two_qubit_iz(self):
        """IZ (qubit-0 = Z, qubit-1 = I) → little-endian: [1, 0]."""
        self.assertEqual(_pauli_to_ints(Pauli("IZ")), [1, 0])

    def test_two_qubit_xi(self):
        """XI (qubit-0 = I, qubit-1 = X) → little-endian: [0, 2]."""
        self.assertEqual(_pauli_to_ints(Pauli("XI")), [0, 2])

    def test_three_qubit_xyz(self):
        """XYZ → label is 'XYZ' (big-endian) → reversed → [Z,Y,X] = [1,3,2]."""
        self.assertEqual(_pauli_to_ints(Pauli("XYZ")), [1, 3, 2])

    def test_returns_list(self):
        self.assertIsInstance(_pauli_to_ints(Pauli("Z")), list)

    def test_all_identity(self):
        self.assertEqual(_pauli_to_ints(Pauli("III")), [0, 0, 0])

    def test_lookup_table_completeness(self):
        """Every key in LOOKUP_TABLE must produce a valid integer output."""
        for char, expected_val in PAULI_TO_INT_LOOKUP_TABLE.items():
            with self.subTest(char=char):
                result = _pauli_to_ints(Pauli(char))
                self.assertEqual(result, [expected_val])


class TestIntsToPauli(unittest.TestCase):
    """Tests for :func:`_ints_to_pauli`."""

    def test_identity_list_gives_identity_pauli(self):
        """A list of all zeros maps to the all-identity Pauli."""
        result = _ints_to_pauli([0])
        self.assertEqual(result, Pauli("I"))

    def test_z_int_gives_z_pauli(self):
        """A list ``[1]`` maps to a Z Pauli."""
        result = _ints_to_pauli([1])
        self.assertEqual(result, Pauli("Z"))

    def test_x_int_gives_x_pauli(self):
        """A list ``[2]`` maps to an X Pauli."""
        result = _ints_to_pauli([2])
        self.assertEqual(result, Pauli("X"))

    def test_y_int_gives_y_pauli(self):
        """A list ``[3]`` maps to a Y Pauli."""
        result = _ints_to_pauli([3])
        self.assertEqual(result, Pauli("Y"))

    def test_two_qubit_ints_roundtrip(self):
        """``[1, 2]`` (little-endian: qubit-0=Z, qubit-1=X) → reversed label ``'XZ'``."""
        result = _ints_to_pauli([1, 2])
        self.assertEqual(result, Pauli("XZ"))

    def test_pauli_to_ints_roundtrip(self):
        """``_ints_to_pauli(_pauli_to_ints(p))`` must recover the original Pauli."""
        for label in ("I", "X", "Y", "Z", "XYZ", "ZZI"):
            p = Pauli(label)
            ints = _pauli_to_ints(p)
            recovered = _ints_to_pauli(ints)
            self.assertEqual(recovered, p, msg=f"roundtrip failed for {label}")


if __name__ == "__main__":
    unittest.main()

# Made with Bob
