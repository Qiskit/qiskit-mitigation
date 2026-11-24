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

"""Tests for the LightCone commutation checker."""

import logging

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Qubit
from qiskit.circuit.library import CXGate, CZGate, PauliGate
from qiskit_mitigation.slc.bounds.light_cone import LightCone


def test_zxz(subtests):
    """Test commutation of different instructions with ZXZ Pauli.

    Args:
        subtests: the pytest-subtests fixture.
    """
    qubits = list([Qubit() for _ in range(3)])
    operations = [(PauliGate("ZXZ"), qubits)]
    lc = LightCone(set(qubits), operations, {})

    with subtests.test("cz(0,2)"):
        assert lc.commutes(CircuitInstruction(CZGate(), (qubits[0], qubits[2])))
        assert len(lc.operations) == 1

    with subtests.test("cx(0,1)"):
        assert lc.commutes(CircuitInstruction(CXGate(), (qubits[0], qubits[1])))
        assert len(lc.operations) == 1


def test_zzz(subtests):
    """Test commutation of different instructions with ZZZ Pauli.

    Args:
        subtests: the pytest-subtests fixture.
    """
    qubits = list([Qubit() for _ in range(3)])
    operations = [(PauliGate("ZZZ"), qubits)]
    lc = LightCone(set(qubits), operations, {})

    with subtests.test("cz(0,2)"):
        assert lc.commutes(CircuitInstruction(CZGate(), (qubits[0], qubits[2])))
        assert len(lc.operations) == 1

    with subtests.test("cx(0,1)"):
        assert not lc.commutes(CircuitInstruction(CXGate(), (qubits[0], qubits[1])))
        assert len(lc.operations) == 2


def test_large_operator_warning(caplog):
    """Test warning when checking commutation for > 10 qubits."""
    qubits = [Qubit() for _ in range(12)]
    operations = [(PauliGate("Z" * 12), qubits)]
    lc = LightCone(set(qubits), operations, {})

    with caplog.at_level(logging.WARNING, logger="qiskit_mitigation.slc.bounds.light_cone"):
        lc.commutes(CircuitInstruction(PauliGate("X" * 12), qubits))
    assert any(
        "Checking commutation of operators of size" in r.getMessage() for r in caplog.records
    )


def test_initialize_from_measurements():
    """Test initialize_from_measurements options."""
    qc = QuantumCircuit(3)
    qc.x(0)

    lc_all = LightCone.initialize_from_measurements(qc, measure_all=True)
    assert len(lc_all.qubits) == 3

    lc_active = LightCone.initialize_from_measurements(qc, measure_active=True)
    assert len(lc_active.qubits) == 1

    with pytest.raises(NotImplementedError):
        LightCone.initialize_from_measurements(qc, measure_all=False, measure_active=False)
