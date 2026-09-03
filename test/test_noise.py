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

"""Tests for the ``noise`` module."""

from __future__ import annotations

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import BoxOp
from qiskit.quantum_info import Clifford, PauliLindbladMap, PauliList, random_pauli_list
from qiskit_mitigation.noise import create_postselected_noise_mask
from samplomatic.annotations import InjectionSite, InjectNoise

# Two-qubit generator set used by the hand-worked tests, in ``to_sparse_list`` convention.
_GENS = (("X", [0]), ("Z", [0]), ("X", [1]), ("Y", [1]), ("Z", [1]), ("XX", [0, 1]))


def _inject(ref: str, modifier_ref: str, site: InjectionSite = InjectionSite.BEFORE) -> InjectNoise:
    """An ``InjectNoise`` with the given site, tolerating the released (site-less) samplomatic."""
    try:
        return InjectNoise(ref, modifier_ref, site=site)
    except TypeError:  # released samplomatic: __slots__ forbids per-instance sites
        return type("_Sited", (InjectNoise,), {"__slots__": (), "site": site})(ref, modifier_ref)


def _boxed(num_qubits: int, boxes: list[tuple[QuantumCircuit, list[int], InjectNoise | None]]):
    """A circuit of annotated boxes."""
    circuit = QuantumCircuit(num_qubits)
    for body, qargs, annotation in boxes:
        circuit.append(BoxOp(body, annotations=[annotation] if annotation else []), qargs)
    return circuit


def _oracle(circuit, noise_maps, detectors):
    """Brute-force reference: signed ``Pauli.evolve`` back-propagation and pairwise
    ``anticommutes`` checks, plus the expected sampling cost. Independent of the sign-free
    GF(2) machinery under test."""
    num_qubits = circuit.num_qubits
    detectors = list(PauliList(detectors))
    scales, kept = {}, 0.0

    def step(operation, qargs):
        nonlocal detectors
        if operation.name != "barrier":
            detectors = [d.evolve(Clifford(operation), qargs=qargs, frame="h") for d in detectors]

    def mask(rates, qubits):
        out = []
        for pauli, indices, _ in rates.to_sparse_list():
            label = ["I"] * num_qubits
            for position, char in enumerate(pauli):
                label[num_qubits - 1 - qubits[indices[position]]] = char
            generator = PauliList(["".join(label)])[0]
            out.append(0.0 if any(d.anticommutes(generator) for d in detectors) else 1.0)
        return np.array(out)

    for instruction in reversed(circuit.data):
        operation = instruction.operation
        qargs = [circuit.find_bit(q).index for q in instruction.qubits]
        if operation.name != "box":
            step(operation, qargs)
            continue
        inject = next((a for a in operation.annotations if isinstance(a, InjectNoise)), None)
        body = operation.blocks[0]
        for site, before_body in [(InjectionSite.AFTER, True), (InjectionSite.BEFORE, False)]:
            if not before_body:
                for sub in reversed(body.data):
                    step(sub.operation, [qargs[body.find_bit(q).index] for q in sub.qubits])
            if inject is not None and inject.ref and inject.site == site:
                m = mask(noise_maps[inject.ref], sorted(qargs))
                scales[inject.modifier_ref] = m
                kept += noise_maps[inject.ref].rates @ m
    return scales, float(np.exp(4 * kept))


class TestComputeLocalScales(unittest.TestCase):
    def setUp(self):
        self.rates = PauliLindbladMap.from_sparse_list(
            [(p, q, 0.01) for p, q in _GENS], num_qubits=2
        )

    def test_hand_worked_masks(self):
        """A CX box with a Z detector, verified against pencil and paper.

        Z on the CX target back-propagates to ZZ, so single X/Y generators anticommute
        (scale 0) while XX survives: symplectic products count parity, not local clashes.
        """
        body = QuantumCircuit(2)
        body.cx(0, 1)
        body.barrier()
        circuit = _boxed(2, [(body, [0, 1], _inject("layer", "mod"))])
        scales, cost = create_postselected_noise_mask(circuit, {"layer": self.rates}, ["ZI"])
        np.testing.assert_array_equal(scales["mod"], [0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(cost, np.exp(4 * 3 * 0.01))

    def test_injection_site(self):
        """BEFORE and AFTER shade opposite sides of the body, giving different masks."""
        body = QuantumCircuit(1)
        body.h(0)
        one_q = PauliLindbladMap.from_sparse_list([("X", [0], 0.01), ("Z", [0], 0.01)], 1)
        for site, expected in [
            (InjectionSite.AFTER, [0.0, 1.0]),
            (InjectionSite.BEFORE, [1.0, 0.0]),
        ]:
            circuit = _boxed(1, [(body, [0], _inject("layer", "mod", site))])
            scales, _ = create_postselected_noise_mask(circuit, {"layer": one_q}, ["Z"])
            np.testing.assert_array_equal(scales["mod"], expected, err_msg=str(site))

    def test_measure_and_reset(self):
        """Z support passes measurements exactly; a reset erases earlier support so preceding
        generators survive; X/Y support crossing either instruction is loudly rejected."""
        detected = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0])  # Z-on-qubit-0 detector vs _GENS
        for mid, early in [("measure", detected), ("reset", np.ones(6))]:
            circuit = QuantumCircuit(2, 2)
            circuit.append(
                BoxOp(QuantumCircuit(2), annotations=[_inject("layer", "early")]), [0, 1]
            )
            circuit.measure(0, 0) if mid == "measure" else circuit.reset(0)
            circuit.append(BoxOp(QuantumCircuit(2), annotations=[_inject("layer", "late")]), [0, 1])
            circuit.measure(1, 1)
            scales, cost = create_postselected_noise_mask(circuit, {"layer": self.rates}, ["IZ"])
            np.testing.assert_array_equal(scales["late"], detected, err_msg=mid)
            np.testing.assert_array_equal(scales["early"], early, err_msg=mid)
            self.assertAlmostEqual(cost, np.exp(4 * 0.01 * (early.sum() + detected.sum())), msg=mid)
            with self.assertRaisesRegex(ValueError, "X or Y support"):
                create_postselected_noise_mask(circuit, {"layer": self.rates}, ["IX"])

    def test_against_brute_force(self):
        """Randomized sweep against the signed-Pauli oracle: subset boxes with shuffled
        qargs, unannotated boxes, barriers, loose top-level gates, uncommon Clifford gates,
        and both injection sites."""
        one_q, two_q = (
            ["h", "s", "sdg", "sx", "x", "y", "z", "id"],
            ["cx", "cz", "swap", "ecr", "iswap"],
        )
        for seed in range(15):
            rng = np.random.default_rng(seed)
            num_qubits = int(rng.integers(3, 6))
            boxes, noise_maps = [], {}
            for index in range(int(rng.integers(2, 5))):
                width = int(rng.integers(2, num_qubits + 1))
                qargs = [int(q) for q in rng.permutation(num_qubits)[:width]]
                body = QuantumCircuit(width)
                for q in range(width):
                    getattr(body, rng.choice(one_q))(q)
                body.barrier()
                getattr(body, rng.choice(two_q))(0, width - 1)
                annotation = None
                if rng.random() < 0.8:
                    ref = f"layer{index}"
                    site = InjectionSite.AFTER if rng.random() < 0.5 else InjectionSite.BEFORE
                    annotation = _inject(ref, f"mod{index}", site)
                    noise_maps[ref] = PauliLindbladMap.from_sparse_list(
                        [
                            (
                                "".join(rng.choice(list("XYZ"), size=2)),
                                [0, width - 1],
                                float(rng.random()),
                            )
                            for _ in range(4)
                        ]
                        + [("Y", [int(rng.integers(width))], float(rng.random()))],
                        num_qubits=width,
                    )
                boxes.append((body, qargs, annotation))
            circuit = _boxed(num_qubits, boxes)
            getattr(circuit, rng.choice(one_q))(int(rng.integers(num_qubits)))  # loose gate
            circuit.barrier()
            detectors = random_pauli_list(
                num_qubits, int(rng.integers(1, 4)), seed=seed, phase=False
            )
            scales, cost = create_postselected_noise_mask(circuit, noise_maps, detectors)
            expected_scales, expected_cost = _oracle(circuit, noise_maps, detectors)
            self.assertEqual(scales.keys(), expected_scales.keys(), msg=f"{seed=}")
            for key, value in expected_scales.items():
                np.testing.assert_array_equal(scales[key], value, err_msg=f"{seed=} {key=}")
            self.assertAlmostEqual(cost, expected_cost, msg=f"{seed=}")

    def test_rotation_gates(self):
        """Axis-commuting rotations act trivially; anticommuting detectors are rejected."""
        body, plain = QuantumCircuit(2), QuantumCircuit(2)
        for target in (body, plain):
            target.cx(0, 1)
        body.rz(0.7, 1)
        body.rzz(0.7, 0, 1)
        with_rotations = _boxed(2, [(body, [0, 1], _inject("layer", "mod"))])
        without = _boxed(2, [(plain, [0, 1], _inject("layer", "mod"))])
        for detectors in (["ZI"], ["IZ", "ZZ"]):
            expected, _ = create_postselected_noise_mask(without, {"layer": self.rates}, detectors)
            scales, _ = create_postselected_noise_mask(
                with_rotations, {"layer": self.rates}, detectors
            )
            np.testing.assert_array_equal(scales["mod"], expected["mod"], err_msg=str(detectors))
        with self.assertRaisesRegex(ValueError, "anticommutes"):
            create_postselected_noise_mask(with_rotations, {"layer": self.rates}, ["XI"])
        unsupported = QuantumCircuit(2)
        unsupported.u(0.1, 0.2, 0.3, 0)
        circuit = _boxed(2, [(unsupported, [0, 1], _inject("layer", "mod"))])
        with self.assertRaisesRegex(ValueError, "non-Clifford"):
            create_postselected_noise_mask(circuit, {"layer": self.rates}, ["ZZ"])

    def test_misuse_errors(self):
        """Detector width mismatches and missing noise-map entries raise informatively."""
        circuit = _boxed(2, [(QuantumCircuit(2), [0, 1], _inject("layer", "mod"))])
        with self.assertRaisesRegex(ValueError, "3 qubits but"):
            create_postselected_noise_mask(circuit, {"layer": self.rates}, ["ZZZ"])
        with self.assertRaisesRegex(ValueError, "'layer'"):
            create_postselected_noise_mask(circuit, {"other": self.rates}, ["ZZ"])

    def test_trivial_detector_is_full_pec(self):
        """An identity detector removes nothing: all-ones masks and the full-PEC gamma^2."""
        body = QuantumCircuit(2)
        body.h(0)
        body.cx(0, 1)
        circuit = _boxed(2, [(body, [0, 1], _inject("layer", "mod"))])
        scales, cost = create_postselected_noise_mask(circuit, {"layer": self.rates}, ["II"])
        np.testing.assert_array_equal(scales["mod"], np.ones(len(_GENS)))
        self.assertAlmostEqual(cost, np.exp(4 * np.sum(self.rates.rates)))


if __name__ == "__main__":
    unittest.main()
