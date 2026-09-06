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

"""Executor-based expectation value calculation using Twirled Readout Error eXtinction (TREX) mitigation method."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qiskit_mitigation.mitigation_task import MitigationTask

import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import (
    Pauli,
    PauliLindbladMap,
    QubitSparsePauli,
    SparsePauliOp,
)
from samplomatic import build
from samplomatic.quantum_program import QuantumProgram, QuantumProgramResult, SamplexItem
from samplomatic.transpiler import generate_boxing_pass_manager


class TREX:
    """Mitigates readout errors in expectation values using the Twirled Readout Error eXtinction (TREX) method.

    The class enables adding a readout error calibration item to a QuantumProgram shared by one
    or more mitigation tasks, that can be executed on hardware using the Executor. The
    calibration results are post processed into a measurement noise model, which is used to
    calculate scale factors that correct the expectation values computed by the connected
    mitigation tasks.
    """

    VERSION = "0.1"

    def __init__(self):
        """Instantiate a TREX task."""
        self.tasks = []
        self.noise_model = None
        self._program_item_index = None

    def _edit_boxing_options(
        self, mitigation_task: MitigationTask, boxing_options: dict | None
    ) -> dict:
        """Edit boxing options."""
        self.tasks.append(mitigation_task)

        if boxing_options is None:
            return {"enable_measures": True, "measure_annotations": "all"}
        if "enable_measures" not in boxing_options:
            boxing_options["enable_measures"] = True
            boxing_options["measure_annotations"] = "all"
            return boxing_options
        if not boxing_options["enable_measures"]:
            raise ValueError('boxing_options["enable_measures"] may not be False.')
        if (
            "measure_annotations" in boxing_options
            and boxing_options["measure_annotations"] != "all"
        ):
            boxing_options["measure_annotations"] = "all"
        return boxing_options

    @staticmethod
    def create_instance_from_passthrough_data(passthrough: dict[str, Any]) -> TREX:
        """Create a MitigationTask instance from a passthrough dictionary loaded from a quantum program.

        Args:
            passthrough: Passthrough_data dictionary loaded from a quantum program.
            trex: A TREX instance containing a calibration circuit results executed in the same quantum program.
                Should remain ``None`` in case TREX mitigation was not used or a TREX calibration was not executed as
                part of thq same quantum program.

        Returns:
            A MitigationTask instance.
        """
        mitigation_type = passthrough.get("mitigation")
        if mitigation_type is None or mitigation_type != "trex":
            raise ValueError("'mitigation' field of the passthrough_data must be 'trex'")

        program_item_index = passthrough.get("program_item_index")
        trex_task = TREX()
        trex_task._program_item_index = program_item_index
        return trex_task

    def prepare(self, num_randomizations: int, quantum_program: QuantumProgram) -> QuantumProgram:
        """Prepare a TREX task."""
        if len(self.tasks) == 0:
            raise ValueError(
                "A TREX must be connected to at least one mitigation task to create a calibration task."
            )
        circuits = [task.circuit for task in self.tasks]
        self._program_item_index = len(quantum_program.items)
        quantum_program.items.append(
            self._prepare_calibration_circuit(circuits, num_randomizations)
        )

        data_for_passthrough = {
            "version": self.VERSION,
            "mitigation": "trex",
            "program_item_index": self._program_item_index,
        }

        self._add_data_to_passthrough_data(data_for_passthrough, quantum_program)
        return quantum_program

    def has_calibration_result(self) -> bool:
        """Whether a TREX calibration task has been added to the quantum program."""
        return self._program_item_index is not None

    @staticmethod
    def _add_data_to_passthrough_data(
        data: dict[str, Any], quantum_program: QuantumProgram
    ) -> None:
        """Add ``data`` into ``quantum_program`` passthrough_data."""
        if quantum_program.passthrough_data is None:
            quantum_program.passthrough_data = {"qiskit_mitigation": [data]}
        else:
            passthrough_data = quantum_program.passthrough_data
            if "qiskit_mitigation" not in passthrough_data:
                passthrough_data["qiskit_mitigation"] = [data]
            else:
                passthrough_data["qiskit_mitigation"].append(data)
        qiskit_mitigation_passthrough = quantum_program.passthrough_data["qiskit_mitigation"]
        for task_passthrough in qiskit_mitigation_passthrough:
            if task_passthrough.get("trex_calibration", None) is not None:
                task_passthrough["trex_calibration"] = True

    @staticmethod
    def _prepare_calibration_circuit(
        circuits: Sequence[QuantumCircuit],
        num_randomizations: int,
    ) -> SamplexItem:
        """Creates a TREX calibration circuit.

        The calibration circuit is based on all circuit terminal measurement layers in all pubs.

        Args:
            circuits: List of circuits to extract relevant qubits from.
            num_randomizations: The number of TREX calibration randomizations.

        Returns:
            Samplex item containing calibration circuit for TREX factors calculation.
        """
        # create the combined noise learning layer of all given inputs
        max_num_qubits = max(circuit.num_qubits for circuit in circuits)

        classical_cal_reg = ClassicalRegister(max_num_qubits, name="_trex_cal")
        trex_circuit = QuantumCircuit(max_num_qubits)
        trex_circuit.add_register(classical_cal_reg)
        trex_circuit.measure_all(add_bits=False)
        boxing_pm = generate_boxing_pass_manager(
            enable_gates=False,
            enable_measures=True,
            measure_annotations="twirl",
        )
        annotated_trex_circuit = boxing_pm.run(trex_circuit)
        template_trex_circuit, trex_samplex = build(annotated_trex_circuit)
        trex_calibration_item = SamplexItem(
            circuit=template_trex_circuit,
            samplex=trex_samplex,
            shape=(num_randomizations,),
        )

        return trex_calibration_item

    def compute_noise_model(self, results: QuantumProgramResult) -> PauliLindbladMap:
        """Compute noise model from program results.

        Args:
            results: QuantumProgramResult which contains the TREX calibration circuit.

        Returns:
            The learned readout noise model as a ``PauliLindbladMap``.
        """
        calibration_result = results[self._program_item_index]
        if "_trex_cal" not in calibration_result:
            raise ValueError("Dedicated TREX calibration circuit is missing from the results.")

        trex_noise_calibration_data = calibration_result["_trex_cal"]
        trex_calibration_measurement_flips = calibration_result["measurement_flips._trex_cal"]
        noise_calibration_data_flipped = np.logical_xor(
            trex_noise_calibration_data, trex_calibration_measurement_flips
        )
        # the shape of the calibration data is (randomizations, shots, measured_qubit)
        num_qubits = noise_calibration_data_flipped.shape[-1]
        noise_list = []
        for qubit_index in range(num_qubits):
            excited_state_count = np.sum(noise_calibration_data_flipped[:, :, qubit_index])
            total_shots = len(noise_calibration_data_flipped[:, :, qubit_index].flatten())
            flip_rate = excited_state_count / total_shots
            noise_list.append(("X", [qubit_index], flip_rate))
        readout_noise = PauliLindbladMap.from_sparse_list(noise_list, num_qubits=num_qubits)
        self.noise_model = readout_noise
        return readout_noise

    @staticmethod
    def calculate_trex_factor(
        noise_data: PauliLindbladMap | np.ndarray,
        observable_term: Pauli | QubitSparsePauli | str,
    ) -> float:
        """Calculate TREX factor relevant for a given observable term based on noise model.

        Args:
            noise_data: PauliLindbladMap containing measurement noise model or a result of TREX
                calibration execution.
            observable_term: observable term to calculate TREX factor for.

        Returns:
            TREX factor for the observable term.
        """
        if isinstance(observable_term, QubitSparsePauli):
            sparse_pauli = observable_term
        else:
            sparse_pauli = QubitSparsePauli(observable_term)
        if isinstance(noise_data, PauliLindbladMap):
            z_sparse_pauli = QubitSparsePauli(
                ("Z" * len(sparse_pauli.indices), sparse_pauli.indices),
                num_qubits=sparse_pauli.num_qubits,
            )
            trex_factor: float = 1 / noise_data.pauli_fidelity(z_sparse_pauli)
            return trex_factor
        # The input is a result of TREX calibration execution treat every non identity Pauli as Z
        evals = np.prod(1 - 2 * noise_data[..., sparse_pauli.indices], axis=-1)
        shots = (
            noise_data.shape[0] * noise_data.shape[-2]
        )  # randomizations * shots_per_randomizations

        # Compute trex factor
        trex_factor = 1 / (np.sum(evals) / shots)
        return trex_factor

    @staticmethod
    def trex_factors_each_term(
        measurement_noise_map: PauliLindbladMap,
        observables: ObservablesArray | Sequence[SparsePauliOp],
    ) -> dict[str, float]:
        r"""Calculates TREX mitigation algorithm's expectation value scale factor for each Pauli term in each observable.

        Calculates :math:`\langle Z^N \rangle` for each non-identity Pauli term in each observable using learned
        measurement noise, where :math:`N` are the non-identity indices in the term.

        Args:
            measurement_noise_map: Learned measurement noise in PauliLindbladMap format.
            observables: Observables in which the TREX algorithm mitigates their expectation values.

        Returns:
            A dictionary mapping Pauli terms to their expectation values scale factors.
        """
        observables = ObservablesArray.coerce(observables).ravel()

        scales_each_observable: dict[str, float] = {}
        for observable in observables:
            scales_each_observable.update(
                {
                    str(obs_term): TREX.calculate_trex_factor(measurement_noise_map, obs_term)
                    for obs_term in observable
                }
            )
        return scales_each_observable
