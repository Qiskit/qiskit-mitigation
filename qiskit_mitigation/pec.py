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

"""Executor-based expectation value calculation using Probabilistic Error Cancellation (PEC) mitigation method."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import DataBin, PubResult
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import Pauli, PauliLindbladMap, PauliList, SparsePauliOp
from samplomatic import InjectNoise, build
from samplomatic.quantum_program import (
    QuantumProgram,
    QuantumProgramItemResult,
    QuantumProgramResult,
    SamplexItem,
)
from samplomatic.utils import get_annotation

from qiskit_mitigation.mitigation_task import MitigationTask
from qiskit_mitigation.trex import TREX
from qiskit_mitigation.utils.expectation_values import (
    _compute_single_term_exp_val_with_twirl_variance,
    executor_expectation_values,
)
from qiskit_mitigation.utils.measurement_bases import (
    _convert_pauli_basis,
    _identify_measure_basis,
)


class PEC(MitigationTask):
    """Calculates expectation values of observables using Probabilistic Error Cancellation (PEC) mitigation method.

    The class enables preparing a QuantumProgram with circuit mitigated using PEC method,
    that can be executed on hardware using the Executor, and post process the results to calculate mitigated
    expectation values of given observables.
    The task parameters should be given as input to the prepare function, and the relevant
    variables needed for post-processing are saved internally:

    .. code-block:: python

        pec = PEC()
        program = pec.prepare(circuit=circuit,
                              observables=observables,
                              parameters=parameter_valuesת
                              noise_maps=learned_noise)
        job = executor.run(program)
        results = job.result()
        mitigated_result = pec.postprocess(results)

    To calculate expectation values of a loaded job result, a PEC task can be created from the job results with all the internal variables needed for post-processing.
    Alternatively, the variables required for post-processing can be given as input directly to the ``compute_expectation_value`` static method.
    Example for running post-processing for a loaded result:

    .. code-block:: python

        pec = PEC()
        program = pec.prepare(circuit=circuit,
                              observables=observables,
                              parameters=parameter_valuesת
                              noise_maps=learned_noise)
        job_id = executor.run(program).job_id

        job = service.job(job_id)
        results = job.result()
        pec = load_tasks_from_result(results)[0]
        mitigated_result = pec.postprocess(results)

    """

    def __init__(self):
        """Instantiate a PEC task."""
        super().__init__()
        self.gamma = None

    @classmethod
    def _box_circuit(
        cls,
        circuit: QuantumCircuit,
        boxing_options: dict | None,
    ) -> QuantumCircuit:
        """Group the operations in the given ``circuit`` into boxes.

        This function removes the final measurement layer from the given circuit and adds a new
        measurement layer with a dedicated register name. Then, it uses the
        :meth:`~samplomatic.transpiler.generate_boxing_pass_manager` to group the operations in a
        circuit into boxes.
        The function forces the boxing options to contain options that are needed for PEC mitigated circuit.

        Args:
            circuit: The circuit to group into boxes.
            boxing_options: Dictionary of boxing options. If ``None``, default boxing options are used.

        Returns:
            The boxed circuit.

        Raises:
            ValueError: If ``boxing_options`` contradict required boxing PEC parameters -
                ``boxing_options["enable_measures"]`` is False,
                ``boxing_options["enable_gates"]`` is False,
                ``boxing_options["inject_noise_targets"]`` not in ["gates", "all"],
                ``boxing_options["inject_noise_strategy"]``is "no_modification".
            ValueError: If the boxing pass manager fails to run.
        `
        """
        if boxing_options is None:
            boxing_options = {}
        # Force PEC related options
        if "enable_gates" not in boxing_options:
            boxing_options["enable_gates"] = True
        elif not boxing_options["enable_gates"]:
            raise ValueError('boxing_options["enable_gates"] may not be False')
        if "inject_noise_targets" not in boxing_options:
            boxing_options["inject_noise_targets"] = "gates"
        elif boxing_options["inject_noise_targets"] not in ["gates", "all"]:
            raise ValueError(
                'boxing_options["inject_noise_targets"] must be one of "gates" or "all".'
            )
        if "inject_noise_strategy" not in boxing_options:
            boxing_options["inject_noise_strategy"] = "uniform_modification"
        elif boxing_options["inject_noise_strategy"] == "no_modification":
            raise ValueError(
                'boxing_options["inject_noise_strategy"] may not be ``no_modification``.'
            )

        return super()._box_circuit(circuit, boxing_options)

    def prepare(
        self,
        circuit: QuantumCircuit,
        observables: ObservablesArray | Sequence[SparsePauliOp],
        parameters: np.ndarray | BindingsArray | None,
        custom_boxing_options: dict | None = None,
        shots_per_randomization: int = 64,
        num_randomizations: int = 128,
        broadcast_obs_and_params: bool = False,
        trex: TREX | None = None,
        quantum_program: QuantumProgram | None = None,
        *,
        noise_maps: dict[str, PauliLindbladMap] | None = None,
        scale_randomizations_by_gamma: bool = True,
        noise_gain: float | Literal["auto"] = "auto",
        max_sampling_overhead: float | None = 100,
    ) -> QuantumProgram:
        """Creates a :class:`~.QuantumProgram` with PEC mitigated item for executing via Executor.

        Creates an item for a :class:`~.QuantumProgram`, that can be executed via Executor and is PEC mitigated
        (injects inverse noise to cancel the learned noise).
        If a ``quantum_program`` is provided, the new item will be added to the existing program,
        otherwise, a new program will be created, containing only the created item.
        If ``broadcast_obs_and_params`` is True, the observables and parameters will be broadcasted
        using the samplomatic broadcasting rules, to allow attaching some of the parameters to some
        of the observables. Otherwise, every parameter will be executed for each observable (outer
        product of the parameters and observables). Note that the post-processing of an outer
        product is usually faster.

        Args:
            circuit: The quantum circuit.
            observables: The observables to calculate their expectation values.
            parameters: The parameters of a parametric circuit.
            custom_boxing_options: The custom boxing options that will be used by :meth:`~samplomatic.transpiler.generate_boxing_pass_manager` function.
            shots_per_randomization: The number of shots per randomization.
            num_randomizations: The number of randomizations.
            broadcast_obs_and_params: Whether to broadcast observables and parameters.
            trex: A TREX mitigation instance that will be used to mitigate readout errors.
            quantum_program: The quantum program to add an item for. If None, a new program will be created.
            noise_maps: A mapping between layer ref to a noise model to use for PEC mitigation
                method. The dict might contain layers not present in the given circuit, but must contain all
                the mitigated layers. Assumes that the unique layers used for noise learning were extracted
                using the ``find_unique_layers`` method with the same custom boxing options.
            scale_randomizations_by_gamma: Whether to automatically scale the number of randomizations by gamma**2.
            noise_gain: The fraction of noise to keep after the mitigation.
                A value of ``0`` corresponds to removing the full learned noise.
                A value of ``1`` corresponds to no removal of the learned noise.
                A value between ``0`` and ``1`` corresponds to partially removing the learned noise.
                A value greater than one corresponds to amplifying the learned noise.
                If ``"auto"``, the value in the range ``[0, 1]`` will be chosen automatically by the formula ``1 - log(max_overhead) / log(gamma^2)``.
            max_sampling_overhead: If scale_randomizations_by_gamma is True, limit the multiplicative ratio of the number of randomizations.

        Returns:
            A :class:`~.QuantumProgram` with PEC mitigated item that can be executed via Executor.
        """
        if noise_maps is None:
            raise ValueError("noise_maps is missing")

        # add measurement twirls to the task if TREX is on
        if trex is not None:
            if self._has_projection_operators(observables):
                raise ValueError(
                    "TREX mitigation is currently not supported when observables contain "
                    "projection operators. You can decompose into pauli operators if the "
                    "exponential cost is acceptable."
                )
            custom_boxing_options = trex._edit_boxing_options(self, custom_boxing_options)
        # save variables in the class
        self._save_basic_variables(
            circuit,
            observables,
            parameters,
            custom_boxing_options,
            shots_per_randomization,
            num_randomizations,
            broadcast_obs_and_params,
            trex,
            quantum_program,
        )

        boxed_circuit = self._box_circuit(self.circuit, self.custom_boxing_options)
        self.boxed_circuit = boxed_circuit
        # Build the template and the samplex
        template, samplex = build(boxed_circuit)

        # Prepare samplex_arguments
        samplex_arguments, shape = self._create_samplex_arguments(samplex, boxed_circuit)

        # set max_overhead
        if max_sampling_overhead is None:
            # This is a backup max number of shots, intended to stop python
            # crashing with an overflow error if the noise is really strong
            max_sampling_overhead = sys.float_info.max / (
                num_randomizations * shots_per_randomization
            )

        # noise_gain - the user facing parameter reflecting "how much noise remains after removal".
        # noise_scale - samplomatic parameter reflecting "how much noise is injected". The noise
        # is injected as quasi-probability and should be negative for removing noise (-1 is full
        # removal of the noise and 0 is no rescaling of the noise).
        # noise_factor - factor for scaled gamma calculation, reflecting the factor by which the
        # noise should be multiplied.

        # add samplex_arguments related to noise injection
        if isinstance(noise_gain, str):  # noise_gain == "auto"
            # calculate the gamma factor without scaling it by noise_factor
            gamma = self.calculate_gamma(boxed_circuit, noise_maps, 1)
            # calculate the noise factor based on gamma and max_sampling_overhead, setting it
            # to ``1`` if ``gamma`` is ``1``--i.e., if there is no noise to mitigate.
            gain = (
                1.0 if gamma == 1 else float(1 - np.log(max_sampling_overhead) / np.log(gamma**2))
            )
            # Truncate to [0, 1]
            gain = min(1.0, max(0.0, gain))
        else:
            gain = float(noise_gain)

        # Adjusting noise_scale to [-1, 0] range from the [0, 1] range of noise_gain
        noise_scale = gain - 1
        # The sampling scaling is proportional to 1 - noise_gain, as 0 is full PEC and 1 is no PEC
        noise_factor = 1 - gain

        # Create a noise model map containing only the layers relevant for the given circuit
        specs = samplex.inputs().get_specs("pauli_lindblad_maps")
        circuit_noise_model = {}
        for spec in specs:
            ref = spec.name.split(".")[-1]
            try:
                circuit_noise_model[ref] = noise_maps[ref]
            except KeyError as ex:
                raise ValueError(f"Noise model is missing for layer with reference {ref}") from ex
            # noise_scales and pauli_lindblad_maps should have the same refs
            samplex_arguments[f"noise_scales.{ref}"] = noise_scale

        samplex_arguments["pauli_lindblad_maps"] = circuit_noise_model
        scaled_gamma = self.calculate_gamma(boxed_circuit, circuit_noise_model, noise_factor)
        self.gamma = scaled_gamma
        # Scale the baseline randomization count by gamma**2 for this pub independently.
        if scale_randomizations_by_gamma:
            sampling_overhead = scaled_gamma**2
            num_randomizations = int(
                np.ceil(
                    min(
                        num_randomizations * max_sampling_overhead,
                        num_randomizations * sampling_overhead,
                    )
                )
            )
            shape = (num_randomizations, *shape[1:])

        # Create SamplexItem
        if quantum_program is not None:
            self._program_item_index = len(quantum_program.items)
            quantum_program.append_samplex_item(
                circuit=template,
                samplex=samplex,
                samplex_arguments=samplex_arguments,
                shape=shape,
            )
        else:
            # Create QuantumProgram
            self._program_item_index = 0
            quantum_program = QuantumProgram(
                shots=shots_per_randomization,
                items=[
                    SamplexItem(
                        circuit=template,
                        samplex=samplex,
                        samplex_arguments=samplex_arguments,
                        shape=shape,
                    )
                ],
            )

        # Store data in passthrough_data
        meas_bases_str = (
            [pauli.to_label() for pauli in self.meas_bases] if self.meas_bases is not None else []
        )
        data_for_passthrough = {
            "version": self.VERSION,
            "mitigation": "pec",
            "circuit_metadata": circuit.metadata,
            "observables": self.observables.tolist(),
            "param_basis_pairs": self.param_basis_pairs,
            "param_shape": self.param_shape,
            "broadcast_obs_and_params": self.broadcast_obs_and_params,
            "meas_bases": meas_bases_str,
            "program_item_index": self._program_item_index,
            "trex_calibration": self.trex.has_calibration_result()
            if self.trex is not None
            else None,
            "pec_gamma": self.gamma,
        }

        self._add_data_to_passthrough_data(data_for_passthrough, quantum_program)
        return quantum_program

    def postprocess(
        self,
        results: QuantumProgramItemResult | QuantumProgramResult,
        measure_noise_data: PauliLindbladMap | np.ndarray | None = None,
    ) -> PubResult:
        """Process expectation values for a single pec mitigated item result.

        Args:
            results: The execution results. Can be either the entire results object or the pec mitigated item result of this task.
                If TREX calibration task is added to the quantum program, its results will be used to compute the measure noise data if the entire results object is provided.
            measure_noise_data: The learned measurement noise data to use for TREX mitigation.
                If `None` and a calibration task is present the quantum program, the measure noise will be computed from the calibration task results if the entire results object is provided.

        Returns:
            A PubResult which contains ``evs`` and ``std`` as fields in its data, where ``evs`` are expectation values,
            and ``std`` are the standard deviation of the expectation values.
            If ``broadcast_obs_and_params`` is set to True, the data will contain also a ``twirl_stds`` field which is
            the standard deviation between different randomizations.

        Raises:
            ValueError: If the task's item result has no ``'_meas'`` key.
            ValueError: If the item result's ``'_meas'`` data has an invalid number of axes.
            ValueError: If ``param_shape`` and ``observables.shape`` cannot be broadcasted against
                each other, while ``broadcast_obs_and_params`` was set to ``True`` in the task
                preparation.
        """
        item_result, measure_noise_data = self._extract_item_data(results, measure_noise_data)

        return self.compute_expectation_value_pec(
            item_result,
            observables=self.observables,
            param_shape=self.param_shape,
            gamma=self.gamma,
            param_basis_pairs=self.param_basis_pairs,
            meas_bases=self.meas_bases,
            broadcast_obs_and_params=self.broadcast_obs_and_params,
            measure_noise_data=measure_noise_data,
        )

    @staticmethod
    def calculate_gamma(
        boxed_circuit: QuantumCircuit,
        noise_maps: dict[str, PauliLindbladMap],
        noise_factor: float,
    ) -> float:
        """Calculate the PEC gamma factor of a circuit based on a noise model.

        The returned gamma is that associated with the inverse noise maps needed
        to cancel the noise in the circuit.

        Args:
            boxed_circuit: The annotated circuit to calculate the PEC gamma for.
            noise_maps: Mapping between layer ref to a noise model
            noise_factor: The noise factor of the noise amplification.

        Returns:
            The PEC gamma factor.
        """
        gamma = 1.0
        for instr in boxed_circuit:
            if annot := get_annotation(instr.operation, InjectNoise):
                ref = annot.ref
                try:
                    noise_model = noise_maps[ref]
                except KeyError as ex:
                    raise ValueError(
                        f"Noise model is missing for layer with reference {ref}"
                    ) from ex
                # scale the noise by noise_factor
                noise_model = noise_model.scale_rates(noise_factor)
                gamma *= noise_model.inverse().gamma()
        return gamma

    @staticmethod
    def _process_broadcasted_expectation_values_pec(
        data: np.ndarray,
        observables: ObservablesArray,
        param_shape: tuple[int, ...],
        param_basis_pairs: list[tuple[tuple[int, ...], str]],
        pec_gamma: float,
        pec_signs: np.ndarray,
        trex_scale_factors: dict[str, float] | None = None,
    ) -> tuple[
        np.typing.NDArray[np.float64], np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]
    ]:
        """Process expectation values for a single pec mitigated item result.

        Calculate the expectation values for an item in which the observables and parameters
        were broadcasted.

        Args:
            data: The pec mitigated result data to process.
            observables: The observables to calculate expectation values for.
            param_shape: The shape of the parameter values in the original PUB.
            param_basis_pairs: The map between params ndindexes to basis.
            pec_gamma: Gamma factor for PEC mitigation.
            pec_signs: A boolean array indicating for each randomization, for each noise term whether
                it was injected in this randomizationsigns for PEC mitigation.
            trex_scale_factors: A dictionary mapping each observable term to its scale factor for TREX mitigation.

        Returns:
            A tuple ``(exp_vals, stds, ensemble_stds)``, where ``exp_vals`` are expectation values,
            ``stds`` are standard deviations, and ``ensemble_stds`` are ensemble standard errors.
            ``stds`` is the variance between different randomizations, while ``ensemble_stds`` is
            the variance of the expectation value with the shots from all the randomizations
            combined.

        Raises:
            ValueError: If ``param_shape`` and ``observables.shape`` cannot be broadcasted against
                each other.
        """
        # Get number of randomizations and shots per randomization
        num_randomizations = data.shape[0]
        shots_per_randomization = data.shape[-2]
        total_shots = num_randomizations * shots_per_randomization

        # Build efficient lookup: param_ndindex -> list of (measurement_basis, config_idx)
        # This allows us to find all available measurement bases for a given parameter
        config_lookup: dict[tuple, list] = {}
        for config_idx, (param_ndindex, basis_label) in enumerate(param_basis_pairs):
            config_lookup.setdefault(tuple(param_ndindex), []).append(
                (Pauli(basis_label), config_idx)
            )

        try:
            output_shape = np.broadcast_shapes(param_shape, observables.shape)
        except ValueError as ex:
            raise ValueError(
                f"Cannot broadcast ``param_shape`` {param_shape} and ``observables`` shape "
                f"{observables.shape}"
            ) from ex

        # Compute expectation values for all observables
        exp_vals = np.empty(output_shape, dtype=float)
        stds = np.empty(output_shape, dtype=float)
        ensemble_stds = np.empty(output_shape, dtype=float)

        # Loop over the broadcast output shape
        for bcast_index in np.ndindex(output_shape):
            # Unbroadcast to get the actual parameter and observable indices
            param_index = PEC._unbroadcast_index(bcast_index, param_shape)
            obs_index = PEC._unbroadcast_index(bcast_index, observables.shape)

            # Get the observable for this index
            observable = observables[obs_index]

            # Get the available (measurement_basis, config_idx) pairs for this parameter index
            try:
                param_basis_list = config_lookup[param_index]  # type: ignore[index]
            except KeyError as ex:
                raise ValueError(
                    f"No measurement basis configurations found for parameter index {param_index}"
                ) from ex

            exp_val = 0.0
            ensemble_variance = 0.0
            twirl_variance = 0.0
            for observable_term, coeff in observable.items():
                # Find which basis can measure this term
                pauli_basis = Pauli(_convert_pauli_basis(observable_term))

                # Use _identify_measure_basis to find the configuration index directly
                config_idx = _identify_measure_basis(pauli_basis, param_basis_list)

                # get the signs for this configuration
                pec_signs_datum = pec_signs[:, config_idx, :]

                # Get measurement data for this configuration
                # Shape: (num_randomizations, shots, num_qubits)
                datum = data[:, config_idx, :, :]
                term_exp_val, term_ensemble_variance, term_twirl_variance = (
                    _compute_single_term_exp_val_with_twirl_variance(
                        observable_term, datum, pec_signs_datum
                    )
                )

                # Calculate scale factor in case TREX mitigation is used
                term_scale_factor = (
                    trex_scale_factors[observable_term] if trex_scale_factors is not None else 1
                )

                # Accumulate with coefficient
                exp_val += coeff * term_exp_val * term_scale_factor
                ensemble_variance += (coeff**2) * term_ensemble_variance * term_scale_factor**2
                twirl_variance += (coeff**2) * term_twirl_variance * term_scale_factor**2

            exp_vals[bcast_index] = exp_val * pec_gamma
            ensemble_stds[bcast_index] = np.sqrt(ensemble_variance * pec_gamma**2 / total_shots)
            if num_randomizations == 1:
                stds[bcast_index] = ensemble_stds[bcast_index]
            else:
                stds[bcast_index] = np.sqrt(twirl_variance * pec_gamma**2 / num_randomizations)

        return exp_vals, stds, ensemble_stds

    @staticmethod
    def create_instance_from_passthrough_data(
        passthrough: dict[str, Any], trex: TREX | None = None
    ) -> PEC:
        """Create a PEC instance from a passthrough dictionary loaded from a quantum program execution result.

        Args:
            passthrough: Passthrough_data dictionary loaded from a quantum program execution result.
            trex: A TREX instance containing a calibration circuit results executed in the same quantum program.
                Should remain ``None`` in case TREX mitigation was not used or a TREX calibration was not executed as
                part of thq same quantum program.

        Returns:
            A PEC instance.
        """
        if (observables := passthrough.get("observables")) is None:
            raise ValueError("Missing 'observables' in passthrough data.")
        param_basis_pairs = passthrough.get("param_basis_pairs")
        param_shape = passthrough.get("param_shape")
        if (broadcast_obs_and_params := passthrough.get("broadcast_obs_and_params")) is None:
            raise ValueError("Missing 'broadcast_obs_and_params' in passthrough data.")
        meas_bases = passthrough.get("meas_bases")
        if (program_item_index := passthrough.get("program_item_index")) is None:
            raise ValueError("Missing 'program_item_index' in passthrough data.")
        if (passthrough.get("trex_calibration", False)) and trex is None:
            raise ValueError(
                "TREX calibration was added to the quantum program at preparation but not supplied."
            )
        if (gamma := passthrough.get("pec_gamma")) is None:
            raise ValueError("Missing 'pec_gamma' in passthrough data.")

        pec = PEC()
        pec.observables = ObservablesArray.coerce(observables)
        pec.param_basis_pairs = param_basis_pairs
        pec.param_shape = param_shape
        pec.broadcast_obs_and_params = broadcast_obs_and_params
        pec.meas_bases = meas_bases
        pec._program_item_index = program_item_index
        pec.gamma = gamma
        pec.trex = trex

        return pec

    @staticmethod
    def compute_expectation_value_pec(
        item_result: QuantumProgramItemResult,
        observables: ObservablesArray | Sequence[SparsePauliOp],
        gamma: float,
        param_shape: tuple[int, ...] | None = None,
        param_basis_pairs: list[tuple[tuple[int, ...], str]] | None = None,
        meas_bases: Sequence[Pauli] | Sequence[str] | PauliList | None = None,
        broadcast_obs_and_params: bool = True,
        measure_noise_data: PauliLindbladMap | np.ndarray | None = None,
    ) -> PubResult:
        """Process expectation values for a single pec mitigated item result.

        This function can be used to calculate expectation values for a single pec mitigated item result without instantiating a new class instance.

        Args:
            item_result: The item result.
            observables: The observables to calculate expectation values for.
            gamma: The gamma factor of the learned noise model for the executed circuit.
            param_shape: The shape of the parameter values.
            param_basis_pairs: The map between params ndindexes to measure basis.
            meas_bases: A list of the measured Pauli bases. The ``i`` th item is a measurement basis assumed to
                correspond to the ``i`` th slice of the data in ``item_result``.
            broadcast_obs_and_params: Whether to broadcast observables and parameter values.
            measure_noise_data: Measurement noise calibration data for TREX mitigation.

        Returns:
            A PubResult which contains ``evs`` and ``std`` as fields in its data, where ``evs`` are expectation values,
            and ``std`` are the standard deviation of the expectation values.
            If ``broadcast_obs_and_params`` is set to True, the data will contain also a ``twirl_stds`` field which is
            the standard deviation between different randomizations.

        Raises:
            ValueError: If ``item_result`` has no ``'_meas'`` key.
            ValueError: If ``item_result['_meas']`` has invalid number of axes.
            ValueError: If ``item_result`` has no ``'pauli_signs'`` key.
            ValueError: If ``param_shape`` and ``observables.shape`` cannot be broadcasted against
                each other.
        """
        try:
            data = item_result["_meas"].copy()
        except KeyError as ex:
            raise ValueError("Dedicated creg ``'_meas'`` is missing from the results.") from ex

        # extract pec_old signs if present
        pec_signs = item_result.get("pauli_signs", None)
        if pec_signs is None:
            raise ValueError("Results must contain ``'pauli_signs'`` in the data if PEC is used.")

        trex_scale_factors = (
            TREX.trex_factors_each_term(measure_noise_data, observables)
            if measure_noise_data is not None
            else None
        )

        if broadcast_obs_and_params:
            if param_basis_pairs is None or param_shape is None:
                raise ValueError(
                    "``param_basis_pairs`` or ``param_shape`` is None while broadcasting of observables and parameters is True."
                )
            if data.ndim != 4:
                # Shape: (num_randomizations, num_configs, shots, num_bits)
                # where num_configs is the total number of (param_index, basis) pairs
                raise ValueError(
                    f"``item_result['_meas']`` has ``{data.ndim}`` axes, expected ``4``."
                )

            # Apply measurement flips if present
            if "measurement_flips._meas" in item_result:
                data ^= item_result["measurement_flips._meas"]

            if isinstance(observables, SparsePauliOp):
                observables = ObservablesArray.coerce(observables)

            exp_vals, twirl_stds, ensemble_stds = PEC._process_broadcasted_expectation_values_pec(
                data,
                observables,
                param_shape,
                param_basis_pairs,
                gamma,
                pec_signs,
                trex_scale_factors,
            )
            data_bin = DataBin(
                evs=exp_vals, stds=ensemble_stds, twirl_stds=twirl_stds, shape=exp_vals.shape
            )
            return PubResult(data=data_bin)

        if meas_bases is None:
            raise ValueError(
                "meas_bases is None while broadcasting of observables and parameters is False."
            )
        if data.ndim < 4 or data.ndim > 5:
            # Shape: (num_randomizations, num_bases, num_parameters, shots, num_bits)
            # if there are no parameters the shape is (num_randomizations, num_bases, shots, num_bits)
            raise ValueError(
                f"``item_result['_meas']`` has ``{data.ndim}`` axes, expected ``4`` or ``5``."
            )

        meas_flips = (
            item_result["measurement_flips._meas"]
            if "measurement_flips._meas" in item_result
            else None
        )
        if isinstance(observables, ObservablesArray):
            observables = [
                SparsePauliOp.from_sparse_observable(sparse_obs)
                for sparse_obs in observables.ravel().sparse_observables_array()
            ]

        exp_val_and_std = executor_expectation_values(
            data,
            (observables, meas_bases),
            meas_basis_axis=1,
            avg_axis=0,
            measurement_flips=meas_flips,
            pauli_signs=pec_signs,
            gamma_factor=gamma,
            rescale_factors=trex_scale_factors,
        )

        exp_vals, ensemble_stds = (np.array(x) for x in zip(*exp_val_and_std, strict=True))
        data_bin = DataBin(evs=exp_vals, stds=ensemble_stds, shape=exp_vals.shape)
        return PubResult(data=data_bin)
