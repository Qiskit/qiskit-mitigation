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

"""Executor-based expectation value calculation using Zero Noise Extrapolation (ZNE) mitigation method."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import DataBin, PubResult
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import Pauli, PauliLindbladMap, PauliList, SparsePauliOp
from qiskit.transpiler import PassManager
from samplomatic import build
from samplomatic.quantum_program import (
    QuantumProgram,
    QuantumProgramItemResult,
    QuantumProgramResult,
    SamplexItem,
)

from qiskit_mitigation.extrapolation.extrapolation_utils import (
    ExtrapolatorType,
    _process_extrapolated_expectation_values,
    _validate_noise_factors,
)
from qiskit_mitigation.extrapolation.gate_folding import GateFolding
from qiskit_mitigation.mitigation_task import MitigationTask
from qiskit_mitigation.trex import TREX
from qiskit_mitigation.utils.expectation_values import (
    _compute_single_term_exp_val_with_twirl_variance,
)
from qiskit_mitigation.utils.measurement_bases import (
    _convert_pauli_basis,
    _identify_measure_basis,
)


class ZNE(MitigationTask):
    """Calculates expectation values of observables using Zero Noise Extrapolation (ZNE) mitigation method.

    The class enables preparing a QuantumProgram with circuit mitigated using ZNE method,
    that can be executed on hardware using the Executor, and post process the results to calculate mitigated
    expectation values of given observables.
    The task parameters should be given as input to the prepare function, and the relevant
    variables needed for post-processing are saved internally:

    .. code-block:: python

        zne_task = ZNE()
        program = zne_task.prepare(circuit=circuit,
                               observables=observables,
                               parameters=parameter_values,
                               noise_factors=noise_factors_array,
                               extrapolator=extrapolators_list)
        job = executor.run(program)
        results = job.result()
        mitigated_result = zne_task.postprocess(results)

    To calculate expectation values of a loaded job result, a ZNE task can be created from the job results with all the internal variables needed for post-processing.
    Alternatively, the variables required for post-processing can be given as input directly to the ``compute_expectation_value`` static method.
    Example for running post-processing for a loaded result:

    .. code-block:: python

        zne_task = ZNE()
        program = zne_task.prepare(circuit=circuit,
                               observables=observables,
                               parameters=parameter_values,
                               noise_factors=noise_factors_array,
                               extrapolator=extrapolators_list)
        job_id = executor.run(program).job_id

        job = service.job(job_id)
        results = job.result()
        zne_task = load_tasks_from_result(results)[0]
        mitigated_result = zne_task.postprocess(results)

    """

    def __init__(self):
        """Instantiate a ZNE task."""
        super().__init__()
        self.noise_factors = None
        self.extrapolator = None
        self.extrapolated_noise_factors = None

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
        The function forces the boxing options to contain options that are needed for a ZNE mitigated circuit.

        Args:
            circuit: The circuit to group into boxes.
            boxing_options: Dictionary of boxing options. If ``None``, default boxing options are used.

        Returns:
            The boxed circuit.

        Raises:
            ValueError: If ``boxing_options["enable_measures"]`` is False.
            ValueError: If the boxing pass manager fails to run.
        """
        # The default boxing options for ZNE are different from samplomatic defaults
        if boxing_options is None:
            boxing_options = {"enable_gates": False}

        return super()._box_circuit(circuit, boxing_options)

    def _extract_items_list_data(
        self,
        results: list[QuantumProgramItemResult] | QuantumProgramResult,
        measure_noise_data: PauliLindbladMap | np.ndarray | None = None,
    ) -> tuple[QuantumProgramItemResult, PauliLindbladMap | np.ndarray | None]:
        """Extract data from ``results`` and ``measure_noise_data``."""
        if isinstance(results, QuantumProgramResult):
            item_results = results[
                self._program_item_index : self._program_item_index + len(self.noise_factors)
            ]
            if self.trex is not None and measure_noise_data is None:
                if self.trex.has_calibration_result():
                    measure_noise_data = self.trex.compute_noise_model(results)
                else:
                    warnings.warn(
                        "TREX is enabled in the task but measure noise data was not provided and a TREX calibration circuit was not found in the quantum program result. TREX mitigation will not be applied.",
                        stacklevel=2,
                    )
        else:
            item_results = results
            if self.trex is not None and measure_noise_data is None:
                warnings.warn(
                    "TREX is enabled in the task but measure noise data was not provided while only the task result is given as input. TREX mitigation will not be applied.",
                    stacklevel=2,
                )
        return item_results, measure_noise_data

    def prepare(
        self,
        circuit: QuantumCircuit,
        observables: ObservablesArray | Sequence[SparsePauliOp],
        parameters: np.ndarray | None,
        custom_boxing_options: dict | None = None,
        shots_per_randomization: int = 64,
        num_randomizations: int = 128,
        broadcast_obs_and_params: bool = False,
        trex: TREX | None = None,
        quantum_program: QuantumProgram | None = None,
        *,
        folding_method: Literal["random", "front", "back"] = "random",
        noise_factors: Sequence[float | int] = (1, 3, 5),
        extrapolator: Sequence[ExtrapolatorType] | None = None,
        extrapolated_noise_factors: list[float] | None = None,
    ) -> QuantumProgram:
        """Creates a :class:`~.QuantumProgram` with ZNE mitigated item for executing via Executor.

        Creates an item for a :class:`~.QuantumProgram`, that can be executed via Executor and is ZNE mitigated
        (amplifies the noise using gate folding techniques and extrapolates to the zero noise point).
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
            quantum_program: The quantum program to add an item for. If None, a new program will
                be created.
            folding_method: The technique to use to fold the 2-qubit gates to amplify the noise.
                If the noise factor requires amplifying only a subset of the gates, then these
                gates are selected according to the method:

                - ``"random"`` (default): the gates are chosen randomly.
                - ``"front"``: the gates are selected from the front of the topologically ordered
                  DAG circuit.
                - ``"back"``: the gates are selected from the back of the topologically ordered
                  DAG circuit.
            noise_factors: The noise factors to use for amplifying the noise.
            extrapolator: The extrapolator model or models planned to be used in post-processing.
                Used in preparation to validate that enough noise factors are present for a valid fit using these extrapolators.
                If ``None``, not validation will be done.
                Supported models (each fits the named function of the noise factor ``x``):

                - ``"linear"``: ``a + b*x``
                - ``"polynomial_degree_k"`` (1 <= k <= 7): a degree-k polynomial
                - ``"exponential"``: ``a*exp(b*x)``
                - ``"double_exponential"``: ``a*exp(b*x) + c*exp(d*x)`` (rates constrained to decay)
                - ``"fallback"``: no fit; the measured value at the lowest noise factor
            extrapolated_noise_factors: The noise factors to evaluate the fits at planned to be used in post-processing.
                Used for saving the data for easy post-processing. Can be overwritten in the post-processing.

        Returns:
            A :class:`~.QuantumProgram` with ZNE mitigated item that can be executed via Executor.
        """
        if folding_method not in ["random", "front", "back"]:
            raise ValueError(
                "Must choose a gate folding noise amplification method for ZNE mitigation."
            )
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

        self.noise_factors = np.array(noise_factors, dtype=float)
        if extrapolator is not None:
            _validate_noise_factors(noise_factors, extrapolator)
            self.extrapolator = extrapolator
        self.extrapolated_noise_factors = (
            np.array(extrapolated_noise_factors, dtype=float)
            if extrapolated_noise_factors is not None
            else None
        )

        # _program_item_index indicates the first item of this task in the program
        self._program_item_index = 0 if quantum_program is None else len(quantum_program.items)

        samplex_items = []
        for noise_factor in self.noise_factors:
            folding_pm = PassManager([GateFolding(noise_factor, folding_method)])
            folded_circuit = folding_pm.run(circuit)

            boxed_circuit = self._box_circuit(folded_circuit, self.custom_boxing_options)
            # Build the template and the samplex
            template, samplex = build(boxed_circuit)

            # Prepare samplex_arguments
            samplex_arguments, shape = self._create_samplex_arguments(samplex, boxed_circuit)

            # Create SamplexItem
            if quantum_program is not None:
                quantum_program.append_samplex_item(
                    circuit=template,
                    samplex=samplex,
                    samplex_arguments=samplex_arguments,
                    shape=shape,
                )
            else:
                samplex_items.append(
                    SamplexItem(
                        circuit=template,
                        samplex=samplex,
                        samplex_arguments=samplex_arguments,
                        shape=shape,
                    )
                )

        if quantum_program is None:
            # Create QuantumProgram
            quantum_program = QuantumProgram(
                shots=shots_per_randomization,
                items=samplex_items,
            )

        # Store data in passthrough_data
        meas_bases_str = (
            [pauli.to_label() for pauli in self.meas_bases] if self.meas_bases is not None else []
        )
        data_for_passthrough = {
            "version": self.VERSION,
            "mitigation": "zne",
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
            "noise_factors": self.noise_factors,
            "extrapolator": self.extrapolator,
            "extrapolated_noise_factors": self.extrapolated_noise_factors,
        }

        self._add_data_to_passthrough_data(data_for_passthrough, quantum_program)
        return quantum_program

    def postprocess(  # type: ignore[override]
        self,
        results: list[QuantumProgramItemResult] | QuantumProgramResult,
        measure_noise_data: PauliLindbladMap | np.ndarray | None = None,
        *,
        noise_factors: Sequence[float] | None = None,
        extrapolator: Sequence[ExtrapolatorType] | None = None,
        extrapolated_noise_factors: list[float] | None = None,
        custom_fit: tuple[
            Callable[..., np.ndarray], list[float], tuple[float, ...] | tuple[list[float], ...]
        ]
        | None = None,
    ) -> PubResult:
        """Process expectation values for a single zne task result.

        Args:
            results: The execution results. Can be either the entire results object or the zne mitigated items of this task.
                If TREX calibration task is added to the quantum program, its results will be used to compute the measure noise data if the entire results object is provided.
            noise_factors: The noise factors that were used to amplify the noise.
            extrapolated_noise_factors: Noise factors to evaluate the fits at.
            extrapolator: The extrapolator model or models to use. Models will be tried in priority
                order. Supported models (each fits the named function of the noise factor ``x``):

                - ``"linear"``: ``a + b*x``
                - ``"polynomial_degree_k"`` (1 <= k <= 7): a degree-k polynomial
                - ``"exponential"``: ``a*exp(b*x)``
                - ``"double_exponential"``: ``a*exp(b*x) + c*exp(d*x)`` (rates constrained to decay)
                - ``"fallback"``: no fit; the measured value at the lowest noise factor
                - ``custom``: The custom extrapolation function given in ``custom_fit`` parameter
            measure_noise_data: Measurement noise calibration data for TREX mitigation. Can be either a
                PauliLindbladMap of a noise model learned upfront, or a result of a calibration circuit.
            custom_fit: A custom fitting function that will be used for extrapolation. Should include a tuple in the form of:
                ``(function, p_0, bounds)``. The function should take the independent variable as the first argument and
                the parameters to fit as separate remaining arguments. p_0 is an array of the initial guess for the parameters,
                and bounds are the bounds on the parameters. These parameters will be sent as arguments for ``scipy.optimize.curve_fit`` function.

        Returns:
            A PubResult which contains the following fields:

            - ``evs``: expectation values evaluated at the zero noise point.
            - ``std``: the standard deviations of the extrapolated expectation values evaluated at the zero noise point.
            - ``evs_noise_factors``: expectation values calculated at the noise_factors points.
            - ``stds_noise_factors``: standard deviations calculated at the noise_factors points.
            - ``stds_twirl_noise_factors``: standard deviations between different randomizations calculated at the
                noise_factors points.
            - ``evs_extrapolated``: expectation values evaluated at the extrapolated_noise_factors points.
                Set to None if ``extrapolated_noise_factors`` is None.
            - ``stds_extrapolated``: standard deviations calculated at the extrapolated_noise_factors points.
                Set to None if ``extrapolated_noise_factors`` is None.

            The PubResult will also contain a ``selected_extrapolators`` field in the metadata, which is the valid
            extrapolators used to extrapolate the data for each observable term for the zero noise extrapolation point.

        Raises:
            ValueError: If one of the task's item results has no ``'_meas'`` key.
            ValueError: If one of the item results' ``'_meas'`` data has an invalid number of axes.
            ValueError: If ``param_shape`` and ``observables.shape`` cannot be broadcasted against each other.
        """
        if noise_factors is None:
            if self.noise_factors is None:
                raise ValueError(
                    "Must specify noise factors if the saved ``noise_factors`` is not set."
                )
            noise_factors = self.noise_factors
        if extrapolator is None:
            if self.extrapolator is None:
                raise ValueError(
                    "Must specify extrapolator if the saved ``extrapolator`` is not set."
                )
            extrapolator = self.extrapolator
        if extrapolated_noise_factors is None:
            extrapolated_noise_factors = self.extrapolated_noise_factors

        item_results, measure_noise_data = self._extract_items_list_data(
            results, measure_noise_data
        )

        return self.compute_expectation_value_zne(
            item_results,
            observables=self.observables,
            param_shape=self.param_shape,
            param_basis_pairs=self.param_basis_pairs,
            noise_factors=list(noise_factors),
            extrapolator=list(extrapolator),
            extrapolated_noise_factors=extrapolated_noise_factors,
            meas_bases=self.meas_bases,
            measure_noise_data=measure_noise_data,
            broadcast_obs_and_params=self.broadcast_obs_and_params,
            custom_fit=custom_fit,
        )

    @staticmethod
    def _calculate_extrapolated_expectation_values(
        noise_amplified_data: np.ndarray,
        observables: ObservablesArray,
        param_shape: tuple[int, ...],
        param_basis_pairs: list[tuple[tuple[int, ...], str]],
        noise_factors: list[float],
        extrapolated_noise_factors: list[float],
        extrapolator: list[ExtrapolatorType],
        trex_scale_factors: dict[str, float] | None = None,
        custom_fit: tuple[
            Callable[..., np.ndarray], list[float], tuple[float, ...] | tuple[list[float], ...]
        ]
        | None = None,
    ) -> tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        list[list[str]],
    ]:
        """Calculate expectation values for given data, observables and params.

        Args:
            noise_amplified_data: The measured data result. Its shape should be:
                ``(noise_factors, num_randomizations, num_configs, shots, num_bits)``
            observables: The observables to calculate expectation values for.
            param_shape: The shape of the parameter values in the original PUB.
            param_basis_pairs: The map between params ndindexes to basis.
            noise_factors: The noise factors used to amplify the noise.
            extrapolated_noise_factors: Noise factors to evaluate the fits at.
            extrapolator: The extrapolator model or models to use. Models will be tried in priority
                order. Supported models (each fits the named function of the noise factor ``x``):

                - ``"linear"``: ``a + b*x``
                - ``"polynomial_degree_k"`` (1 <= k <= 7): a degree-k polynomial
                - ``"exponential"``: ``a*exp(b*x)``
                - ``"double_exponential"``: ``a*exp(b*x) + c*exp(d*x)`` (rates constrained to decay)
                - ``"fallback"``: no fit; the measured value at the lowest noise factor
                - ``custom``: The custom extrapolation function given in ``custom_fit`` parameter
            trex_scale_factors: A dictionary mapping each observable term to its scale factor for TREX mitigation.
            custom_fit: A custom fitting function that will be used for extrapolation. Should include a tuple in the form of:
                ``(function, p_0, bounds)``. The function should take the independent variable as the first argument and
                the parameters to fit as separate remaining arguments. p_0 is an array of the initial guess for the parameters,
                and bounds are the bounds on the parameters. These parameters will be sent as arguments for ``scipy.optimize.curve_fit`` function.

        Returns:
            A tuple with the following entries:

            - ``zero_extrapolated_exp_vals``: expectation values evaluated at the zero noise point.
            - ``zero_extrapolated_stds``: the standard deviations of the extrapolated expectation
              values evaluated at the zero noise point.
            - ``noise_factors_exp_vals``: expectation values calculated at the noise_factors points.
            - ``noise_factors_ensemble_stds``: ensemble standard errors calculated at the noise_factors points.
            - ``noise_factors_stds``: standard errors calculated at the noise_factors points.
            - ``extrapolated_exp_vals``: expectation values evaluated at the extrapolated_noise_factors points.
            - ``extrapolated_stds``: standard errors evaluated at the extrapolated_noise_factors points.
            - ``selected_extrapolators``: the valid extrapolators used to extrapolate the data for each observable term.

        Raises:
            ValueError: If ``param_shape`` and ``observables.shape`` cannot be broadcasted against each other.
        """
        # Get number of randomizations and shots per randomization
        # Shape: (noise_factors, num_randomizations, num_configs, shots_per_rand, num_bits)
        num_randomizations = noise_amplified_data.shape[1]
        shots_per_randomization = noise_amplified_data.shape[-2]
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
        zero_extrapolated_exp_vals = np.zeros(shape=output_shape, dtype=float)
        zero_extrapolated_vars = np.zeros(shape=output_shape, dtype=float)
        # Save the data for the extrapolated points (only exp_vals and ensamble_stds)
        extrapolated_shape = (
            *output_shape,
            len(extrapolator),
            len(extrapolated_noise_factors),
        )
        extrapolated_exp_vals = np.empty(shape=extrapolated_shape, dtype=float)
        extrapolated_stds = np.empty(shape=extrapolated_shape, dtype=float)

        # Save also the data for all of the noise amplified points
        noise_factors_output_shape = (*output_shape, len(noise_factors))
        noise_factors_exp_vals = np.zeros(shape=noise_factors_output_shape, dtype=float)
        noise_factors_ensemble_variance = np.zeros(shape=noise_factors_output_shape, dtype=float)
        noise_factors_twirl_variance = np.zeros(shape=noise_factors_output_shape, dtype=float)
        noise_factors_ensemble_stds = np.empty(shape=noise_factors_output_shape, dtype=float)
        noise_factors_twirl_stds = np.empty(shape=noise_factors_output_shape, dtype=float)
        # save for each extrapolated observable term (in each observable, for each parameter
        # configuration), the selected extrapolator
        selected_extrapolators: list[list[str]] = []

        # Loop over the broadcast output shape
        for bcast_index in np.ndindex(output_shape):
            # Unbroadcast to get the actual parameter and observable indices
            param_index = ZNE._unbroadcast_index(bcast_index, param_shape)
            obs_index = ZNE._unbroadcast_index(bcast_index, observables.shape)

            # Get the observable for this index
            observable = observables[obs_index]

            # Get the available (measurement_basis, config_idx) pairs for this parameter index
            try:
                param_basis_list = config_lookup[param_index]  # type: ignore[index]
            except KeyError as ex:
                raise ValueError(
                    f"No measurement basis configurations found for parameter index {param_index}"
                ) from ex

            # each item should contain results for each point in extrapolated_noise_factors
            exp_vals_extrapolated = np.zeros(
                shape=(len(extrapolator), len(extrapolated_noise_factors)), dtype=float
            )
            ensemble_var_extrapolated = np.zeros(
                shape=(len(extrapolator), len(extrapolated_noise_factors)), dtype=float
            )

            selected_extrapolators_per_term: list[str] = []

            for observable_term, coeff in observable.items():
                # Find which basis can measure this term
                pauli_basis = Pauli(_convert_pauli_basis(observable_term))

                # Use identify_measure_basis to find the configuration index directly
                config_idx = _identify_measure_basis(pauli_basis, param_basis_list)

                # Calculate scale factor in case TREX mitigation is used
                term_scale_factor = (
                    trex_scale_factors[observable_term] if trex_scale_factors is not None else 1
                )

                noise_scaled_exp_vals = []
                noise_scaled_ensemble_std = []
                for noise_factor_index in range(len(noise_factors)):
                    noise_factor_data = noise_amplified_data[noise_factor_index]
                    # Get measurement data for this configuration
                    # datum shape: (num_randomizations, shots_per_randomization, num_qubits)
                    datum = noise_factor_data[:, config_idx, :, :]
                    term_exp_val, term_ensemble_variance, term_twirl_variance = (
                        _compute_single_term_exp_val_with_twirl_variance(observable_term, datum)
                    )
                    noise_scaled_exp_vals.append(term_exp_val)
                    noise_scaled_ensemble_std.append(np.sqrt(term_ensemble_variance))

                    noise_factors_exp_vals[(*bcast_index, noise_factor_index)] += (
                        coeff * term_exp_val * term_scale_factor
                    )
                    noise_factors_ensemble_variance[(*bcast_index, noise_factor_index)] += (
                        (coeff**2) * term_ensemble_variance * term_scale_factor**2
                    )
                    noise_factors_twirl_variance[(*bcast_index, noise_factor_index)] += (
                        (coeff**2) * term_twirl_variance * term_scale_factor**2
                    )

                (
                    zero_noise_exp_val,
                    zero_noise_std,
                    sel_extrapolator,
                    extrap_exp_vals,
                    extrap_stds,
                ) = _process_extrapolated_expectation_values(
                    noise_scaled_exp_vals,
                    noise_scaled_ensemble_std,
                    observable_term,
                    noise_factors,
                    extrapolator,
                    extrapolated_noise_factors,
                    custom_fit,
                )

                # Only the selected extrapolator of the zero point is returned
                selected_extrapolators_per_term.append(sel_extrapolator)
                zero_extrapolated_exp_vals[bcast_index] += (
                    coeff * zero_noise_exp_val * term_scale_factor
                )
                zero_extrapolated_vars[bcast_index] += (
                    (coeff**2) * (zero_noise_std**2) * (term_scale_factor**2)
                )

                for model_index, (extrap_model_exp_val, extrap_model_std) in enumerate(
                    zip(extrap_exp_vals, extrap_stds, strict=True)
                ):
                    for extrap_index, (extrap_exp_val, extrap_std) in enumerate(
                        zip(extrap_model_exp_val, extrap_model_std, strict=True)
                    ):
                        # Accumulate with coefficient
                        exp_vals_extrapolated[(model_index, extrap_index)] += (
                            coeff * extrap_exp_val * term_scale_factor
                        )
                        ensemble_var_extrapolated[(model_index, extrap_index)] += (
                            (coeff**2) * (extrap_std**2) * (term_scale_factor**2)
                        )

            idx: tuple[int | slice, ...] = (*bcast_index, slice(None), slice(None))
            extrapolated_exp_vals[idx] = exp_vals_extrapolated
            extrapolated_stds[idx] = np.sqrt(ensemble_var_extrapolated)
            for noise_factor_index in range(len(noise_factors)):
                noise_factors_ensemble_stds[(*bcast_index, noise_factor_index)] = np.sqrt(
                    noise_factors_ensemble_variance[(*bcast_index, noise_factor_index)]
                    / total_shots
                )
                noise_factors_twirl_stds[(*bcast_index, noise_factor_index)] = np.sqrt(
                    noise_factors_twirl_variance[(*bcast_index, noise_factor_index)]
                    / num_randomizations
                )
            selected_extrapolators.append(selected_extrapolators_per_term)

        return (
            zero_extrapolated_exp_vals,
            np.sqrt(zero_extrapolated_vars),
            noise_factors_exp_vals,
            noise_factors_ensemble_stds,
            noise_factors_twirl_stds,
            extrapolated_exp_vals,
            extrapolated_stds,
            selected_extrapolators,
        )

    @staticmethod
    def create_instance_from_passthrough_data(
        passthrough: dict[str, Any], trex: TREX | None = None
    ) -> ZNE:
        """Create a ZNE instance from a passthrough dictionary loaded from a quantum program execution result.

        Args:
            passthrough: Passthrough_data dictionary loaded from a quantum program execution result.
            trex: A TREX instance containing a calibration circuit results executed in the same quantum program.
                Should remain ``None`` in case TREX mitigation was not used or a TREX calibration was not executed as
                part of thq same quantum program.

        Returns:
            A ZNE instance.
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
        if (noise_factors := passthrough.get("noise_factors")) is None:
            raise ValueError("Missing 'noise_factors' in passthrough data.")
        extrapolator = passthrough.get("extrapolator")
        extrapolated_noise_factors = passthrough.get("extrapolated_noise_factors")

        zne = ZNE()
        zne.observables = ObservablesArray.coerce(observables)
        zne.param_basis_pairs = param_basis_pairs
        zne.param_shape = param_shape
        zne.broadcast_obs_and_params = broadcast_obs_and_params
        zne.meas_bases = meas_bases
        zne._program_item_index = program_item_index
        zne.noise_factors = noise_factors
        zne.extrapolator = extrapolator
        zne.extrapolated_noise_factors = extrapolated_noise_factors
        zne.trex = trex

        return zne

    @staticmethod
    def compute_expectation_value_zne(
        item_results: list[QuantumProgramItemResult],
        observables: ObservablesArray | Sequence[SparsePauliOp],
        param_shape: tuple[int, ...] | None,
        param_basis_pairs: list[tuple[tuple[int, ...], str]] | None,
        noise_factors: list[float],
        extrapolator: list[ExtrapolatorType],
        extrapolated_noise_factors: float | int | list[float] | None = None,
        meas_bases: Sequence[Pauli] | Sequence[str] | PauliList | None = None,
        broadcast_obs_and_params: bool = True,
        measure_noise_data: PauliLindbladMap | np.ndarray | None = None,
        custom_fit: tuple[
            Callable[..., np.ndarray], list[float], tuple[float, ...] | tuple[list[float], ...]
        ]
        | None = None,
    ) -> PubResult:
        """Process expectation values for a single item result.

        This function can be used to calculate expectation values for a single zne mitigated item result without instantiating a new class instance.

        Args:
            item_results: List of the item results - one for each noise factor.
            observables: The observables to calculate expectation values for.
            param_shape: The shape of the parameter values.
            param_basis_pairs: The map between params ndindexes to measure basis.
            noise_factors: The noise factors that were used to amplify the noise.
            extrapolated_noise_factors: Noise factors to evaluate the fits at.
            extrapolator: The extrapolator model or models to use. Models will be tried in priority
                order. Supported models (each fits the named function of the noise factor ``x``):

                - ``"linear"``: ``a + b*x``
                - ``"polynomial_degree_k"`` (1 <= k <= 7): a degree-k polynomial
                - ``"exponential"``: ``a*exp(b*x)``
                - ``"double_exponential"``: ``a*exp(b*x) + c*exp(d*x)`` (rates constrained to decay)
                - ``"fallback"``: no fit; the measured value at the lowest noise factor
                - ``custom``: The custom extrapolation function given in ``custom_fit`` parameter
            meas_bases: A list of the measured Pauli bases. The ``i`` th item is a measurement basis assumed to
                correspond to the ``i`` th slice of the data in each of the ``item_results``.
            broadcast_obs_and_params: Whether to broadcast observables and parameter values.
            measure_noise_data: Measurement noise calibration data for TREX mitigation. Can be either a
                PauliLindbladMap of a noise model learned upfront, or a result of a calibration circuit.
            custom_fit: A custom fitting function that will be used for extrapolation. Should include a tuple in the form of:
                ``(function, p_0, bounds)``. The function should take the independent variable as the first argument and
                the parameters to fit as separate remaining arguments. p_0 is an array of the initial guess for the parameters,
                and bounds are the bounds on the parameters. These parameters will be sent as arguments for ``scipy.optimize.curve_fit`` function.

        Returns:
            A PubResult which contains the following fields:

            - ``evs``: expectation values evaluated at the zero noise point.
            - ``std``: the standard deviations of the extrapolated expectation values evaluated at the zero noise point.
            - ``evs_noise_factors``: expectation values calculated at the noise_factors points.
            - ``stds_noise_factors``: standard deviations calculated at the noise_factors points.
            - ``stds_twirl_noise_factors``: standard deviations between different randomizations calculated at the
                noise_factors points.
            - ``evs_extrapolated``: expectation values evaluated at the extrapolated_noise_factors points.
                Set to None if ``extrapolated_noise_factors`` is None.
            - ``stds_extrapolated``: standard deviations calculated at the extrapolated_noise_factors points.
                Set to None if ``extrapolated_noise_factors`` is None.

            The PubResult will also contain a ``selected_extrapolators`` field in the metadata, which is the valid
            extrapolators used to extrapolate the data for each observable term for the zero noise extrapolation point.

        Raises:
            ValueError: If one of the ``item_results`` has no ``'_meas'`` key.
            ValueError: If one of the ``item_results``' ``'_meas'`` data has an invalid number of axes.
            ValueError: If ``param_shape`` and ``observables.shape`` cannot be broadcasted against each other.
            ValueError: If ``noise_factors`` is under-specified for any extrapolator.
        """
        _validate_noise_factors(noise_factors, extrapolator)

        if extrapolated_noise_factors is None:
            extrapolated_noise_factors = []
        elif isinstance(extrapolated_noise_factors, (float, int)):
            extrapolated_noise_factors = [extrapolated_noise_factors]

        observables_arr = cast(
            ObservablesArray,
            ObservablesArray.coerce(observables)
            if isinstance(observables, SparsePauliOp)
            or (
                isinstance(observables, Sequence)
                and all(isinstance(obs, SparsePauliOp) for obs in observables)
            )
            else observables,
        )

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
        else:
            if meas_bases is None:
                raise ValueError(
                    "meas_bases is None while broadcasting of observables and parameters is False."
                )
            try:
                data_shape = item_results[0]["_meas"].shape
            except KeyError as ex:
                raise ValueError(
                    "Dedicated creg ``'_meas'`` is missing from one of the results."
                ) from ex
            if param_shape is None:
                if len(data_shape) == 4:
                    param_shape = ()
                elif len(data_shape) == 5:
                    # Shape: (num_randomizations, num_bases, num_parameters, shots, num_bits)
                    param_shape = (data_shape[2],)
                else:
                    raise ValueError(
                        f"Could not calculate parameters shape from the data, has ``{len(data_shape)}`` axes, expected ``4`` or ``5``."
                    )

            if param_basis_pairs is None:
                broadcast_shape = observables_arr.shape + param_shape
                meas_bases = PauliList(meas_bases)
                param_basis_pairs = ZNE._compute_param_basis_pairs(
                    observables_arr, param_shape, broadcast_shape, meas_bases
                )

        # Combine the data from each noise factor
        noise_amplified_data = []
        for item_result in item_results:
            try:
                data = item_result["_meas"]
            except KeyError as ex:
                raise ValueError(
                    "Dedicated creg ``'_meas'`` is missing from one of the results."
                ) from ex

            if broadcast_obs_and_params and data.ndim != 4:
                # Shape: (num_randomizations, num_configs, shots, num_bits)
                # where num_configs is the total number of (param_index, basis) pairs
                raise ValueError(
                    f"one of the ``item_result['_meas']`` has ``{data.ndim}`` axes, expected ``4``."
                )
            if not broadcast_obs_and_params and (data.ndim < 4 or data.ndim > 5):
                # Shape: (num_randomizations, num_bases, num_parameters, shots, num_bits)
                # where num_configs is the total number of (param_index, basis) pairs
                raise ValueError(
                    f"``item_result['_meas']`` has ``{data.ndim}`` axes, expected ``4`` or ``5``."
                )

            # Apply measurement flips if present
            meas_flips = item_result.pop("measurement_flips._meas", None)
            if meas_flips is not None:
                data ^= meas_flips

            noise_amplified_data.append(data)

        (
            zero_extrapolated_exp_vals,
            zero_extrapolated_stds,
            noise_factors_exp_vals,
            noise_factors_stds,
            noise_factors_twirl_stds,
            extrapolated_exp_vals,
            extrapolated_stds,
            selected_extrapolators,
        ) = ZNE._calculate_extrapolated_expectation_values(
            np.array(noise_amplified_data),
            observables_arr,
            param_shape,
            param_basis_pairs,
            noise_factors,
            extrapolated_noise_factors,
            extrapolator,
            trex_scale_factors,
            custom_fit,
        )

        returned_extrapolated_exp_vals = (
            None if extrapolated_noise_factors == [] else extrapolated_exp_vals
        )
        returned_extrapolated_stds = None if extrapolated_noise_factors == [] else extrapolated_stds

        data_bin = DataBin(
            evs=zero_extrapolated_exp_vals,
            stds=zero_extrapolated_stds,
            evs_noise_factors=noise_factors_exp_vals,
            stds_noise_factors=noise_factors_stds,
            stds_twirl_noise_factors=noise_factors_twirl_stds,
            evs_extrapolated=returned_extrapolated_exp_vals,
            stds_extrapolated=returned_extrapolated_stds,
            shape=zero_extrapolated_exp_vals.shape,
        )
        return PubResult(data=data_bin, metadata={"selected_extrapolators": selected_extrapolators})
