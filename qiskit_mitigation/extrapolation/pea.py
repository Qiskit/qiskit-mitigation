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

"""Executor-based expectation value calculation using Probabilistic Error Amplification (PEA) mitigation method."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import DataBin, PubResult
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import Pauli, PauliLindbladMap, PauliList, SparsePauliOp
from samplomatic import build
from samplomatic.quantum_program import (
    QuantumProgram,
    QuantumProgramItemResult,
    QuantumProgramResult,
    SamplexItem,
)

from qiskit_mitigation.extrapolation.extrapolation_utils import (
    ExtrapolatorType,
    _validate_noise_factors,
)
from qiskit_mitigation.extrapolation.zne import ZNE
from qiskit_mitigation.mitigation_task import MitigationTask
from qiskit_mitigation.trex import TREX


class PEA(MitigationTask):
    """Calculates expectation values of observables using Probabilistic Error Amplification (PEA) mitigation method.

    The class enables preparing a QuantumProgram with circuit mitigated using PEA method,
    that can be executed on hardware using the Executor, and post process the results to calculate mitigated
    expectation values of given observables.
    The task parameters should be given as input to the prepare function, and the relevant
    variables needed for post-processing are saved internally:

    .. code-block:: python

        pea_task = PEA()
        program = pea_task.prepare(circuit=circuit,
                                   observables=observables,
                                   parameters=parameter_values,
                                   noise_factors=noise_factors_array,
                                   extrapolator=extrapolators_list)
        job = executor.run(program)
        results = job.result()
        mitigated_result = pea_task.postprocess(results)

    To calculate expectation values of a loaded job result, a PEA task can be created from the job results with all the internal variables needed for post-processing.
    Alternatively, the variables required for post-processing can be given as input directly to the ``compute_expectation_value`` static method.
    Example for running post-processing for a loaded result:

    .. code-block:: python

        pea_task = PEA()
        program = pea_task.prepare(circuit=circuit,
                                   observables=observables,
                                   parameters=parameter_values,
                                   noise_factors=noise_factors_array,
                                   extrapolator=extrapolators_list)
        job_id = executor.run(program).job_id

        job = service.job(job_id)
        results = job.result()
        pea_task = load_tasks_from_result(results)[0]
        mitigated_result = pea_task.postprocess(results)

    """

    def __init__(self):
        """Instantiate a PEA task."""
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
        The function forces the boxing options to contain options that are needed for a PEA mitigated circuit.

        Args:
            circuit: The circuit to group into boxes.
            boxing_options: Dictionary of boxing options. If ``None``, default boxing options are used.

        Returns:
            The boxed circuit.

        Raises:
            ValueError: If ``boxing_options`` contradict required boxing PEA parameters -
                ``boxing_options["enable_measures"]`` is False,
                ``boxing_options["enable_gates"]`` is False,
                ``boxing_options["inject_noise_targets"]`` not in ["gates", "all"],
                ``boxing_options["inject_noise_strategy"]`` is "no_modification".
            ValueError: If the boxing pass manager fails to run.
        """
        if boxing_options is None:
            boxing_options = {}
        # Force PEA related options
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
        parameters: np.ndarray | None,
        custom_boxing_options: dict | None = None,
        shots_per_randomization: int = 64,
        num_randomizations: int = 128,
        broadcast_obs_and_params: bool = False,
        trex: TREX | None = None,
        quantum_program: QuantumProgram | None = None,
        *,
        noise_maps: dict[str, PauliLindbladMap] | None = None,
        noise_factors: Sequence[float | int] = (1, 3, 5),
        extrapolator: Sequence[ExtrapolatorType] | None = None,
        extrapolated_noise_factors: list[float] | None = None,
    ) -> QuantumProgram:
        """Creates a :class:`~.QuantumProgram` with PEA mitigated item for executing via Executor.

        Creates an item for a :class:`~.QuantumProgram`, that can be executed via Executor and is PEA mitigated
        (amplifies the noise by scaling the injected Pauli noise and extrapolates to the zero noise point).
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
            noise_maps: A mapping between layer ref to a noise model to use for noise amplification.
                The dict might contain layers not present in the given circuit, but must contain all
                the mitigated layers. Assumes that the unique layers used for noise learning were extracted
                using the ``find_unique_layers`` method with the same custom boxing options.
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
            A :class:`~.QuantumProgram` with PEA mitigated item that can be executed via Executor.
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

        self.noise_factors = np.array(noise_factors, dtype=float)
        if extrapolator is not None:
            _validate_noise_factors(noise_factors, extrapolator)
            self.extrapolator = extrapolator
        self.extrapolated_noise_factors = (
            np.array(extrapolated_noise_factors, dtype=float)
            if extrapolated_noise_factors is not None
            else None
        )

        boxed_circuit = self._box_circuit(self.circuit, self.custom_boxing_options)
        self.boxed_circuit = boxed_circuit
        # Build the template and the samplex
        template, samplex = build(boxed_circuit)

        # Prepare samplex_arguments
        samplex_arguments, shape = self._create_samplex_arguments(samplex, boxed_circuit)

        # Subtract 1 from noise_factors, since a value of 1 represents the noise
        # that is present in the circuit in the absence of amplification.
        # Also, make noise_scales broadcastable with the parameters and randomizations.
        noise_scales = np.expand_dims(np.array(noise_factors) - 1, (-1, -2))

        # Create a noise model map containing only the layers relevant for the given circuit
        specs = samplex.inputs().get_specs("pauli_lindblad_maps")
        pub_noise_model = {}
        for spec in specs:
            ref = spec.name.split(".")[-1]
            try:
                noise_model = noise_maps[ref]
            except KeyError as ex:
                raise ValueError(f"Noise model is missing for layer with reference {ref}") from ex
            pub_noise_model[ref] = noise_model
            samplex_arguments[f"noise_scales.{ref}"] = noise_scales

        samplex_arguments["pauli_lindblad_maps"] = pub_noise_model

        # Create SamplexItem with noise_factors as first axis
        shape = (len(noise_scales), *shape)
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
            "mitigation": "pea",
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
        results: QuantumProgramItemResult | QuantumProgramResult,
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
        """Process expectation values for a single item result.

        Args:
            results: The execution results. Can be either the entire results object or the pea mitigated item result of this task.
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
            ValueError: If the task's item result has no ``'_meas'`` key.
            ValueError: If the item result's ``'_meas'`` data has an invalid number of axes.
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

        item_result, measure_noise_data = self._extract_item_data(results, measure_noise_data)

        return self.compute_expectation_value_pea(
            item_result,
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
    def create_instance_from_passthrough_data(
        passthrough: dict[str, Any], trex: TREX | None = None
    ) -> PEA:
        """Create a PEA instance from a passthrough dictionary loaded from a quantum program execution result.

        Args:
            passthrough: Passthrough_data dictionary loaded from a quantum program execution result.
            trex: A TREX instance containing a calibration circuit results executed in the same quantum program.
                Should remain ``None`` in case TREX mitigation was not used or a TREX calibration was not executed as
                part of thq same quantum program.

        Returns:
            A PEA instance.
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

        pea = PEA()
        pea.observables = ObservablesArray.coerce(observables)
        pea.param_basis_pairs = param_basis_pairs
        pea.param_shape = param_shape
        pea.broadcast_obs_and_params = broadcast_obs_and_params
        pea.meas_bases = meas_bases
        pea._program_item_index = program_item_index
        pea.noise_factors = noise_factors
        pea.extrapolator = extrapolator
        pea.extrapolated_noise_factors = extrapolated_noise_factors
        pea.trex = trex

        return pea

    @staticmethod
    def compute_expectation_value_pea(
        item_result: QuantumProgramItemResult,
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

        This function can be used to calculate expectation values for a single pea mitigated item result without instantiating a new class instance.

        Args:
            item_result: The pea mitigated item result.
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
                correspond to the ``i`` th slice of the data in ``item_result``.
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
            ValueError: If ``item_result`` has no ``'_meas'`` key.
            ValueError: If ``item_result['_meas']`` has invalid number of axes.
            ValueError: If ``param_shape`` and ``observables.shape`` cannot be broadcasted against each other.
            ValueError: If ``noise_factors`` is under-specified for any extrapolator.
        """
        try:
            data = item_result["_meas"]
        except KeyError as ex:
            raise ValueError(
                "Dedicated creg ``'_meas'`` is missing from one of the results."
            ) from ex

        _validate_noise_factors(noise_factors, extrapolator)

        if data.shape[0] != len(noise_factors):
            raise ValueError(
                "Number of noise factors in the data does not match the length of ``noise_factors``."
            )

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
            data_shape = data.shape

            if param_shape is None:
                if len(data_shape) == 5:
                    param_shape = ()
                elif len(data_shape) == 6:
                    # Shape: (noise_factors, num_randomizations, num_bases, num_parameters, shots, num_bits)
                    param_shape = (data_shape[3],)
                else:
                    raise ValueError(
                        f"Could not calculate parameters shape from the data, has ``{len(data_shape)}`` axes, expected ``5`` or ``6``."
                    )

            if param_basis_pairs is None:
                broadcast_shape = observables_arr.shape + param_shape
                meas_bases = PauliList(meas_bases)
                param_basis_pairs = PEA._compute_param_basis_pairs(
                    observables_arr, param_shape, broadcast_shape, meas_bases
                )

        # Apply measurement flips if present
        meas_flips = item_result.pop("measurement_flips._meas", None)
        if meas_flips is not None:
            data ^= meas_flips

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
            data,
            observables_arr,
            param_shape,
            param_basis_pairs,
            noise_factors,
            extrapolated_noise_factors,
            extrapolator,
            trex_scale_factors,
            custom_fit=custom_fit,
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
