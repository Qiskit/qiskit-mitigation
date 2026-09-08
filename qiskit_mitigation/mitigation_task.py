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

"""Basic Executor-based expectation value calculation workflow."""

from __future__ import annotations

import copy
import warnings
from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, cast

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, ClassicalRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.primitives import DataBin, PubResult
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import Pauli, PauliLindbladMap, PauliList, SparsePauliOp
from samplomatic import ChangeBasis, build
from samplomatic.quantum_program import (
    QuantumProgram,
    QuantumProgramItemResult,
    QuantumProgramResult,
    SamplexItem,
)
from samplomatic.samplex import Samplex
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import find_unique_box_instructions, get_annotation

from qiskit_mitigation.trex import TREX
from qiskit_mitigation.utils.expectation_values import (
    _compute_single_term_exp_val_with_twirl_variance,
    executor_expectation_values,
)
from qiskit_mitigation.utils.measurement_bases import (
    _convert_pauli_basis,
    _identify_measure_basis,
    _pauli_to_ints,
    get_measurement_bases,
)


class MitigationTask:
    """Calculates expectation values of observables using the executor.

    This class is the base class for all mitigation methods. It enables preparing a QuantumProgram
    that can be executed on hardware using the Executor, and post process the results to calculate
    expectation values of given observables.

    The task parameters should be given as input to the prepare function, and the relevant
    variables needed for post-processing are saved internally:

    .. code-block:: python

        task = MitigationTask()
        program = task.prepare(circuit=circuit,
                               observables=observables,
                               parameters=parameter_values)
        job = executor.run(program)
        results = job.result()
        mitigated_result = task.postprocess(results)

    To calculate expectation values of a loaded job result, a task can be created from the job results with all the internal variables needed for post-processing.
    Alternatively, the variables required for post-processing can be given as input directly to the ``compute_expectation_value`` static method.
    Example for running post-processing for a loaded result:

    .. code-block:: python

        task = MitigationTask()
        program = task.prepare(circuit=circuit,
                               observables=observables,
                               parameters=parameter_values)
        job_id = executor.run(program).job_id

        job = service.job(job_id)
        results = job.result()
        task = load_tasks_from_result(results)[0]
        mitigated_result = task.postprocess(results)

    """

    VERSION = "0.1"

    def __init__(self):
        """Instantiate a MitigationTask."""
        self.circuit: QuantumCircuit = None
        self.boxed_circuit: QuantumCircuit = None
        self.observables: ObservablesArray = None
        self._sparse_observables: SparsePauliOp = None
        self.parameters: BindingsArray = None
        self.broadcast_obs_and_params: bool = None
        self.broadcast_shape: tuple[int, ...] = None
        self.param_basis_pairs: list[tuple[tuple[int, ...], str]] | None = None
        self.param_shape = None
        self.meas_bases: list[Pauli] | None = None
        self.custom_boxing_options: dict = None
        self.shots_per_randomization: int = None
        self.num_randomizations: int = None
        self.trex: TREX | None = None
        self._program_item_index = None

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
        circuit into boxes. The input custom boxing options are used as input for the boxing
        function.

        Args:
            circuit: The circuit to group into boxes.
            boxing_options: Dictionary of :meth:`~samplomatic.transpiler.generate_boxing_pass_manager` options.
                If ``None``, default boxing options are used.

        Returns:
            The boxed circuit.

        Raises:
            ValueError: If ``boxing_options["enable_measures"]`` is False.
            ValueError: If the boxing pass manager fails to run.
        """
        if boxing_options is None:
            boxing_options = {}
        if "enable_measures" not in boxing_options:
            boxing_options["enable_measures"] = True
            boxing_options["measure_annotations"] = "change_basis"
        elif not boxing_options["enable_measures"]:
            raise ValueError('boxing_options["enable_measures"] may not be False.')
        if (
            "measure_annotations" in boxing_options
            and boxing_options["measure_annotations"] == "twirl"
        ):
            boxing_options["measure_annotations"] = "all"

        # Remove any existing final measurements
        prepared_circuit = circuit.remove_final_measurements(inplace=False)

        # Add final measurements
        creg = ClassicalRegister(prepared_circuit.num_qubits, "_meas")
        try:
            prepared_circuit.add_register(creg)
        except CircuitError as ex:
            raise ValueError("Name `_meas` is reserved for a dedicated classical register.") from ex
        prepared_circuit.barrier()
        prepared_circuit.measure(prepared_circuit.qubits, creg)

        try:
            boxing_pm = generate_boxing_pass_manager(**boxing_options)
        except Exception as ex:
            raise ValueError(
                f"Failed to generate boxing pass manager with the following error, {ex}"
            ) from ex
        boxed_circuit = boxing_pm.run(prepared_circuit)
        return boxed_circuit

    @classmethod
    def _make_samplex_arguments(
        cls,
        samplex: Samplex,
        boxed_circuit: QuantumCircuit,
        flat_parameter_values: np.typing.NDArray[np.float64],
        change_basis: np.typing.NDArray[np.uint8],
    ) -> dict[str, Any]:
        """Build a samplex args dictionary consisting of ``change_basis`` and parameters data.

        Args:
            samplex: A samplex object to create args to.
            boxed_circuit: A boxed circuit related to the samplex.
            flat_parameter_values: A flattened array of parameter values.
            change_basis: An array of bases to change.

        Returns:
            A samplex args dictionary.
        """
        # Prepare samplex_arguments
        samplex_arguments: dict[str, Any] = {}
        if samplex.inputs().get_specs("parameter_values"):
            if flat_parameter_values.shape == (0,):
                raise ValueError(
                    "flat_parameter_values must not be empty for a parameterized circuit."
                )
            samplex_arguments["parameter_values"] = flat_parameter_values

        # Set changing basis gates
        for spec in samplex.inputs().get_specs("basis_changes"):
            # Default to np.zeros, to ensure that every mid-circuit measurement
            # that may be present is performed without basis changing gates
            samplex_arguments[spec.name] = np.zeros(spec.shape)

        # Finalize basis changing gates for the final measurements
        for instr in boxed_circuit.reverse_ops():
            op = instr.operation
            if op.name == "box" and (change_basis_annot := get_annotation(op, ChangeBasis)):
                samplex_arguments[f"basis_changes.{change_basis_annot.ref}"] = change_basis
                break
        else:
            # This should not be reachable
            raise ValueError("Could not find a change basis annotation.")

        return samplex_arguments

    @staticmethod
    def _unbroadcast_index(
        bc_index: tuple[int | slice, ...], shape: tuple[int, ...]
    ) -> tuple[int | slice, ...]:
        """Index an array using an index from a compatible broadcasted shape.

        An ND-array ``arr`` is broadcastable to any shape ``bc_shape = (*pad_shape, *arr.shape)``.
        This function allows indexing ``arr`` using an ND-index or slice from ``bc_shape`` and
        will return the index for ``arr`` that accesses the same value.

        Args:
            bc_index: An ND-index from a broadcasted shape.
            shape: The shape of the broadcasting compatible array to index.

        Returns:
            The equivalent un-broadcasted ND-index of the array with specified shape.
        """

        @lru_cache
        def _pad_broadcast_shape(shape: tuple[int, ...], ndims: int) -> tuple[int | slice, ...]:
            # Pad a shape with trivial dimensions.
            shape_ndims = len(shape)
            pad = ndims - shape_ndims
            if pad > 0:
                return pad * (1,) + shape
            return shape

        shape_ndims = len(shape)
        if shape_ndims == 0:
            return ()

        pad_shape = _pad_broadcast_shape(shape, len(bc_index))
        bc_index = tuple(0 if dim == 1 else i for i, dim in zip(bc_index, pad_shape, strict=True))
        return bc_index[-shape_ndims:]

    def _create_samplex_arguments(
        self, samplex: Samplex, boxed_circuit: QuantumCircuit
    ) -> tuple[dict[str, Any], tuple[int | None, ...]]:
        """Build a samplex args dictionary.

        Args:
            samplex: A samplex object to create args to.
            boxed_circuit: A boxed circuit related to the samplex.

        Returns:
            A samplex args dictionary.
        """
        conf_shape: tuple[int, ...]
        if self.broadcast_obs_and_params:
            parameter_values, change_basis, param_basis_pairs = (
                self._compute_broadcasted_samplex_arguments()
            )
            self.param_basis_pairs = param_basis_pairs
            conf_shape = (change_basis.shape[0],)
        else:
            change_basis, basis_obs_term_map = get_measurement_bases(self._sparse_observables)
            meas_bases = [basis for basis in basis_obs_term_map]
            self.meas_bases = meas_bases
            if self.parameters is not None:
                parameter_values = self.parameters.as_array()
                conf_shape = (change_basis.shape[0], len(parameter_values))
            else:
                parameter_values = np.empty(0)
                conf_shape = (change_basis.shape[0],)

        samplex_args = self._make_samplex_arguments(
            samplex, boxed_circuit, parameter_values, change_basis
        )
        samplex_shape = (self.num_randomizations, *conf_shape)
        return samplex_args, samplex_shape

    @staticmethod
    def _compute_param_basis_pairs(
        observables: ObservablesArray,
        parameters_shape: tuple[int, ...],
        broadcast_shape: tuple[int, ...],
        measure_bases: PauliList,
    ) -> list[tuple[tuple[int, ...], str]]:
        """Compute a map between parameter indices and the measurement bases needed for each.

        For each parameter index (under broadcasting rules), this function determines which
        bases from ``measure_bases`` are required to measure all observable terms associated
        with that parameter index.

        A basis is considered compatible with an observable term when, on every qubit where
        the term is non-identity, the basis measures the same Pauli axis (qubit-wise commuting).

        Args:
            observables: The observables whose expectation values are to be measured.
            parameters_shape: The shape of the parameter values array.
            broadcast_shape: The broadcasted shape of ``observables`` and the parameter values.
            measure_bases: The set of candidate measurement bases (one entry per measurement
                configuration) from which compatible bases are selected.

        Returns:
            A list of ``N`` tuples ``(ndindex, basis_label)`` describing the correspondence
            between parameter indices and measurement bases.  For the i-th tuple:

            - ``ndindex`` is the N-dimensional index of the parameter entry.
            - ``basis_label`` is the string label of the measurement basis associated with
              that parameter entry.
        """
        param_basis_map: dict[tuple[int, ...], dict[Pauli, None]] = {}
        for bcast_index in np.ndindex(broadcast_shape):
            param_index = cast(
                tuple[int, ...],
                MitigationTask._unbroadcast_index(bcast_index, parameters_shape),
            )
            obs = observables[MitigationTask._unbroadcast_index(bcast_index, observables.shape)]
            if param_index not in param_basis_map:
                param_basis_map[param_index] = {}
            for basis in measure_bases:
                for obs_term, _ in obs.items():
                    pauli_obs_term = _convert_pauli_basis(obs_term)
                    pauli_support = np.logical_or(pauli_obs_term.z, pauli_obs_term.x)
                    if np.array_equal(
                        pauli_obs_term.z[pauli_support], basis.z[pauli_support]
                    ) and np.array_equal(pauli_obs_term.x[pauli_support], basis.x[pauli_support]):
                        param_basis_map[param_index][basis] = None
                        break  # this basis is already matched; move to the next basis

        param_basis_pairs: list[tuple[tuple[int, ...], str]] = [
            (ndindex, basis.to_label())
            for ndindex, bases in param_basis_map.items()
            for basis in bases
        ]

        return param_basis_pairs

    def _compute_broadcasted_samplex_arguments(
        self,
    ) -> tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.uint8],
        list[tuple[tuple[int, ...], str]],
    ]:
        """Compute parameter values and basis changes to be used as inputs by the samplex.

        To minimize the total number of circuit executions, this function takes the following
        steps:
            1. It creates a map between subsets of parameters and the observables that need to
               be measured for each subset, applying broadcasting rules to params and observables.
            2. It replaces the observables in that map with the minimal set of Pauli bases that
               can be used to measure all such observables.
            3. It flattens the map into two 1D arrays of equal length, containing the subsets of
               parameters and basis changing gates respectively. When a subset of parameters maps
               to more than one basis changing gate, the flattened array contains multiple copies
               of it.

        Overall, the two 1D arrays returned contain ``N`` elements, where ``N`` is the total number
        of basis changing gates that need to be measured across all the different parameter sets.
        Zipping them yields parameter-basis pairs, where each parameter value must be measured using
        its associated basis change.

        The two arrays have the format required by samplomatic and can be pass straight to the samplex
        via ``samplex.inputs()``.

        Return:
            A tuple ``(flat_parameter_values, change_basis, param_basis_pairs)`` where:

                * ``flat_parameter_values`` is a 1-D array of parameter values in the format expected
                by ``samplex.inputs()``. The array is of length ``N``, the total number of
                basis-changing gates required across all parameter sets.

                * ``change_basis`` is a 1-D array of length ``N`` containing the basis-changing gates
                associated with ``flat_parameter_values``, also in the format expected by
                ``samplex.inputs()``.

                * ``param_basis_pairs`` is a list of ``N`` tuples ``(ndindex, basis)`` describing the
                correspondence between the two arrays. For the i-th tuple:
                    - ``ndindex`` is the N-dimensional index of the parameter entry in
                        ``pub.parameter_values``.
                    - ``basis`` is the measurement basis associated with that parameter entry.
        """
        # Step 1.
        # enerate a map between param indices to pauli bases and the observable terms that they
        # measure
        param_obs_map: dict[set] = defaultdict(lambda: defaultdict(set))  # type: ignore[type-arg]
        for bcast_index in np.ndindex(self.broadcast_shape):
            param_index = self._unbroadcast_index(bcast_index, self.param_shape)
            obs = self.observables[self._unbroadcast_index(bcast_index, self.observables.shape)]
            for obs_term, _ in obs.items():
                pauli_basis = _convert_pauli_basis(obs_term)
                param_obs_map[param_index][pauli_basis].add(obs_term)

        # Step 2.
        # Collect the Paulis to measure for each parameter value in commuting sets
        param_meas_groups_map = {}
        for param_index, pauli_map in param_obs_map.items():
            pauli_set = list(pauli_map)
            meas_groups = PauliList(pauli_set).group_commuting(qubit_wise=True)
            param_meas_groups_map[param_index] = meas_groups

        # Figure out measurement Pauli basis for each set of commuting Paulis
        param_basis_map = {}
        for param_index, meas_groups in param_meas_groups_map.items():
            param_basis_map[param_index] = [
                Pauli((np.logical_or.reduce(paulis.z), np.logical_or.reduce(paulis.x)))
                for paulis in meas_groups
            ]

        # Step 3. Flatten the params.
        # We flatten params into a 1D array and generate a corresponding 1D `change_basis` array. Both
        # arrays contain ``num_basis`` elements.
        num_basis = sum(len(basis) for basis in param_basis_map.values())
        flat_parameter_values = np.empty((num_basis, self.parameters.num_parameters), dtype=float)
        change_basis = np.empty((num_basis, self.observables.num_qubits), dtype=int)

        basis_idx = 0
        for ndindex, basis in param_basis_map.items():
            for bases in basis:
                change_basis[basis_idx] = _pauli_to_ints(bases)
                flat_parameter_values[basis_idx] = self.parameters.as_array(
                    self.circuit.parameters
                )[ndindex]
                basis_idx += 1

        # Step 4. Log info.
        param_basis_pairs: list[tuple[tuple[int, ...], str]] = [
            (ndindex, bases.to_label())
            for ndindex, basis in param_basis_map.items()
            for bases in basis
        ]

        return flat_parameter_values, change_basis, param_basis_pairs

    def _save_basic_variables(
        self,
        circuit: QuantumCircuit,
        observables: ObservablesArray | Sequence[SparsePauliOp],
        parameters: np.ndarray | BindingsArray | None,
        custom_boxing_options: dict | None,
        shots_per_randomization: int,
        num_randomizations: int,
        broadcast_obs_and_params: bool,
        trex: TREX | None,
        quantum_program: QuantumProgram | None,
    ):
        """Save the basic variables of the circuit and observables as internal variables.

        Args:
            circuit: The circuit to execute.
            observables: The observables to calculate their expectation values.
            parameters: The parameters of a parametric circuit.
            custom_boxing_options: The custom boxing options that will be used by :meth:`~samplomatic.transpiler.generate_boxing_pass_manager` function.
            shots_per_randomization: The number of shots per randomization.
            num_randomizations: The number of randomizations.
            broadcast_obs_and_params: Whether to broadcast observables and parameters.
            trex: A TREX mitigation instance that will be used to mitigate readout errors.
            quantum_program: The quantum program to add an item for.
        """
        # save variables in the class
        self.circuit = circuit
        # convert to ObservablesArray
        self.observables = ObservablesArray(observables)
        self._sparse_observables = [
            SparsePauliOp.from_sparse_observable(sparse_obs)
            for sparse_obs in self.observables.ravel().sparse_observables_array()
        ]
        if parameters is not None and isinstance(parameters, np.ndarray):
            values = {tuple(circuit.parameters): parameters}
            self.parameters = BindingsArray.coerce(values)
        else:
            self.parameters = parameters
        self.param_shape = self.parameters.shape if self.parameters is not None else None
        self.broadcast_obs_and_params = broadcast_obs_and_params

        if self.parameters is None:
            self.broadcast_shape = self.observables.shape
            self.broadcast_obs_and_params = False
        elif broadcast_obs_and_params:
            try:
                self.broadcast_shape = np.broadcast_shapes(self.observables.shape, self.param_shape)
            except ValueError as ex:
                raise ValueError(
                    f"The observables shape {self.observables.shape} and the "
                    f"parameter values shape {self.param_shape} are not broadcastable."
                ) from ex
        else:
            self.broadcast_shape = self.observables.shape + self.param_shape

        if quantum_program is not None and quantum_program.shots != shots_per_randomization:
            raise ValueError(
                "shots_per_randomization must be equal to number of shots of the quantum program."
            )

        custom_boxing_options = (
            {} if custom_boxing_options is None else copy.deepcopy(custom_boxing_options)
        )
        self.custom_boxing_options = custom_boxing_options
        self.num_randomizations = num_randomizations
        self.shots_per_randomization = shots_per_randomization
        self.trex = trex

    def _extract_item_data(
        self,
        results: QuantumProgramItemResult | QuantumProgramResult,
        measure_noise_data: PauliLindbladMap | np.ndarray | None = None,
    ) -> tuple[QuantumProgramItemResult, PauliLindbladMap | np.ndarray | None]:
        """Extract data from ``results`` and ``measure_noise_data``."""
        if isinstance(results, QuantumProgramResult):
            item_result = results[self._program_item_index]
            if self.trex is not None and measure_noise_data is None:
                if self.trex.has_calibration_result():
                    measure_noise_data = self.trex.compute_noise_model(results)
                else:
                    warnings.warn(
                        "TREX is enabled in the task but measure noise data was not provided and a TREX calibration circuit was not found in the quantum program result. TREX mitigation will not be applied.",
                        stacklevel=2,
                    )
        else:
            item_result = results
            if self.trex is not None and measure_noise_data is None:
                warnings.warn(
                    "TREX is enabled in the task but measure noise data was not provided while only the task result is given as input. TREX mitigation will not be applied.",
                    stacklevel=2,
                )
        return item_result, measure_noise_data

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

    @staticmethod
    def _has_projection_operators(observables: ObservablesArray | Sequence[SparsePauliOp]) -> bool:
        """Return whether a set of observables contain projection operators."""
        observables = ObservablesArray(observables)
        projection_set = set("01rl+-")
        for observable in observables.ravel():
            for observable_term in observable:
                if bool(set(observable_term) & projection_set):
                    return True
        return False

    def find_unique_layers(
        self, circuit: QuantumCircuit, custom_boxing_options: dict | None = None
    ) -> list[CircuitInstruction]:
        """Return the unique boxed layers of the given circuit using the given boxing options.

        Args:
            circuit: The circuit to found its unique layers.
            custom_boxing_options: The custom boxing options that will be used by :meth:`~samplomatic.transpiler.generate_boxing_pass_manager` function.

        Returns:
            Unique boxed layers of the given circuit.
        """
        custom_boxing_options = (
            {} if custom_boxing_options is None else copy.deepcopy(custom_boxing_options)
        )
        boxed_circuit = self._box_circuit(circuit, custom_boxing_options)
        instructions = (box for box in boxed_circuit)
        unique_boxes: list[CircuitInstruction] = find_unique_box_instructions(
            instructions=instructions, normalize_annotations=None, undress_boxes=True
        )
        return unique_boxes

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
    ) -> QuantumProgram:
        """Creates a :class:`~.QuantumProgram` for executing via Executor.

        Creates an item for a :class:`~.QuantumProgram`, that can be executed via Executor.
        If a ``quantum_program`` is provided, the new item will be added to the existing program,
        otherwise, a new program will be created, containing only the created item.
        The basic options creates an item without any mitigation methods applied to the circuit.
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

        Returns:
            A :class:`~.QuantumProgram` that can be executed via Executor.
        """
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
            "mitigation": None,
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
        }

        self._add_data_to_passthrough_data(data_for_passthrough, quantum_program)
        return quantum_program

    def postprocess(
        self,
        results: QuantumProgramItemResult | QuantumProgramResult,
        measure_noise_data: PauliLindbladMap | np.ndarray | None = None,
    ) -> PubResult:
        """Process expectation values for a single item result.

        Args:
            results: The execution results. Can be either the entire results object or the item result of this task.
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

        return self.compute_expectation_value(
            item_result,
            self.observables,
            param_shape=self.param_shape,
            param_basis_pairs=self.param_basis_pairs,
            meas_bases=self.meas_bases,
            broadcast_obs_and_params=self.broadcast_obs_and_params,
            measure_noise_data=measure_noise_data,
        )

    @staticmethod
    def _process_broadcasted_expectation_values(
        data: np.ndarray,
        observables: ObservablesArray,
        param_shape: tuple[int, ...],
        param_basis_pairs: list[tuple[tuple[int, ...], str]],
        trex_scale_factors: dict[str, float] | None = None,
    ) -> tuple[
        np.typing.NDArray[np.float64], np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]
    ]:
        """Process expectation values for a single item result.

        Calculate the expectation values for an item in which the observables and parameters
        were broadcasted.

        Args:
            data: The result data to process.
            observables: The observables to calculate expectation values for.
            param_shape: The shape of the parameter values.
            param_basis_pairs: The map between params ndindexes to measure basis.
            trex_scale_factors: A dictionary mapping each observable term to its scale factor for TREX mitigation.

        Returns:
            A tuple ``(exp_vals, stds, ensemble_stds)``, where ``exp_vals`` are expectation values,
            ``stds`` are standard deviations, and ``ensemble_stds`` are ensemble standard errors.
            ``stds`` is the variance between different randomizations, while ``ensemble_stds`` is
            the variance of the expectation value with the shots from all the randomizations
            combined.
        """
        # Get number of randomizations and shots per randomization
        num_randomizations = data.shape[0]
        shots_per_randomization = data.shape[-2]
        total_shots = num_randomizations * shots_per_randomization

        try:
            broadcast_shape = np.broadcast_shapes(param_shape, observables.shape)
        except ValueError as ex:
            raise ValueError(
                f"Cannot broadcast ``param_shape`` {param_shape} and ``observables`` shape "
                f"{observables.shape}"
            ) from ex

        exp_vals = np.empty(broadcast_shape, dtype=float)
        stds = np.empty(broadcast_shape, dtype=float)
        ensemble_stds = np.empty(broadcast_shape, dtype=float)

        # Build efficient lookup: param_ndindex -> list of (measurement_basis, config_idx)
        # This allows us to find all available measurement bases for a given parameter
        config_lookup: dict[tuple, list] = {}
        for config_idx, (param_ndindex, basis_label) in enumerate(param_basis_pairs):
            config_lookup.setdefault(tuple(param_ndindex), []).append(
                (Pauli(basis_label), config_idx)
            )

        # Loop over the broadcast output shape
        for bcast_index in np.ndindex(broadcast_shape):
            # Unbroadcast to get the actual parameter and observable indices
            param_index = MitigationTask._unbroadcast_index(bcast_index, param_shape)
            obs_index = MitigationTask._unbroadcast_index(bcast_index, observables.shape)

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

                # Get measurement data for this configuration
                # datum shape: (num_randomizations, shots_per_randomization, num_qubits)
                datum = data[:, config_idx, :, :]
                term_exp_val, term_ensemble_variance, term_twirl_variance = (
                    _compute_single_term_exp_val_with_twirl_variance(observable_term, datum)
                )

                # Calculate scale factor in case TREX mitigation is used
                term_scale_factor = (
                    trex_scale_factors[observable_term] if trex_scale_factors is not None else 1
                )

                # Accumulate with coefficient
                exp_val += coeff * term_exp_val * term_scale_factor
                ensemble_variance += (coeff**2) * term_ensemble_variance * (term_scale_factor**2)
                twirl_variance += (coeff**2) * term_twirl_variance * (term_scale_factor**2)

            exp_vals[bcast_index] = exp_val
            ensemble_stds[bcast_index] = np.sqrt(ensemble_variance / total_shots)
            # When twirling is off (num_randomizations=1), stds equals ensemble_standard_error
            if num_randomizations == 1:
                stds[bcast_index] = ensemble_stds[bcast_index]
            else:
                stds[bcast_index] = np.sqrt(twirl_variance / num_randomizations)

        return exp_vals, stds, ensemble_stds

    @staticmethod
    def create_instance_from_passthrough_data(
        passthrough: dict[str, Any], trex: TREX | None = None
    ) -> MitigationTask:
        """Create a MitigationTask instance from a passthrough dictionary loaded from a quantum program execution result.

        Args:
            passthrough: Passthrough_data dictionary loaded from a quantum program execution result.
            trex: A TREX instance containing a calibration circuit results executed in the same quantum program.
                Should remain ``None`` in case TREX mitigation was not used or a TREX calibration was not executed as
                part of thq same quantum program.

        Returns:
            A MitigationTask instance.
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

        task = MitigationTask()
        task.observables = ObservablesArray.coerce(observables)
        task.param_basis_pairs = param_basis_pairs
        task.param_shape = param_shape
        task.broadcast_obs_and_params = broadcast_obs_and_params
        task.meas_bases = meas_bases
        task._program_item_index = program_item_index
        task.trex = trex
        return task

    @staticmethod
    def compute_expectation_value(
        item_result: QuantumProgramItemResult,
        observables: ObservablesArray | Sequence[SparsePauliOp],
        param_shape: tuple[int, ...] | None = None,
        param_basis_pairs: list[tuple[tuple[int, ...], str]] | None = None,
        meas_bases: Sequence[Pauli] | Sequence[str] | PauliList | None = None,
        broadcast_obs_and_params: bool = True,
        measure_noise_data: PauliLindbladMap | np.ndarray | None = None,
    ) -> PubResult:
        """Process expectation values for a single item result.

        This function can be used to calculate expectation values for a single unmitigated item result without instantiating a new class instance.

        Args:
            item_result: The item result.
            observables: The observables to calculate expectation values for.
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
            ValueError: If ``param_shape`` and ``observables.shape`` cannot be broadcasted against
                each other.
        """
        try:
            data = item_result["_meas"].copy()
        except KeyError as ex:
            raise ValueError("Dedicated creg ``'_meas'`` is missing from the results.") from ex

        trex_scale_factors = (
            TREX.trex_factors_each_term(measure_noise_data, observables)
            if measure_noise_data is not None
            else None
        )

        if broadcast_obs_and_params:
            if param_basis_pairs is None or param_shape is None:
                raise ValueError(
                    "``param_basis_pairs`` or ``param_shape`` is None while broadcasting of "
                    "observables and parameters is True."
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

            exp_vals, twirl_stds, ensemble_stds = (
                MitigationTask._process_broadcasted_expectation_values(
                    data,
                    observables,
                    param_shape,
                    param_basis_pairs,
                    trex_scale_factors,
                )
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
            rescale_factors=trex_scale_factors,
        )

        exp_vals, ensemble_stds = (np.array(x) for x in zip(*exp_val_and_std, strict=True))
        data_bin = DataBin(evs=exp_vals, stds=ensemble_stds, shape=exp_vals.shape)
        return PubResult(data=data_bin)
