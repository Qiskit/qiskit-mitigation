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

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Constant values."""

DEFAULT_POST_SELECTION_SUFFIX = "_ps"
"""
The default suffix to append to the names of the classical registers used for post selection measurements.
"""

DEFAULT_SPECTATOR_CREG_NAME = "spec"
"""
The default name of the classical register used for measuring spectator qubits.
"""

# Constants for the ``postselection`` sub-package. Both pre- and post-circuit
# non-Markovian error checks feed a single post-selection routine; the "pre"/"post" labels
# below distinguish *where in the circuit* the non-Markovian error check is inserted (start vs.
# end), not two different selection techniques.

DEFAULT_POST_CHECK_SUFFIX = "_ps"
"""
The default suffix appended to classical registers holding post-circuit non-Markovian error check measurements.
"""

DEFAULT_PRE_CHECK_SUFFIX = "_pre"
"""
The default suffix appended to classical registers holding pre-circuit non-Markovian error check measurements.
"""

DEFAULT_SPECTATOR_PRE_CREG_NAME = "spec_pre"
"""
The default name of the classical register used for measuring spectator qubits in pre-circuit non-Markovian error checks.
"""

RX_PULSE_COUNT = 20
"""
The number of ``rx(pi / RX_PULSE_COUNT)`` gates used to emulate a slow X-pulse when ``x_pulse_type="rx"``.
"""
