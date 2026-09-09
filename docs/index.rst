#################
Qiskit Mitigation
#################

Qiskit Mitigation is a package for handling noise in quantum computations. It contains a set of techniques that can be used to build advanced or customized error mitigation pipelines directly with `samplomatic <https://github.com/Qiskit/samplomatic>`__. Specifically, it contains:

1. Implementations of PEC, TREX, PEA, and gate folding ZNE 
2. Functionality for computing a postselected noise channel, from circuit symmetries or spacetime checks, to be used with PEC to reduce sampling overhead
3. Postselection using non-Markovian error checks to filter corrupted samples out of measured data
4. Support for expectation value calculation with advanced error mitigation

Contributing
------------

The source code is available `on GitHub <https://github.com/Qiskit/qiskit-mitigation>`_.

The developer guide is located at `CONTRIBUTING.md <https://github.com/Qiskit/qiskit-mitigation/blob/main/CONTRIBUTING.md>`_
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's `code of conduct <https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

We use `GitHub issues <https://github.com/Qiskit/qiskit-mitigation/issues/new/choose>`_ for tracking requests and bugs.

License
-------

`Apache License 2.0 <https://github.com/Qiskit/qiskit-mitigation/blob/main/LICENSE.txt>`_

Deprecation Policy
------------------

We follow `semantic versioning <https://semver.org/>`_. We may occasionally make breaking changes in order to
improve the user experience. When possible, we will keep old interfaces and mark them as deprecated, as long
as they can co-exist with the new ones. Each substantial improvement, breaking change, or deprecation will be
documented in the `release notes <https://quantum.cloud.ibm.com/docs/api/qiskit-mitigation/release-notes>`_.

.. toctree::
   :hidden:

   Documentation home <self>
   Installation instructions <install>
   Guides <guides/index>
   GitHub <https://github.com/Qiskit/qiskit-mitigation>

.. toctree::
   :hidden:
   :caption: Tutorials

   Postselection with non-Markovian error checks <https://quantum.cloud.ibm.com/docs/guides/postselection_with_non_markovian_error_checks.ipynb>

.. toctree::
   :hidden:
   :caption: API reference

   Python API reference <apidocs/index>
