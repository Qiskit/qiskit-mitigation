> [!NOTE]
> This repository is under active development and the code here should not be considered stable.

# Qiskit Mitigation

Qiskit Mitigation is a package for handling noise in quantum computations. It contains a set of techniques that can be used to build advanced or customized error mitigation pipelines directly with [samplomatic](https://github.com/Qiskit/samplomatic). Specifically, it contains:

1. Implementations of Probabilistic Error Cancellation (PEC), Twirling Readout Error eXtinction (TREX), and Zero-Noise Extrapolation (ZNE) with Probabilistic Error Amplification (PEA) and gate folding amplification methods.
2. Functionality for computing a postselected noise channel, from circuit symmetries or spacetime checks, to be used with PEC to reduce sampling overhead [1].
3. Postselection using non-Markovian error checks to filter corrupted samples out of measured data.
4. Support for expectation value calculation with advanced error mitigation.

----------------------------------------------------------------------------------------------------

### Documentation

All documentation is available at https://quantum.cloud.ibm.com/docs/addons/qiskit-mitigation.

----------------------------------------------------------------------------------------------------

### Installation

We encourage installing this package via `pip`, when possible:

```bash
pip install 'qiskit-mitigation'
```

For more installation information refer to these [installation instructions](docs/install.rst).

----------------------------------------------------------------------------------------------------

### Contributing

The source code is available [on GitHub](https://github.com/Qiskit/qiskit-mitigation).

The developer guide is located at [CONTRIBUTING.md](https://github.com/Qiskit/qiskit-mitigation/blob/main/CONTRIBUTING.md)
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's [code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).

----------------------------------------------------------------------------------------------------

### Citing this package

If you use this package in your research, use the [CITATION.bib](CITATION.bib) file in this project’s repository to cite the appropriate reference(s).

----------------------------------------------------------------------------------------------------

### License

[Apache License 2.0](LICENSE.txt)

----------------------------------------------------------------------------------------------------

### Deprecation Policy

We follow [semantic versioning](https://semver.org/). We may occasionally make breaking changes in
order to improve the user experience. When possible, we will keep old interfaces and mark them as
deprecated, as long as they can co-exist with the new ones. Each substantial improvement, breaking
change, or deprecation will be documented in the [release notes](https://quantum.cloud.ibm.com/docs/api/qiskit-mitigation/release-notes).

----------------------------------------------------------------------------------------------------

### References

1. Laurin E. Fischer, et al., [Spacetime mitigation of logical errors](https://arxiv.org/abs/2609.13108), arXiv:2609.13108 [quant-ph].
