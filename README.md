<!-- SHIELDS -->
<div align="left">

  [![Stability](https://img.shields.io/badge/Stability-alpha-f4d03f.svg)](https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/releases)
  [![Release](https://img.shields.io/pypi/v/circuit-knitting-toolbox.svg?label=Release)](https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/releases)
  ![Platform](https://img.shields.io/badge/%F0%9F%92%BB%20Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
  [![Python](https://img.shields.io/pypi/pyversions/circuit-knitting-toolbox?label=Python&logo=python)](https://www.python.org/)
  [![Qiskit](https://img.shields.io/badge/Qiskit%20-%20%3E%3D1.0%20-%20%236133BD?logo=Qiskit)](https://github.com/Qiskit/qiskit)
<br />
  [![Docs (stable)](https://img.shields.io/badge/%F0%9F%93%84%20Docs-stable-blue.svg)](https://qiskit-extensions.github.io/circuit-knitting-toolbox/)
  [![DOI](https://zenodo.org/badge/543181258.svg)](https://zenodo.org/badge/latestdoi/543181258)
  [![License](https://img.shields.io/github/license/Qiskit-Extensions/circuit-knitting-toolbox?label=License)](LICENSE.txt)
  [![Downloads](https://img.shields.io/pypi/dm/circuit-knitting-toolbox.svg?label=Downloads)](https://pypi.org/project/circuit-knitting-toolbox/)
  [![Tests](https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/actions/workflows/test_latest_versions.yml/badge.svg)](https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/actions/workflows/test_latest_versions.yml)
  [![Coverage](https://coveralls.io/repos/github/Qiskit-Extensions/circuit-knitting-toolbox/badge.svg?branch=main)](https://coveralls.io/github/Qiskit-Extensions/circuit-knitting-toolbox?branch=main)

# Circuit Knitting Toolbox

### Table of Contents

* [About](#about)
* [Documentation](#documentation)
* [Installation](#installation)
* [Deprecation Policy](#deprecation-policy)
* [References](#references)
* [License](#license)

----------------------------------------------------------------------------------------------------

### About

Circuit Knitting is the process of decomposing a quantum circuit into smaller circuits, executing those smaller circuits on a quantum processor(s), and then knitting their results into a reconstruction of the original circuit's outcome.

Each tool in the CKT partitions a user's problem into quantum and classical components to enable efficient use of resources constrained by scaling limits, i.e. size of quantum processors and classical compute capability. It can assign the execution of "quantum code" to QPUs or QPU simulators and "classical code" to various heterogeneous classical resources such as CPUs, GPUs, and TPUs made available via hybrid cloud, on-prem, data centers, etc. 

The toolbox enables users to run parallelized and hybrid (quantum + classical) workloads without worrying about allocating and managing underlying infrastructure.

The toolbox currently contains the following tools:
- Circuit Cutting [[1-6]](#references)
  
----------------------------------------------------------------------------------------------------
  
### Documentation

All CKT documentation is available at https://qiskit-extensions.github.io/circuit-knitting-toolbox/.

----------------------------------------------------------------------------------------------------
  
### Installation

We encourage installing CKT via ``pip``, when possible. Users intending to use the automatic cut finding functionality in the ``CutQC`` package should install the ``cplex`` optional dependency.

```bash
pip install 'circuit-knitting-toolbox[cplex]'
```

For information on installing from source, running CKT in a container, and platform support, refer to the [installation instructions](https://qiskit-extensions.github.io/circuit-knitting-toolbox/install.html) in the CKT documentation.

----------------------------------------------------------------------------------------------------

### Deprecation Policy

This project is meant to evolve rapidly and, as such, does not follow [Qiskit's deprecation policy](https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md).  We may occasionally make breaking changes in order to improve the user experience.  When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the new ones.  Each substantial improvement, breaking change, or deprecation will be documented in the [release notes](https://qiskit-extensions.github.io/circuit-knitting-toolbox/release-notes.html).

----------------------------------------------------------------------------------------------------

### References

[1] Kosuke Mitarai, Keisuke Fujii, [Constructing a virtual two-qubit gate by sampling single-qubit operations](https://iopscience.iop.org/article/10.1088/1367-2630/abd7bc), New J. Phys. 23 023021.

[2] Kosuke Mitarai, Keisuke Fujii, [Overhead for simulating a non-local channel with local channels by quasiprobability sampling](https://quantum-journal.org/papers/q-2021-01-28-388/), Quantum 5, 388 (2021).

[3] Christophe Piveteau, David Sutter, [Circuit knitting with classical communication](https://arxiv.org/abs/2205.00016), arXiv:2205.00016 [quant-ph].

[4] Lukas Brenner, Christophe Piveteau, David Sutter, [Optimal wire cutting with classical communication](https://arxiv.org/abs/2302.03366), arXiv:2302.03366 [quant-ph].

[5] Wei Tang, Teague Tomesh, Martin Suchara, Jeffrey Larson, Margaret Martonosi, [CutQC: Using small quantum computers for large quantum circuit evaluations](https://doi.org/10.1145/3445814.3446758), Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems. pp. 473 (2021).
  
[6] K. Temme, S. Bravyi, and J. M. Gambetta, [Error mitigation for short-depth quantum circuits](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.180509), Physical Review Letters, 119(18), (2017).
  
----------------------------------------------------------------------------------------------------

<!-- LICENSE -->
### License
[Apache License 2.0](LICENSE.txt)
