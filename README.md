<!-- SHIELDS -->
<div align="left">

  [![Stability](https://img.shields.io/badge/Stability-alpha-f4d03f.svg)](https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/releases)
  ![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-informational)
  [![Python](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-informational)](https://www.python.org/)
  [![Qiskit](https://img.shields.io/badge/Qiskit-%E2%89%A5%200.39.0-6133BD)](https://github.com/Qiskit/qiskit)
  [![Qiskit Nature](https://img.shields.io/badge/Qiskit%20Nature-%E2%89%A5%200.4.4-6133BD)](https://github.com/Qiskit/qiskit-nature)
<br />
  [![License](https://img.shields.io/github/license/Qiskit-Extensions/circuit-knitting-toolbox?label=License)](LICENSE.txt)
  [![Code style: Black](https://img.shields.io/badge/Code%20style-Black-000.svg)](https://github.com/psf/black)
  [![Tests](https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/actions/workflows/test_latest_versions.yml/badge.svg)](https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/actions/workflows/test_latest_versions.yml)
  [![Coverage](https://coveralls.io/repos/github/Qiskit-Extensions/circuit-knitting-toolbox/badge.svg?branch=main)](https://coveralls.io/github/Qiskit-Extensions/circuit-knitting-toolbox?branch=main)

# Circuit Knitting Toolbox

### Table of Contents

* [About](#about)
* [Documentation](#documentation)
* [Deprecation Policy](#deprecation-policy)
* [References](#references)
* [License](#license)

----------------------------------------------------------------------------------------------------

### About

Circuit Knitting is the process of decomposing a quantum circuit into smaller circuits, executing those smaller circuits on a quantum processor(s), and then knitting their results into a reconstruction of the original circuit's outcome. Circuit knitting includes techniques such as entanglement forging, circuit cutting, and classical embedding. The Circuit Knitting Toolbox (CKT) is a collection of such tools.

Each tool in the CKT partitions a user's problem into quantum and classical components to enable efficient use of resources constrained by scaling limits, i.e. size of quantum processors and classical compute capability. It can assign the execution of "quantum code" to QPUs or QPU simulators and "classical code" to various heterogeneous classical resources such as CPUs, GPUs, and TPUs made available via hybrid cloud, on-prem, data centers, etc. 

The toolbox enables users to run parallelized and hybrid (quantum + classical) workloads without worrying about allocating and managing underlying infrastructure.

The toolbox currently contains the following tools:
- Entanglement Forging [[1]](#references)
- Circuit Cutting [[2]](#references)
  
----------------------------------------------------------------------------------------------------
  
### Documentation

The documentation, including installation instructions, is available at https://qiskit-extensions.github.io/circuit-knitting-toolbox/.

----------------------------------------------------------------------------------------------------

### Deprecation Policy

This project is meant to evolve rapidly and, as such, does not follow [Qiskit's deprecation policy](https://qiskit.org/documentation/contributing_to_qiskit.html#deprecation-policy).  We may occasionally make breaking changes in order to improve the user experience.  When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the new ones.  Each substantial improvement, breaking change, or deprecation will be documented in the release notes.

----------------------------------------------------------------------------------------------------

### References

[1] Andrew Eddins, Mario Motta, Tanvi P. Gujarati, Sergey Bravyi, Antonio Mezzacapo, Charles Hadfield, Sarah Sheldon, [Doubling the size of quantum simulators by entanglement forging](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.010309). PRX Quantum 3, 010309 (2022).

[2] Wei Tang, Teague Tomesh, Martin Suchara, Jeffrey Larson, Margaret Martonosi, [CutQC: Using Small Quantum Computers for Large Quantum Circuit Evaluations](https://doi.org/10.1145/3445814.3446758), Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems. pp. 473 (2021).

----------------------------------------------------------------------------------------------------

<!-- LICENSE -->
### License
[Apache License 2.0](LICENSE.txt)
