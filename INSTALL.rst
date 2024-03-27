Installation Instructions
=========================

Let's see how to install the Circuit Knitting Toolbox (CKT). The first
thing to do is choose how you're going to run and install the
packages. There are three primary ways to do this:

- :ref:`Option 1`
- :ref:`Option 2`
- :ref:`Option 3`

CutQC users should consult the :ref:`Platform Support` section to determine
which installation option is appropriate for them. Users who wish to run within a
containerized environment may skip the pre-installation and move straight
to :ref:`Option 3`.

.. note::

    If a user wishes to use the circuit cutting toolbox to
    automatically find optimized wire cuts for a circuit too large for
    the free version of CPLEX, they should acquire a license and install
    the `full
    version <https://www.ibm.com/products/ilog-cplex-optimization-studio>`__.

Pre-Installation
^^^^^^^^^^^^^^^^

Users who wish to install locally (using either :ref:`Option 1` or :ref:`Option 2`) are encouraged to
follow a brief set of common instructions to prepare a Python environment for
installation of CKT:

First, create a minimal environment with only Python installed in it. We recommend using `Python virtual environments <https://docs.python.org/3.10/tutorial/venv.html>`__.

.. code:: sh
    
    python3 -m venv /path/to/virtual/environment

Activate your new environment.

.. code:: sh
    
    source /path/to/virtual/environment/bin/activate

Note: If you are using Windows, use the following commands in PowerShell:

.. code:: pwsh
    
    python3 -m venv c:\path\to\virtual\environment
    c:\path\to\virtual\environment\Scripts\Activate.ps1


.. _Option 1:

Option 1: Pip Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Upgrade pip and install the CKT package.

.. code:: sh

    pip install --upgrade pip
    pip install circuit-knitting-toolbox

Users intending to use the automatic cut finding functionality in the CutQC package should install the ``cplex`` optional dependency.

Adjust the options below to suit your needs.

.. code:: sh
    
    pip install 'circuit-knitting-toolbox[cplex]'


.. _Option 2:

Option 2: Install from Source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users who wish to develop in the repository or run the tutorials locally may want to install from source.

In either case, the first step is to clone the CKT repository.

.. code:: sh

    git clone git@github.com:Qiskit-Extensions/circuit-knitting-toolbox.git
    
Next, upgrade pip and enter the repository. 

.. code:: sh
    
    pip install --upgrade pip
    cd circuit-knitting-toolbox

The next step is to install CKT to the virtual environment. If you plan on running the tutorials, install the
notebook dependencies in order to run all the visualizations in the notebooks.
If you plan on developing in the repository, you may want to install the ``dev`` dependencies.

Users intending to use the automatic cut finding functionality in the CutQC package should install the ``cplex`` optional dependency.

Adjust the options below to suit your needs.

.. code:: sh
    
    pip install tox notebook -e '.[notebook-dependencies,dev,cplex]'

If you installed the notebook dependencies, you can get started with CKT by running the notebooks in the docs.

.. code::
    
    cd docs/
    jupyter notebook


.. _Option 3:

Option 3: Use within Docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have provided a `Dockerfile <https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/blob/main/Dockerfile>`__, which can be used to
build a Docker image, as well as a
`compose.yaml <https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/blob/main/compose.yaml>`__ file, which allows one
to use the Docker image with just a few simple commands.

.. code:: sh

    git clone git@github.com:Qiskit-Extensions/circuit-knitting-toolbox.git
    cd circuit-knitting-toolbox
    docker compose build
    docker compose up

Depending on your system configuration, you may need to type ``sudo``
before each ``docker compose`` command.

.. note::

   If you are instead using `podman <https://podman.io/>`_ and
   `podman-compose <https://github.com/containers/podman-compose>`_,
   the commands are:

   .. code:: sh

       podman machine start
       podman-compose --podman-pull-args short-name-mode="permissive" build
       podman-compose up

Once the container is running, you should see a message like this:

::

    notebook_1  |     To access the server, open this file in a browser:
    notebook_1  |         file:///home/jovyan/.local/share/jupyter/runtime/jpserver-7-open.html
    notebook_1  |     Or copy and paste one of these URLs:
    notebook_1  |         http://e4a04564eb39:8888/lab?token=00ed70b5342f79f0a970ee9821c271eeffaf760a7dcd36ec
    notebook_1  |      or http://127.0.0.1:8888/lab?token=00ed70b5342f79f0a970ee9821c271eeffaf760a7dcd36ec

Locate the *last* URL in your terminal (the one that includes
``127.0.0.1``), and navigate to that URL in a web browser to access the
Jupyter Notebook interface.

The home directory includes a subdirectory named ``persistent-volume``.
All work you’d like to save should be placed in this directory, as it is
the only one that will be saved across different container runs.


.. _Platform Support:

Platform Support
^^^^^^^^^^^^^^^^

Users of Mac M1 or M2 chips and Windows users may have issues running certain components of CKT.

If you are using Linux or macOS with an Intel chip (i.e., not the
new M1 or M2 chips), everything should work natively, so we
recommend either :ref:`Option 1` or :ref:`Option 2`.

All users on ARM chips, as well as all Windows users, may have to
take care when installing the toolbox, depending on which tools they
intend to use.
  
  - The automatic wire cut search in the ``cutqc`` package depends
    on CPLEX, which is only available on Intel chips and is not yet available
    for Python 3.12.

In each case, one method that is guaranteed to work is to :ref:`use
the toolbox within Docker <Option 3>`.  Other methods include:

  - Users on Apple's M series of chips may wish to install an x86
    version of Python.  For instance, `conda
    <https://docs.conda.io/en/latest/miniconda.html>`__ users can run
    ``CONDA_SUBDIR=osx-64 conda create -n x86_venv python=3`` to
    create a virtual environment that uses Python compiled for the x86
    instruction set.  No matter the installation method, there is a
    performance cost due to emulation.
