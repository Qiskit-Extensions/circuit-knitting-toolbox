##########################
How to specify the problem
##########################

To specify the problem, we follow the Qiskit Nature workflow. We set up the ``molecule`` object, specify the driver, and instantiate the ``ElectronicStructureProblem``.

We first require the following modules:

.. jupyter-execute::

   from qiskit_nature.drivers.formats import MoleculeInfo
   from qiskit_nature.second_q.drivers import PySCFDriver

To set up the ``molecule`` object, we specify the individual atoms and their positions:

.. jupyter-execute::

   molecule = MoleculeInfo(
       ["H", "H"],
       [
           (0.0, 0.0, 0.0),
           (0.0, 0.0, 0.735),
       ],
       charge=0,
       multiplicity=1, # Multiplicity (2S+1) of the molecule, where S is the total spin angular momentum
   )

We then specify the driver (see ``PySCFDriver``) and load the molecule into the driver:

.. jupyter-execute::

   driver = PySCFDriver.from_molecule(molecule=molecule, basis="sto6g")

Here, the driver is an algorithm class that knows how to calculate the second quantized operators.

Finally, we instantiate the ``ElectronicStructureProblem``, a class which wraps a number of different types of drivers:

.. jupyter-execute::

   problem = driver.run()
