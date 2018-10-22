from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from ase.calculators.calculator import Calculator, all_changes


class OpenMMCalculator(Calculator):
    """
    Simple implementation of a ASE calculator for OpenMM.

    Parameters:
        input : PDB file with topology.
        nonbondedMethod : The nonbonded method to use (see https://simtk.org/api_docs/openmm/api10/classOpenMM_1_1NonbondedForce.html). Defaults to CutoffNonPeriodic.
        nonbondedCutoff : The nonbonded cutoff distance to use (in Angstroms). Default : 10 Angstroms.
    """
    implemented_properties = ['energy', 'forces']
    default_parameters = {'input' : "openmm.pdb",
                          'nonbondedMethod' : CutoffNonPeriodic,
                          'nonbondedCutoff' : 10 * angstrom}

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)
        input = self.parameters.input
        pdb = PDBFile(input)
        forcefield = ForceField('amber99sb.xml')

        print("Generating OpenMM system")
        self.system = forcefield.createSystem(pdb.topology, nonbondedMethod=self.parameters.nonbondedMethod,
                                              nonbondedCutoff=self.parameters.nonbondedCutoff)
        # Create a dummy integrator, this doesn't really matter.
        self.integrator = VerletIntegrator(0.001 * picosecond)
        self.platform = Platform.getPlatformByName("CPU")
        self.simulation = Simulation(pdb.topology, self.system, self.integrator, self.platform)

        self.simulation.context.setPositions(pdb.positions)
        state = self.simulation.context.getState(getEnergy=True)
        print("Energy: ", state.getPotentialEnergy(), len(pdb.positions))
        self.n_atoms = len(pdb.positions)

    def calculate(self, atoms=None,
                  properties=['energy', 'forces'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        positions = [x for x in atoms.positions]
        self.simulation.context.setPositions(positions * angstrom)
        state = self.simulation.context.getState(getEnergy=True, getForces=True)
        energyKJMol = state.getPotentialEnergy()
        kjMol2ev = 0.01036; # ...roughly
        energy = energyKJMol.value_in_unit(kilojoules_per_mole) * kjMol2ev
        forcesOpenmm = state.getForces()
        # There must be a more elegant way of doing this
        forces = [[f.value_in_unit(kilojoule_per_mole/angstrom) * kjMol2ev for f in force] for force in
                  forcesOpenmm]
        self.results['energy'] = energy
        self.results['forces'] = forces