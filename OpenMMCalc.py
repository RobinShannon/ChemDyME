# Copyright (c) Intangible Realities Lab, University Of Bristol. All rights reserved.
# Licensed under the GPL. See License.txt in the project root for license information.
from typing import Optional
import numpy as np
from ase import Atoms, constraints
from ase.constraints import FixBondLengths
from ase.calculators.calculator import Calculator, all_changes
from simtk.openmm import System, State, XmlSerializer
from simtk.openmm.app import Simulation
from simtk.unit import angstrom, kilojoules_per_mole, kilojoule_per_mole, Quantity
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import simtk.openmm.app as app

class OpenMMCalculator(Calculator):
    """
    Simple implementation of a ASE calculator for OpenMM. Initialises an OpenMM context with the given
    serialised OpenMM simulation file.

    Parameters:
        input_xml :  An OpenMM simulation, serialised with :module: serializer.
        pbc       : Boolean determining whether to set periodic boundary conditions in ASE
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, input_xml, atoms: Optional[Atoms] = None, parallel = False, pbc=False, **kwargs):
        Calculator.__init__(self, **kwargs)
        f = open(input_xml, 'r')
        sys = f.read()
        self.system = XmlSerializer.deserialize(sys)
        positions = [x for x in atoms.get_positions()]
        self.integrator = VerletIntegrator(0.001 * picosecond)
        self.platform = Platform.getPlatformByName("CPU")
        if parallel:
            self.context = openmm.Context(self.system, self.integrator, self.platform)
        else:
            self.context = openmm.Context(self.system, self.integrator)
        self.context.setPositions(positions * angstrom)
        state = self.context.getState(getEnergy=True)
        print("Energy: ", state.getPotentialEnergy(), len(positions))
        self.n_atoms = len(positions)
        if pbc:
            self.set_periodic_bounds(atoms)
        self.set_constraints(atoms)
        self.kjmol_to_ev = 0.01036

    def calculate(self, atoms: Optional[Atoms] = None,
                  properties=('energy', 'forces'),
                  system_changes=all_changes):

        if atoms is None:
            atoms = self.atoms.copy()
        if atoms is None:
            raise ValueError('No ASE atoms supplied to calculator, and no ASE atoms supplied with initialisation.')

        self._set_positions(atoms.positions)
        energy, forces = self._calculate_openmm()
        if 'energy' in properties:
            self.results['energy'] = energy
        if 'forces' in properties:
            self.results['forces'] = forces

    def _calculate_openmm(self):
        state: State = self.context.getState(getEnergy=True, getForces=True)
        energy_kj_mol = state.getPotentialEnergy()
        energy = energy_kj_mol.value_in_unit(kilojoules_per_mole) * self.kjmol_to_ev
        forces_openmm = state.getForces(asNumpy=True)
        forces_angstrom = forces_openmm.value_in_unit(kilojoule_per_mole / angstrom)
        forces = forces_angstrom * self.kjmol_to_ev
        return energy, forces

    def _set_positions(self, positions):
        self.context.setPositions(positions * angstrom)

    def set_periodic_bounds(self, atoms: Atoms):
        """
        Sets ASE atoms object with the same periodic boundaries as that used in the given OpenMM system.
        :param atoms: ASE Atoms
        :return:
        """
        if self.system.usesPeriodicBoundaryConditions():
            boxvectors: Quantity = self.system.getDefaultPeriodicBoxVectors()
            atoms.set_pbc(True)
            atoms.set_cell(np.array([vector.value_in_unit(angstrom) for vector in boxvectors]))

    def set_constraints(self, atoms: Atoms):
        """
        Sets ASE atoms object with the same constraints as those used in the given OpenMM system.
        :param atoms: ASE Atoms
        :return:
        """
        fix = []
        for i in range(0,self.system.getNumConstraints()):
            index1, index2, distance = self.system.getConstraintParameters(i)
            fix .append([index1,index2])
        if len(fix) != 0:
            atoms.constraints = FixBondLengths(fix)


