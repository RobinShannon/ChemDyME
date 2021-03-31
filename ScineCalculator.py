# Copyright (c) Intangible Realities Lab, University Of Bristol. All rights reserved.
# Licensed under the GPL. See License.txt in the project root for license information.
from typing import Optional, Collection
import os
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from scine_sparrow import Calculation
import time

EV_PER_HARTREE = 27.2114
ANG_PER_BOHR = 0.529177

class SparrowCalculator(Calculator):
    """
    Simple implementation of an ASE calculator for Sparrow.

    Parameters:
        method :  The electronic structure method to use in calculations.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, atoms: Optional[Atoms] = None, method='AM1', **kwargs):
        super().__init__(**kwargs)
        self.atoms = atoms
        self.method = method
        #self.calc =  Calculation(method = self.method)
        if atoms is None:
            self.has_atoms = False

    def calculate(self, atoms: Optional[Atoms] = None,
                  properties=('energy', 'forces'),
                  system_changes=all_changes):
        self.calc = Calculation(method=self.method)
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError('No ASE atoms supplied to calculator, and no ASE atoms supplied with initialisation.')
        if not self.has_atoms:
            sym = atoms.get_chemical_symbols()
            is_O = len(sym) == 1 and sym[0] == 'O'
            is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] == 'O'
            s = sum(atoms.get_atomic_numbers())
            if s % 2 != 0:
                self.spin_mult = 2
                self.unrestricted = True
            elif is_O or is_OO:
                self.spin_mult = 3
                self.unrestricted = False
            else:
                self.spin_mult = 1
                self.unrestricted = False
            t1 = time.clock()

            self.calc.set_elements(sym)
            settings = {}
            settings['spin_multiplicity'] = self.spin_mult
            settings['unrestricted_calculation'] = self.unrestricted
            self.calc.set_settings(settings)
        self.has_atoms = False
        self._calculate_sparrow(atoms, properties)
        t2 = time.clock()


    def _calculate_sparrow(self, atoms: Atoms, properties: Collection[str]):
        positions = atoms.positions
        self.calc.set_positions(positions)
        if 'energy' in properties:
            energy_hartree = self.calc.calculate_energy()
            self.results['energy'] = energy_hartree * EV_PER_HARTREE
        if 'forces' in properties:
            gradients_hartree_bohr = np.array(self.calc.calculate_gradients())
            self.results['forces'] = - gradients_hartree_bohr * EV_PER_HARTREE / ANG_PER_BOHR
        return


    def close(self):
        self.calc = None

