# Copyright (c) Intangible Realities Lab, University Of Bristol. All rights reserved.
# Licensed under the GPL. See License.txt in the project root for license information.
from typing import Optional, Collection

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from scine_sparrow import Calculation

EV_PER_HARTREE = 27.2114
ANG_PER_BOHR = 0.529177


class SparrowCalculator(Calculator):
    """
    Simple implementation of an ASE calculator for Sparrow.

    Parameters:
        method :  The electronic structure method to use in calculations.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, atoms: Optional[Atoms] = None, method='PM6', **kwargs):
        super().__init__(**kwargs)
        self.atoms = atoms
        self.method = method

    def calculate(self, atoms: Optional[Atoms] = None,
                  properties=('energy', 'forces'),
                  system_changes=all_changes):

        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError('No ASE atoms supplied to calculator, and no ASE atoms supplied with initialisation.')
        self._calculate_sparrow(atoms, properties)

    def _calculate_sparrow(self, atoms: Atoms, properties: Collection[str]):
        positions = atoms.positions
        elements = atoms.get_chemical_symbols()
        calc = Calculation(method=self.method)
        calc.set_elements(elements)
        calc.set_positions(positions)
        kwargs = {property_name: True for property_name in properties}
        # TODO pass these to calculate in wrapper.
        if 'energy' in properties:
            energy_hartree = calc.calculate_energy()
            self.results['energy'] = energy_hartree * EV_PER_HARTREE
        if 'forces' in properties:
            #TODO make np array come out of wwrapper.
            gradients_hartree_bohr = np.array(calc.calculate_gradients())
            self.results['forces'] = - gradients_hartree_bohr * EV_PER_HARTREE / ANG_PER_BOHR
        return

