# Copyright (c) Intangible Realities Lab, University Of Bristol. All rights reserved.
# Licensed under the GPL. See License.txt in the project root for license information.
from typing import Optional, Collection
import os
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from scine_sparrow import Calculation
from ase.io import write, read
import scine_readuct
import io
import contextlib
import sys

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
        # Determine spin multiplicity
        print('calculating_sparrow')
        sym = atoms.get_chemical_symbols()
        print('calculating_sparrow 2')
        is_O = len(sym) == 1 and sym[0] == 'O'
        print('calculating_sparrow 3')
        is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] == 'O'
        print('calculating_sparrow 4')
        s = sum(atoms.get_atomic_numbers())
        print('calculating_sparrow 5')

        calculation = Calculation('AM1')
        calculation.set_elements(['H', 'H'])
        calculation.set_positions([[0, 0, 0], [1, 0, 0]])
        calculation.calculate_energy()

        if s % 2 != 0:
            self.spin_mult = 2
            self.unrestricted = True
        elif is_O or is_OO:
            self.spin_mult = 3
            self.unrestricted = False
        else:
            self.spin_mult = 1
            self.unrestricted = False

        positions = atoms.positions
        print('calculating_sparrow 6')
        elements = atoms.get_chemical_symbols()
        calc = Calculation()
        calc.set_elements(elements)
        calc.set_positions(positions)
        print('calculating_sparrow 7')
        settings = {}
        settings['spin_multiplicity'] = self.spin_mult
        settings['unrestricted_calculation'] = self.unrestricted
        calc.set_settings(settings)
        print('calculating_sparrow 8 ')
        if 'energy' in properties:
            print('calculating_sparrow 9 ')
            energy_hartree = calc.calculate_energy()
            self.results['energy'] = energy_hartree * EV_PER_HARTREE
            print('energy = ' + str(energy_hartree * EV_PER_HARTREE))
        if 'forces' in properties:
            print('calculating_sparrow 11')
            gradients_hartree_bohr = np.array(calc.calculate_gradients())
            self.results['forces'] = - gradients_hartree_bohr * EV_PER_HARTREE / ANG_PER_BOHR
            print('gradients = ' + str(gradients_hartree_bohr * EV_PER_HARTREE / ANG_PER_BOHR))
        return

    def minimise_stable(self,path, atoms: Optional[Atoms] = None, ):
        current_dir = os.getcwd()
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
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
        if atoms is None:
            atoms = self.atoms
        atoms.write('temp.xyz')
        system1 = scine_readuct.load_system('temp.xyz', self.method, program='Sparrow',
                                            molecular_charge=0, unrestricted_calculation=self.unrestricted, spin_multiplicity=self.spin_mult)
        systems = {}
        systems['reac'] = system1
        try:
            systems, success = scine_readuct.run_opt_task(systems, ['reac'], output = ['reac_opt'], optimizer ='bfgs', allow_unconverged = True)
            atoms.set_positions(systems['reac_opt'].positions * ANG_PER_BOHR)
        except:
            pass
        os.remove('temp.xyz')
        os.chdir(current_dir)



    def minimise_ts(self,path, atoms: Optional[Atoms] = None ):
        current_dir = os.getcwd()
        os.chdir(path)
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
        if atoms is None:
            atoms = self.atoms
        write('temp.xyz',atoms)
        rmol=atoms.copy()
        pmol=atoms.copy()
        irc_for = atoms.copy()
        irc_rev = atoms.copy()
        system1 = scine_readuct.load_system('temp.xyz', self.method, program='Sparrow',
                                            molecular_charge=0, unrestricted_calculation=self.unrestricted, spin_multiplicity=self.spin_mult, convergence_max_iterations=500)
        systems = {}
        systems['reac'] = system1
        try:
            systems, success = scine_readuct.run_tsopt_task(systems, ['reac'], output= ['ts_opt'], allow_unconverged = True)
            atoms.set_positions(systems['ts_opt'].positions * ANG_PER_BOHR)
            systems, success = scine_readuct.run_irc_task(systems, ['ts_opt'], output=['forward','reverse'],
                                                            allow_unconverged=True, convergence_max_iterations=500)
            rmol.set_positions(systems['forward'].positions * ANG_PER_BOHR)
            pmol.set_positions(systems['reverse'].positions * ANG_PER_BOHR)
            irc_for = read('forward/forward.irc.forward.trj.xyz', ':')
            irc_rev = read('reverse/reverse.irc.backward.trj.xyz', ':')
        except:
            pass
        os.remove('temp.xyz')
        os.chdir(current_dir)
        return atoms, rmol, pmol, irc_for, irc_rev

    def minimise_bspline(self,path, reac, prod ):
        current_dir = os.getcwd()
        os.chdir(path)
        sym = reac.get_chemical_symbols()
        is_O = len(sym) == 1 and sym[0] == 'O'
        is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] == 'O'
        s = sum(reac.get_atomic_numbers())
        if s % 2 != 0:
            self.spin_mult = 2
            self.unrestricted = True
        elif is_O or is_OO:
            self.spin_mult = 3
            self.unrestricted = False
        else:
            self.spin_mult = 1
            self.unrestricted = False

        write('reac.xyz',reac)
        write('prod.xyz', prod)

        system1 = scine_readuct.load_system('reac.xyz', self.method, program='Sparrow',
                                            molecular_charge=0, unrestricted_calculation=self.unrestricted, spin_multiplicity=self.spin_mult)
        system2 = scine_readuct.load_system('prod.xyz', self.method, program='Sparrow',
                                            molecular_charge=0, unrestricted_calculation=self.unrestricted,
                                            spin_multiplicity=self.spin_mult)

        systems = {}
        systems['reac'] = system1
        systems['prod'] = system2
        spline_traj = []
        try:
            systems, success = scine_readuct.run_bspline_task(systems, ['reac','prod'], output = ['spline'], num_structures = 50)
            spline_traj = read('spline/spline_optimized.xyz', index=':')
        except:
            pass
        os.remove('reac.xyz')
        os.remove('prod.xyz')
        os.chdir(current_dir)
        return spline_traj
