# Copyright (c) Intangible Realities Lab, University Of Bristol. All rights reserved.
# Licensed under the GPL. See License.txt in the project root for license information.
from typing import Optional, Collection
import os
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from scine_sparrow import Calculation
import scine_sparrow
from ase.io import write, read
import scine_readuct
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

    def __init__(self, atoms: Optional[Atoms] = None, method='PM6', triplet=False, **kwargs):
        super().__init__(**kwargs)
        self.atoms = atoms
        self.method = method
        self.triplet = triplet
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
            elif is_O or is_OO or self.triplet:
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

    def minimise_stable(self,path = os.getcwd(), atoms: Optional[Atoms] = None):
        current_dir = os.getcwd()
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        sym = atoms.get_chemical_symbols()
        if len(sym) == 1:
            return
        is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] == 'O'
        s = sum(atoms.get_atomic_numbers())
        if s % 2 != 0:
            self.spin_mult = 2
            self.unrestricted = True
        elif is_OO:
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

    def close(self):
        self.calc = None

    def minimise_ts(self,path, atoms: Optional[Atoms] = None ):
        current_dir = os.getcwd()
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        sym = atoms.get_chemical_symbols()
        is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] == 'O'
        s = sum(atoms.get_atomic_numbers())
        if s % 2 != 0:
            self.spin_mult = 2
            self.unrestricted = True
        elif is_OO:
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
                                            molecular_charge=0, unrestricted_calculation=self.unrestricted, spin_multiplicity=self.spin_mult)
        systems = {}
        systems['reac'] = system1
        try:
            systems, success = scine_readuct.run_tsopt_task(systems, ['reac'], output= ['ts_opt'], optimizer='ef',  allow_unconverged = False)
            atoms.set_positions(systems['ts_opt'].positions * ANG_PER_BOHR)
            systems, success = scine_readuct.run_irc_task(systems, ['ts_opt'], output=['forward','reverse'], convergence_max_iterations =5000, allow_unconverged=True)
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
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        sym = reac.get_chemical_symbols()
        is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] == 'O'
        s = sum(reac.get_atomic_numbers())
        if s % 2 != 0:
            self.spin_mult = 2
            self.unrestricted = True
        elif is_OO:
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
        try:
            systems, success = scine_readuct.run_bspline_task(systems, ['reac','prod'], output = ['spline'],  num_integration_points=20, num_control_points=10,  num_structures = 60)
        except:
            try:
                systems, success = scine_readuct.run_bspline_task(systems, ['reac','prod'], output = ['spline'],  num_integration_points=10, num_control_points=5,  num_structures = 60)
            except:
                pass
        try:
            spline_traj = read('spline/spline_optimized.xyz', index=':')
        except:
            print('spline_failed')
            spline_traj = read('spline/spline_interpolated.xyz', index=':')
        os.remove('spline/spline_optimized.xyz')
        os.remove('spline/spline_interpolated.xyz')
        os.remove('reac.xyz')
        os.remove('prod.xyz')
        try:
            os.chdir(current_dir)
        except:
            pass
        return spline_traj
