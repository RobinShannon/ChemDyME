import os
import copy
from collections.abc import Iterable
from shutil import which
from typing import Dict, Optional
import openbabel, pybel
import re
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator, EnvironmentError
from pathlib import Path

EV_PER_HARTREE = 27.2114

class Molpro(FileIOCalculator):
    implemented_properties = ['energy']
    command = 'molpro -d /nobackup/chmrsh/scratch PREFIX.inp'
    discard_results_on_any_change = True

    def __init__(self, *args, label='Molpro', **kwargs):
        FileIOCalculator.__init__(self, *args, label=label, **kwargs)
        if int(self.parameters['multiplicity']) ==2:
            self.parameters['method'] = 'u'+str(self.parameters['method'])


    def calculate(self, *args, **kwargs):
        try:
            FileIOCalculator.calculate(self, *args, **kwargs)
        except:
            pass

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        at = atoms.get_atomic_numbers()
        cart = atoms.get_positions()
        BABmol = openbabel.OBMol()
        for i in range(0, at.size):
            a = BABmol.NewAtom()
            a.SetAtomicNum(int(at[i]))
            a.SetVector(float(cart[i, 0]), float(cart[i, 1]), float(cart[i, 2]))

        # Assign bonds and fill out angles and torsions
        BABmol.ConnectTheDots()
        BABmol.FindAngles()
        BABmol.FindTorsions()
        BABmol.PerceiveBondOrders()

        # Create converter object to convert from XYZ to smiles
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "mp")
        name = (obConversion.WriteString(BABmol))
        name = name.replace("!memory,INSERT MEMORY HERE", "!memory,2M")
        name = name.replace("!hf", str(self.parameters['method']))
        name = name.replace("!INSERT QM METHODS HERE", "hf")
        name = name.replace("!basis,INSERT BASIS SET HERE", "basis," + str(self.parameters['basis']))
        f = open("Molpro.inp", "w")
        f.write(str(name))
        f.close()


    def read_results(self):
        f = open("Molpro.out", "r")
        energy_hartree = 1000
        for line in f:
            if re.search("!CCSD\(T\)-F12b total energy", line):
                energy_hartree = float(line.split()[3])
            if re.search("!RHF-UCCSD\(T\)-F12b energy", line):
                energy_hartree = float(line.split()[2])
        self.results['energy'] = energy_hartree * EV_PER_HARTREE
        f.close()
        if energy_hartree == 1000:
            print('molpro error')
            i = 0
            while Path('molerror'+str(i)+'.out').exists():
                i += 1
            os.rename('molpro.out', 'molerror.out')
        os.remove("Molpro.out")
        os.remove("Molpro.xml")

    # Method(s) defined in the old calculator, added here for
    # backwards compatibility
    def clean(self):
        for suffix in ['.com', '.chk', '.log']:
            try:
                os.remove(os.path.join(self.directory, self.label + suffix))
            except OSError:
                pass

