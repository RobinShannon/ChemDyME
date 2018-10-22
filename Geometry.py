import openbabel, pybel
import numpy as np
import Tools as tl
import subprocess
from ase import Atoms
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase import constraints
from ase.neb import NEB
from ase.optimize import MDMin
from ase.neb import NEBtools

from ase.calculators.hotbit import Hotbit


# Class to store stable species and perform geometry optimisations and energy refinements
class Reaction:

    # Initialize reaction type with only the reactant molecule and give dummy values to
    def __init__(self, cartesians, species):
        self.Reacfreqs = np.zeroes((cartesians.size * 3) -  6 )
        self.Reac=Atoms(symbols=species, positions=self.geom)
        self.ReacName = tl.getSMILES(self.mol)
        self.TS = self.Reac
        self.TSFreqs = np.zeroes((cartesians.size * 3) -7)
        self.Prodfreqs = np.zeroes((cartesians.size * 3) -  6 )
        self.Prod= self.Reac
        self.ProdName = tl.getSMILES(self.mol)
        self.biProd = False
        self.biReact = False
        self.Reac.set_calculator(hotbit)
        self.Prod.set_calculator(hotbit)
        self.TS.set_calculator(hotbit)

    def optReac(self, cart):
        self.Reac.set_positions(cart)
        min = BFGS(self.Reac)
        min.run(fmax=0.05)
        vib = Vibrations(self.Reac)
        vib.run()
        self.Reacfreqs = vib.get_frequencies()

    def optProd(self, cart):
        self.biReact = False
        self.ProdName = tl.getSMILES(cart)
        FullName = self.ProdName.split('____')
        if FullName.size() == 0:
            self.biReact = True
            self.CombProd = tl.getMolFromSmile(self.ProdName)
            self.ProdName = FullName[0]
            self.biProdName = FullName[1]
            self.Prod = tl.getMolFromSmile(self.ProdName)
            self.biProd = tl.getMolFromSmile(self.biProdName)
        else:
            self.Prod.set_positions(cart)
        min = BFGS(self.Prod)
        min.run(fmax=0.05)
        vib = Vibrations(self.Prod)
        vib.run()
        self.Prodfreqs = vib.get_frequencies()
        if self.biReact == True:
            min = BFGS(self.biProd)
            min.run(fmax=0.05)
            vib = Vibrations(self.biProd)
            vib.run()
            self.biProdfreqs = vib.get_frequencies()

    def optTS(self, cart, i1, i2):
        if self.biReact == True:
            c = constraints.FixBondLength(i1, i2)
            self.CombReac.set_constraint(c)
            rmin = BFGS(self.CombReac)
            rmin.run(fmax=0.05)
        else:
            self.CombReac = self.Reac
        if self.biProd == True:
            c = constraints.FixBondLength(i1, i2)
            self.CombProd.set_constraint(c)
            rmin = BFGS(self.CombProd)
            rmin.run(fmax=0.05)
        else:
            self.CombProd = self.Reac
        # Read initial and final states:
        initial = self.CombReac
        final = self.CombProd
        # Make a band consisting of 5 images:
        images = [initial]
        images += [initial.copy() for i in range(3)]
        images += [final]
        neb = NEB(images)
        # Interpolate linearly the potisions of the three middle images:
        neb.interpolate()
        # Set calculators:
        for image in images[1:4]:
            image.set_calculator(hotbit(...))
        # Optimize:
        optimizer = MDMin(neb, trajectory='A2B.traj')
        optimizer.run(fmax=0.04)

        nebtools = NEBtools(images)
        self.TS = nebtools.get_barrier(fit=True, raw=True)

        vib = Vibrations(self.biProd)
        vib.run()
        self.TSFreqs = vib.get_frequencies()




