import Tools as tl
import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.neb import NEB
import os
from ase.optimize import FIRE
from ase.neb import NEBtools
from ase.io import write, read
from ase.vibrations import Vibrations

# Class to store stable species and perform geometry optimisations and energy refinements
class Reaction:

    # Initialize reaction type with only the reactant molecule and give dummy values to
    def __init__(self, reac, ts, prod):
        self.reac = reac
        self.ts = ts
        self.prod = prod
        self.activationEnergy = 0
        self.events_forward = 0
        self.events_reverse = 0
