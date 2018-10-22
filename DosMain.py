from ase import Atoms
import Trajectory
from ase.optimize import BFGS
import Tools as tl
import os
from ase.io import write, read
from ase.calculators.OpenMMCalc import OpenMMCalculator

def run(gl):
    mol = read("test.pdb")
    mol.set_calculator(OpenMMCalculator(input="test.pdb"))
    min = BFGS(mol)
    try:
        min.run(fmax=0.1, steps=150)
    except:
        min.run(fmax=0.1, steps=50)
    t = Trajectory.Trajectory(mol,gl,os.getcwd(),0,False)
    energy = mol.get_potential_energy()
    t.runBXDEconvergence(1,energy-0.001,False,mol,"adaptive",20, 20, 5)