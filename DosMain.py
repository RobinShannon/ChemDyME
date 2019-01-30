from ase import Atoms
import Trajectory
from ase.optimize import BFGS
import Tools as tl
import os
from ase.io import write, read


def run(gl):
    mol = read("test.pdb")
    min = BFGS(mol)
    try:
        min.run(fmax=0.1, steps=150)
    except:
        min.run(fmax=0.1, steps=50)
    t = Trajectory.Trajectory(mol,gl,os.getcwd(),0,False)
    energy = mol.get_potential_energy()
    t.runBXDEconvergence(1,energy-0.001,False,mol,"fixed",20, 20, 5)
