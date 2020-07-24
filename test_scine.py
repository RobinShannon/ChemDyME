import scine_sparrow
import scine_readuct
import scine_utilities
from ase import Atoms
from ase.io import write
import openbabel, pybel
import numpy as np

def getMolFromSmile(smile):

    # Create OBabel object from smiles
    smile = smile.replace("____", ".")
    mol = pybel.readstring('smi' , smile)
    mol.addh()
    mol.make3D()
    dim = len(mol.atoms)
    a = np.zeros(dim)
    b = np.zeros((dim , 3))
    i = 0
    for Atom in mol:
        a[i]= Atom.atomicnum
        b[i] = Atom.coords
        i += 1

    aseMol = Atoms(symbols=a, positions=b)
    return aseMol

mol = getMolFromSmile('C[C][CH]O')
mol.write('temp.xyz')
mol2 = getMolFromSmile('O[CH]C#C')
mol2.write('temp2.xyz')

system1 = scine_readuct.load_system('temp.xyz', 'PM6', program='Sparrow',
                                    molecular_charge=0, spin_multiplicity=2)
system1 = scine_readuct.load_system('temp2.xyz', 'PM6', program='Sparrow',
                                    molecular_charge=0, spin_multiplicity=2)
systems = {}
systems['reac'] = system1
systems['prod'] = system1
systems, success = scine_readuct.run_bspline_task(systems, ['reac','prod'], output=['spline'])
if success:
    print(systems['spline'])