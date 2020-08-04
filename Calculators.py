from ase.calculators.nwchem import NWChem
from ase.calculators.gaussian import Gaussian

from ChemDyME.ScineCalculator import SparrowCalculator

def scine(mol,lab,level):
    mol.set_calculator(SparrowCalculator(method = level))
    return mol



def nwchem(mol, lab, level):
    
    level = level.split('_')
    
    Trip = False
    if len(level) == 4:
        Trip = True
        bas = level[1]
        lev = level[0]
        Igrid = level[2]
    if len(level) == 3:
        bas = level[1]
        lev = level[0]
        Igrid = level[2]
    elif len(level) ==2:
        bas = level[1]
        lev = level[0]
        Igrid = 'coarse'
    else:
        bas = basis
        lev = level
        Igrid = 'coarse'

    sym = mol.get_chemical_symbols()
    is_O = len(sym) == 1 and sym[0] == 'O'
    is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] =='O'

    if is_O or is_OO or Trip == True:
            mol.set_calculator(NWChem(label=lab,
                maxiter=60,
                geometry='noautoz noautosym nocenter',
                xc=lev,
                basis=bas,
                vectors= 'input mo.movecs output mo.movecs',
                grid = Igrid,
                task='energy ignore \nset dft:converged true \ntask dft gradient',
                mult=3))
    else:
        atom_num = mol.get_atomic_numbers()
        s = sum(atom_num)
        if s % 2 == 0:
            mol.set_calculator(NWChem(label=lab,
                maxiter=60,
                geometry='noautoz noautosym nocenter',
                xc=lev,
                basis=bas,
                vectors= 'input mo.movecs output mo.movecs',
                grid = Igrid,
                CONVERGENCE = 'ncysh 5 ncydp 5 damp 50',
                task='energy ignore \nset dft:converged true \ntask dft gradient',
                mult=1))
        else:
            mol.set_calculator(NWChem(label=lab,
                maxiter=60,
                geometry='noautoz noautosym nocenter',
                xc=lev,
                basis=bas,
                vectors= 'input mo.movecs output mo.movecs',
                grid = Igrid,
                CONVERGENCE = 'ncysh 5 ncydp 5 damp 50',
                mult=2,
                task='energy ignore \nset dft:converged true \ntask dft gradient'
            ))
    return mol

def nwchemTS(mol, lab, level):
    
    level = level.split('_')
    
    Trip = False
    if len(level) == 4:
        Trip = True
        bas = level[1]
        lev = level[0]
        Igrid = level[2]
    if len(level) == 3:
        bas = level[1]
        lev = level[0]
        Igrid = level[2]
    elif len(level) ==2:
        bas = level[1]
        lev = level[0]
        Igrid = 'coarse'
    else:
        bas = basis
        lev = level
        Igrid = 'coarse'
    
    sym = mol.get_chemical_symbols()
    is_O = len(sym) == 1 and sym[0] == 'O'
    is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] =='O'
    
    if is_O or is_OO or Trip == True:
        mol.set_calculator(NWChem(label=lab,
                                  maxiter=60,
                                  geometry='noautoz noautosym nocenter',
                                  xc=lev,
                                  basis=bas,
                                  grid = Igrid,
                                  raw = 'driver \nxyz ts \nMAXITER 50 \n end\n',
                                  task='saddle',
                                  mult=3))
    else:
        atom_num = mol.get_atomic_numbers()
        s = sum(atom_num)
        if s % 2 == 0:
            mol.set_calculator(NWChem(label=lab,
                                      maxiter=60,
                                      geometry='noautoz noautosym nocenter',
                                      xc=lev,
                                      basis=bas,
                                      grid = Igrid,
                                      raw = 'driver \nxyz ts \nMAXITER 50 \n end\n',
                                      CONVERGENCE = 'ncysh 5 ncydp 5 damp 50',
                                      task=' saddle',
                                      mult=1))
        else:
            mol.set_calculator(NWChem(label=lab,
                                      maxiter=60,
                                      geometry='noautoz noautosym nocenter',
                                      xc=lev,
                                      basis=bas,
                                      grid = Igrid,
                                      CONVERGENCE = 'ncysh 5 ncydp 5 damp 50',
                                      mult=2,
                                      raw = 'driver \nxyz opt \nMAXITER 50 \n end\n',
                                      task=' saddle'
                                      ))
    return mol

def nwchem2(mol, lab, level):
    
    level = level.split('_')
    
    
    Trip = False
    if len(level) == 4:
        Trip = True
        bas = level[1]
        lev = level[0]
        Igrid = level[2]
    elif len(level) == 3:
        bas = level[1]
        lev = level[0]
        Igrid = level[2]
    elif len(level) ==2:
        bas = level[1]
        lev = level[0]
        Igrid = 'coarse'
    else:
        bas = basis
        lev = level
        Igrid = 'coarse'

    sym = mol.get_chemical_symbols()
    is_O = len(sym) == 1 and sym[0] == 'O'
    is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] =='O'

    if is_O or is_OO or Trip ==True:
            mol.set_calculator(NWChem(label=lab,
                maxiter=60,
                geometry='noautoz noautosym nocenter',
                xc=lev,
                basis=bas,
                vectors= 'output mo.movecs',
                grid = Igrid,
                task='energy ignore \nset dft:converged true \ntask dft gradient',
                mult=3))
    else:
        atom_num = mol.get_atomic_numbers()
        s = sum(atom_num)
        if s % 2 == 0:
            mol.set_calculator(NWChem(label=lab,
                maxiter=60,
                geometry='noautoz noautosym nocenter',
                xc=lev,
                basis=bas,
                vectors= 'output mo.movecs',
                grid = Igrid,
                task='energy ignore \nset dft:converged true \ntask dft gradient',
                mult=1))
        else:
            mol.set_calculator(NWChem(label=lab,
                maxiter=60,
                geometry='noautoz noautosym nocenter',
                xc=lev,
                basis=bas,
                vectors= 'output mo.movecs',
                grid = Igrid,
                CONVERGENCE = 'ncysh 5 ncydp 5 damp 50',
                mult=2,
                task='energy ignore \nset dft:converged true \ntask dft gradient'
            ))
    return mol

def nwchemOpt(mol, lab, level):
    
    level = level.split('_')
    
    Trip = False
    if len(level) == 4:
        Trip = True
        bas = level[1]
        lev = level[0]
        Igrid = level[2]
    if len(level) == 3:
        bas = level[1]
        lev = level[0]
        Igrid = level[2]
    elif len(level) ==2:
        bas = level[1]
        lev = level[0]
        Igrid = 'coarse'
    else:
        bas = basis
        lev = level
        Igrid = 'coarse'
    
    sym = mol.get_chemical_symbols()
    is_O = len(sym) == 1 and sym[0] == 'O'
    is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] =='O'
    
    if is_O or is_OO or Trip == True:
        mol.set_calculator(NWChem(label=lab,
                                  maxiter=60,
                                  geometry='noautoz noautosym nocenter',
                                  xc=lev,
                                  basis=bas,
                                  grid = Igrid,
                                  raw = 'driver \nxyz opt \nMAXITER 40 \n end\n',
                                  task='optimize',
                                  mult=3))
    else:
        atom_num = mol.get_atomic_numbers()
        s = sum(atom_num)
        if s % 2 == 0:
            mol.set_calculator(NWChem(label=lab,
                                      maxiter=60,
                                      geometry='noautoz noautosym nocenter',
                                      xc=lev,
                                      basis=bas,
                                      grid = Igrid,
                                      raw = 'driver \nxyz opt \nMAXITER 40 \n end\n',
                                      CONVERGENCE = 'ncysh 5 ncydp 5 damp 50',
                                      task=' optimize',
                                      mult=1))
        else:
            mol.set_calculator(NWChem(label=lab,
                                      maxiter=60,
                                      geometry='noautoz noautosym nocenter',
                                      xc=lev,
                                      basis=bas,
                                      grid = Igrid,
                                      CONVERGENCE = 'ncysh 5 ncydp 5 damp 50',
                                      mult=2,
                                      raw = 'driver \nxyz opt \nMAXITER 40 \n end\n',
                                      task=' optimize'
                                      ))
    return mol

def nwchemFreq(mol, lab, level):
    
    level = level.split('_')
    
    Trip = False
    if len(level) == 4:
        Trip = True
        bas = level[1]
        lev = level[0]
        Igrid = level[2]
    if len(level) == 3:
        bas = level[1]
        lev = level[0]
        Igrid = level[2]
    elif len(level) ==2:
        bas = level[1]
        lev = level[0]
        Igrid = 'coarse'
    else:
        bas = basis
        lev = level
        Igrid = 'coarse'
    
    sym = mol.get_chemical_symbols()
    is_O = len(sym) == 1 and sym[0] == 'O'
    is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] =='O'
    
    if is_O or is_OO or Trip == True:
        mol.set_calculator(NWChem(label=lab,
                                  maxiter=60,
                                  geometry='noautoz noautosym nocenter',
                                  xc=lev,
                                  basis=bas,
                                  grid = Igrid,
                                  raw = 'freq \nanimate \n end\n',
                                  task='freq',
                                  mult=3))
    else:
        atom_num = mol.get_atomic_numbers()
        s = sum(atom_num)
        if s % 2 == 0:
            mol.set_calculator(NWChem(label=lab,
                                      maxiter=60,
                                      geometry='noautoz noautosym nocenter',
                                      xc=lev,
                                      basis=bas,
                                      grid = Igrid,
                                      raw = 'freq \nanimate \n end\n',
                                      CONVERGENCE = 'ncysh 5 ncydp 5 damp 50',
                                      task=' freq',
                                      mult=1))
        else:
            mol.set_calculator(NWChem(label=lab,
                                      maxiter=60,
                                      geometry='noautoz noautosym nocenter',
                                      xc=lev,
                                      basis=bas,
                                      grid = Igrid,
                                      CONVERGENCE = 'ncysh 5 ncydp 5 damp 50',
                                      mult=2,
                                      raw = 'freq \nanimate \n end\n',
                                      task=' freq'
                                      ))
    return mol

def gaus(mol, lab, level):

    level = level.split('_')

    Trip = False

    sym = mol.get_chemical_symbols()
    is_O = len(sym) == 1 and sym[0] == 'O'
    is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] =='O'

    if is_O or is_OO or Trip == True:
            mol.set_calculator(Gaussian(label=lab,
                maxiter=60,
                geometry='noautoz noautosym nocenter',
                xc='b3lyp',
                basis='6-31+G',
                task='energy ignore \nset dft:converged true \ntask dft gradient',
                mult=3))
    else:
        atom_num = mol.get_atomic_numbers()
        s = sum(atom_num)
        if s % 2 == 0:
            mol.set_calculator(Gaussian(label=lab,
                maxiter=60,
                geometry='noautoz noautosym nocenter',
                xc='b3lyp',
                basis='6-31+G',
                task='energy ignore \nset dft:converged true \ntask dft gradient',
                mult=1))
        else:
            mol.set_calculator(Gaussian(label=lab,
                maxiter=60,
                geometry='noautoz noautosym nocenter',
                xc='b3lyp',
                basis='6-31+G',
                mult=2,
                task='energy ignore \nset dft:converged true \ntask dft gradient'
            ))
    return mol

def gaussianTS(mol, lab, level):

    level = level.split('_')

    Trip = False
    if len(level) == 4:
        Trip = True
        bas = level[1]
        lev = level[0]
        Igrid = level[2]
    if len(level) == 3:
        bas = level[1]
        lev = level[0]
        Igrid = level[2]
    elif len(level) ==2:
        bas = level[1]
        lev = level[0]
        Igrid = 'coarse'
    else:
        bas = basis
        lev = level
        Igrid = 'coarse'

    sym = mol.get_chemical_symbols()
    is_O = len(sym) == 1 and sym[0] == 'O'
    is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] =='O'

    if is_O or is_OO or Trip == True:
        mol.set_calculator(NWChem(label=lab,
                                  maxiter=60,
                                  geometry='noautoz noautosym nocenter',
                                  xc=lev,
                                  basis=bas,
                                  grid = Igrid,
                                  raw = 'driver \nxyz ts \nMAXITER 50 \n end\n',
                                  task='saddle',
                                  mult=3))
    else:
        atom_num = mol.get_atomic_numbers()
        s = sum(atom_num)
        if s % 2 == 0:
            mol.set_calculator(NWChem(label=lab,
                                      maxiter=60,
                                      geometry='noautoz noautosym nocenter',
                                      xc=lev,
                                      basis=bas,
                                      grid = Igrid,
                                      raw = 'driver \nxyz ts \nMAXITER 50 \n end\n',
                                      CONVERGENCE = 'ncysh 5 ncydp 5 damp 50',
                                      task=' saddle',
                                      mult=1))
        else:
            mol.set_calculator(NWChem(label=lab,
                                      maxiter=60,
                                      geometry='noautoz noautosym nocenter',
                                      xc=lev,
                                      basis=bas,
                                      grid = Igrid,
                                      CONVERGENCE = 'ncysh 5 ncydp 5 damp 50',
                                      mult=2,
                                      raw = 'driver \nxyz opt \nMAXITER 50 \n end\n',
                                      task=' saddle'
                                      ))
    return mol

def gaussian2(mol, lab, level):

    level = level.split('_')


    Trip = False
    if len(level) == 4:
        Trip = True
        bas = level[1]
        lev = level[0]
        Igrid = level[2]
    if len(level) == 3:
        bas = level[1]
        lev = level[0]
        Igrid = level[2]
    elif len(level) ==2:
        bas = level[1]
        lev = level[0]
        Igrid = 'coarse'
    else:
        bas = basis
        lev = level
        Igrid = 'coarse'

    sym = mol.get_chemical_symbols()
    is_O = len(sym) == 1 and sym[0] == 'O'
    is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] =='O'

    if is_O or is_OO or Trip ==True:
            mol.set_calculator(NWChem(label=lab,
                maxiter=60,
                geometry='noautoz noautosym nocenter',
                xc=lev,
                basis=bas,
                vectors= 'output mo.movecs',
                grid = Igrid,
                task='energy ignore \nset dft:converged true \ntask dft gradient',
                mult=3))
    else:
        atom_num = mol.get_atomic_numbers()
        s = sum(atom_num)
        if s % 2 == 0:
            mol.set_calculator(NWChem(label=lab,
                maxiter=60,
                geometry='noautoz noautosym nocenter',
                xc=lev,
                basis=bas,
                vectors= 'output mo.movecs',
                grid = Igrid,
                task='energy ignore \nset dft:converged true \ntask dft gradient',
                mult=1))
        else:
            mol.set_calculator(NWChem(label=lab,
                maxiter=60,
                geometry='noautoz noautosym nocenter',
                xc=lev,
                basis=bas,
                vectors= 'output mo.movecs',
                grid = Igrid,
                CONVERGENCE = 'ncysh 5 ncydp 5 damp 50',
                mult=2,
                task='energy ignore \nset dft:converged true \ntask dft gradient'
            ))
    return mol

def gaussian(mol, lab, level):
    level = level.split('_')
    lev = level[0]
    bas = level[1]
    sym = mol.get_chemical_symbols()
    is_O = len(sym) == 1 and sym[0] == 'O'
    is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] =='O'
    if is_OO or is_O:
        m = 3
    else:
        atom_num = mol.get_atomic_numbers()
        s = sum(atom_num)
        if s % 2 == 0:
            m = 1
        else:
            m = 2

    mol.calc = Gaussian(mem='1GB',
                          chk='MyJob.chk',
                          save=None,
                          method=lev,
                          basis=bas,
                          mult=m,
                          scf='qc')

    return mol

def gaussianFreq(mol, lab, level):

    level = level.split('_')

    Trip = False
    if len(level) == 4:
        Trip = True
        bas = level[1]
        lev = level[0]
        Igrid = level[2]
    if len(level) == 3:
        bas = level[1]
        lev = level[0]
        Igrid = level[2]
    elif len(level) ==2:
        bas = level[1]
        lev = level[0]
        Igrid = 'coarse'
    else:
        bas = basis
        lev = level
        Igrid = 'coarse'

    sym = mol.get_chemical_symbols()
    is_O = len(sym) == 1 and sym[0] == 'O'
    is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] =='O'

    if is_O or is_OO or Trip == True:
        mol.set_calculator(NWChem(label=lab,
                                  maxiter=60,
                                  geometry='noautoz noautosym nocenter',
                                  xc=lev,
                                  basis=bas,
                                  grid = Igrid,
                                  raw = 'freq \nanimate \n end\n',
                                  task='freq',
                                  mult=3))
    else:
        atom_num = mol.get_atomic_numbers()
        s = sum(atom_num)
        if s % 2 == 0:
            mol.set_calculator(NWChem(label=lab,
                                      maxiter=60,
                                      geometry='noautoz noautosym nocenter',
                                      xc=lev,
                                      basis=bas,
                                      grid = Igrid,
                                      raw = 'freq \nanimate \n end\n',
                                      CONVERGENCE = 'ncysh 5 ncydp 5 damp 50',
                                      task=' freq',
                                      mult=1))
        else:
            mol.set_calculator(NWChem(label=lab,
                                      maxiter=60,
                                      geometry='noautoz noautosym nocenter',
                                      xc=lev,
                                      basis=bas,
                                      grid = Igrid,
                                      CONVERGENCE = 'ncysh 5 ncydp 5 damp 50',
                                      mult=2,
                                      raw = 'freq \nanimate \n end\n',
                                      task=' freq'
                                      ))
    return mol