from ase import Atoms
import numpy as np
import openbabel, pybel
import ChemDyME.Calculators as calc
import os
import re
from ase.io import read
import shutil
from ase.optimize import BFGS
try:
    from OpenMMCalc import OpenMMCalculator
except:
    pass

# Function takes a molecule in ASE format, converts it into an OBmol and then returns a SMILES string as a name
def getSMILES(mol, opt, partialOpt = False):
    if opt:
        min = BFGS(mol)
        if partialOpt:
            try:
                min.run(fmax=0.1, steps=15)
            except:
                min.run(fmax=0.1, steps=1)
        elif opt:
            try:
                min.run(fmax=0.1, steps=200)
            except:
                min.run(fmax=0.1, steps=1)

    # Get list of atomic numbers and cartesian coords from ASEmol
    atoms = mol.get_atomic_numbers()
    cart = mol.get_positions()

    # Create open babel molecule BABMol
    BABmol = openbabel.OBMol()
    for i in range(0,atoms.size):
        a = BABmol.NewAtom()
        a.SetAtomicNum(int(atoms[i]))
        a.SetVector(float(cart[i,0]), float(cart[i,1]), float(cart[i,2]))

    # Assign bonds and fill out angles and torsions
    BABmol.ConnectTheDots()
    BABmol.FindAngles()
    BABmol.FindTorsions()
    BABmol.PerceiveBondOrders()

    #Create converter object to convert from XYZ to smiles
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "can")
    name = (obConversion.WriteString(BABmol))

    # Convert "." to "____" to clearly differentiate between species
    name = name.replace('.', '____')
    name = name.replace('/', '')
    # These options make trans / cis isomers indistinguishable and ignore chirality
    name = name.replace('\\', '')
    name = name.replace('@', '')
    name = name.strip('\n\t')
    return name

# Function takes a molecule in ASE format, converts it into an OBmol and then returns a CML stream
def getCML(ASEmol, name):

    # Get list of atomic numbers and cartesian coords from ASEmol
    atoms = ASEmol.get_atomic_numbers()
    cart = ASEmol.get_positions()

    # Create open babel molecule BABMol
    BABmol = openbabel.OBMol()
    for i in range(0,atoms.size):
        a = BABmol.NewAtom()
        a.SetAtomicNum(int(atoms[i]))
        a.SetVector(float(cart[i,0]), float(cart[i,1]), float(cart[i,2]))

    # Assign bonds and fill out angles and torsions
    BABmol.ConnectTheDots()
    BABmol.FindAngles()
    BABmol.FindTorsions()
    BABmol.PerceiveBondOrders()
    BABmol.SetTitle('xxxx')

    #Create converter object to convert from XYZ to cml
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "cml")
    cml = (obConversion.WriteString(BABmol))
    cml = cml.replace('xxxx', name)

    return cml

def getSpinMult(mol, name, trip = False):
    #Babel incorrectly guessing spin multiplicity from trajectory snapshots
    #If molecule is O2 or O assume triplet rather than singlet
    if name == '[O][O]' or name == '[O]':
        spinMult = 3
    else:
    # else count electrons to determine whether singlet or doublet
        atom_num = mol.get_atomic_numbers()
        s = sum(atom_num)
        if s % 2 == 0 and trip:
            spinMult = 3
        elif s % 2 == 0:
            spinMult = 1
        else:
            spinMult = 2
    return spinMult

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

# Function gets distance of current geometry from either reactant or product
def getDistAlongS(ref, mol):
    return np.linalg.norm(mol-ref)

def prettyPrint(x, path):
    i=0
    f = open(path, 'w')
    for line in x.toprettyxml().split('\n'):
        if not line.strip() == '':
            if i == -1:
                words = line.split()
                del words[2]
                del words[3]
                del words[4]
                words.append('>')
                f.write(' '.join(words))
            elif i >= 0:
                f.write(line + '\n')
                if i == 0:
                    f.write("<?xml-stylesheet type='text/xsl' href='../../mesmer2.xsl' media='other'?>\n")
                    f.write("<?xml-stylesheet type='text/xsl' href='../../mesmer1.xsl' media='screen'?>\n")
            i += 1

def printTraj(file, Mol):

    # Get symbols and positions of current molecule
    symbols = Mol.get_chemical_symbols()
    size = len(symbols)
    xyz = Mol.get_positions()

    # write xyz format to already open file
    file.write((str(size) + ' \n'))
    file.write( '\n')
    for i in range(0,size):
        file.write(( str(symbols[i]) + ' \t' + str(xyz[i][0]) + '\t' + str(xyz[i][1]) + '\t' + str(xyz[i][2]) + '\n'))

def getVibString(viblist, bi, TS):
    zpe = 0
    if bi:
        max = 4
    else:
        max = 5

    if TS:
        max += 1

    vibs = []

    for i in range(0, len(viblist)):
        if i > max:
            if viblist[i].real != 0:
                vibs.append(viblist[i].real)
                zpe += viblist[i].real
            else:
                vibs.append(100.0)
                zpe += 100.0
    zpe *= 0.00012
    zpe /= 2
    return vibs,zpe

def getImageFreq(viblist):
    Imag = viblist[0].imag
    return Imag

def getOptGeom(path, opath, mol, method):
    if method == 'nwchem':
        outmol = getNWOptGeom(path, opath, mol)
        return outmol
    elif method == 'mopac':
        outmol = getMopOptGeom(path, opath, mol)
        return outmol
    elif method =='scine':
        mol._calc.minimise_stable(path, mol)
        outmol = mol.copy()
        return outmol
    else:
        print('unknown opt method')
        return mol

def getNWOptGeom(path, opath, mol):
    pPath = os.getcwd()
    os.chdir(path)
    with open('min.xyz','wb') as wfd:
        for file in os.listdir():
            if (file.endswith(".xyz") and file != 'min.xyz'):
                with open(file,'rb') as fd:
                    shutil.copyfileobj(fd, wfd, 1024*1024*10)
                    mol = read(file)
                    os.remove(file)
    if opath != 'none':
        shutil.copyfile('min.xyz', opath + '.xyz')
        shutil.copyfile('calc.out', opath + '.out')
    os.remove('min.xyz')
    os.chdir(pPath)
    return mol

def getMopOptGeom(path, opath, mol):
    pPath = os.getcwd()
    os.chdir(path)
    size = 0
    j= 0
    with open('calc.arc','r') as wfd:
        for num, line in enumerate(wfd, 1):
            if('Empirical Formula' in line):
                words = line.split()
                size = [int(word) for word in words if word.isdigit()]
                size = size[0]
    species = np.zeros(size, dtype='str')
    cartesians = np.zeros((size,3), dtype='float')
    with open('calc.arc','r') as wfd:
         point = 100000
         for num2, line2 in enumerate(wfd, 1):
            if('FINAL GEOMETRY OBTAINED' in line2):
                point = num2
            if(num2 > point + 3 and num2 < point + 4 + size):
                words = line2.split()
                species[j] = words[0]
                cartesians[j][0] = float(words[1])
                cartesians[j][1] = float(words[3])
                cartesians[j][2] = float(words[5])
                j += 1
    os.chdir(pPath)
    mol = Atoms(symbols=species, positions = cartesians)
    return mol

def TSCalcfinished(path):
    os.chdir(path)
    with open('TSDirect.out','wb') as wfd:
        for num, line in enumerate(wfd, 1):
            if('dft saddle failed' in line):
                return False
    return True

def getFreqs(path,method):
    freqs = []
    zpe = 0
    if method == 'nwchem':
        freqs,zpe = getNWFreqs(path)
        return freqs,zpe
    elif method == 'mopac':
        freqs,zpe = getMopFreqs(path)
        return freqs,zpe
    if len(freqs) == 0:
            raise Exception('freq returned empty list')
    elif method == 'scine':
        freqs,zpe = mol
    else:
        print('unknown opt method')
        return freqs,zpe

def checkFreqNumber(mol,freqs, zpe):
    numAtom = len(mol.get_atomic_numbers())
    numFreq = len(freqs)
    zpe /=0.00012
    if numAtom > 3:
        while len(freqs) < (3*numAtom)-6:
            freqs.append(freqs[0])
            zpe += (freqs[0]/2)
    zpe *= 0.00012
    return freqs, zpe

def checkTSFreqNumber(mol,freqs, zpe):
    numAtom = len(mol.get_atomic_numbers())
    numFreq = len(freqs)
    zpe /=0.00012
    if numAtom > 3:
        while len(freqs) < (3*numAtom)-7:
            freqs.append(freqs[0])
            zpe += (freqs[0]/2)
    zpe *= 0.00012
    return freqs, zpe

def getTSFreqs(path,opath,method,mol):
    imagFreq = 0.0
    freqs = []
    zpe = 0
    rmol = mol.copy()
    pmol = mol.copy()
    if method == 'nwchem':
        imagFreq,freqs,zpe,rmol,pmol = getNWTSFreqs(path,opath)
        return imagFreq,freqs,zpe,rmol,pmol
    elif method == 'mopac':
        imagFreq,freqs,zpe,rmol,pmol = getMopTSFreqs(path,mol,opath)
        return imagFreq,freqs,zpe,rmol,pmol
    else:
        print('unknown opt method')
        return freqs,zpe

def getNWFreqs(path):
    freqs = []
    zpe = 0
    pPath = os.getcwd()
    os.chdir(path)
    End = False
    lineNum = 100000
    inFile = open("calc.out")
    for num, line in enumerate(inFile, 1):
        if('Projected Derivative Dipole Moments' in line):
            lineNum = num
        if (num > (lineNum + 2) and End == False):
            line = line.split()
            try:
                if float(line[1].strip('-')) > 5.0:
                    freqs.append(float(line[1].strip('-')))
                    zpe += float(line[1].strip('-'))
            except:
                End = True
    for file in os.listdir():
            if (file.endswith(".xyz") and file != 'min.xyz'):
                os.remove(file)
    os.chdir(pPath)
    zpe *= 0.00012
    zpe /= 2
    return freqs,zpe

def getMopTSFreqs(path,mol,opath):
    freqs = []
    imagFreq = 0.0
    zpe = 0
    pPath = os.getcwd()
    os.chdir(path)
    End = False
    lineNum = 100000
    inFile = open("calc.out")
    for num, line in enumerate(inFile, 1):
        if('FREQ.' in line):
            line = line.split()
            freqs.append(float(line[1].strip('-')))
            zpe += float(line[1].strip('-'))
    imagFreq = freqs[0]
    freqs.pop(0)
    os.chdir(pPath)
    zpe *= 0.00012
    zpe /= 2
    rmol = mol.copy()
    rmol = setCalc(rmol, 'calc', 'mopacIRC1', 'none')
    try:
        rmol.get_forces()
    except:
        try:
            rmol = read('calc.xyz')
            shutil.copyfile('calc.xyz', opath + 'IRC1.xyz')
            print('IRC1')
        except:
            pass
    pmol = mol.copy()
    pmol = setCalc(pmol, 'calc', 'mopacIRC2', 'none')
    try:
        pmol.get_forces()
    except:
        try:
            pmol = read('calc.xyz')
            shutil.copyfile('calc.xyz', opath + 'IRC2.xyz')
            print('IRC2')
        except:
            pass
    return imagFreq,freqs,zpe,rmol,pmol

def getMopFreqs(path):
    freqs = []
    zpe = 0
    pPath = os.getcwd()
    os.chdir(path)
    inFile = open("calc.out")
    for line in inFile:
        if('FREQUENCY' in line):
            line = line.split()
            freqs.append(float(line[1].strip('-')))
            zpe += float(line[1].strip('-'))
    os.chdir(pPath)
    zpe *= 0.00012
    zpe /= 2
    return freqs,zpe

def getNWTSFreqs(path, opath):
    pPath = os.getcwd()
    os.chdir(path)
    End = False
    lineNum = 1000000
    imagFreq = 0.0
    freqs = []
    zpe = 0.0
    inFile = open("calc.out")
    for num, line in enumerate(inFile, 1):
        if('Projected Derivative Dipole Moments' in line):
            lineNum = num
        if (num == lineNum + 3):
            line = line.split()
            imagFreq = float(line[1].strip('-'))
        if (num > (lineNum + 3) and End == False):
            line = line.split()
            try:
                if float(line[1].strip('-')) > 1.0:
                    freqs.append(float(line[1].strip('-')))
                    zpe += float(line[1].strip('-'))
            except:
                End = True
    with open('imagFreq.xyz','wb') as wfd:
        rmol = read('freq.m-001.s-005.xyz')
        pmol = read('freq.m-001.s-015.xyz')
        for file in os.listdir():
            if (file.endswith(".xyz") and file != 'imagFreq.xyz'):
                if 'freq.m-001' in str(file):
                    with open(file,'rb') as fd:
                        shutil.copyfileobj(fd, wfd, 1024*1024*10)
                os.remove(file)
    shutil.copyfile('imagFreq.xyz', opath + '.xyz')
    os.remove('imagFreq.xyz')
    os.chdir(pPath)
    zpe *= 0.00012
    zpe /= 2
    return imagFreq,freqs,zpe,rmol,pmol

def getGausOut(workPath, keyWords, mol):
    path = os.getcwd()
    if not os.path.exists(workPath):
        os.makedirs(workPath)
    os.chdir(workPath)
    if 'triplet' in keyWords:
        level = keyWords.split('_')
        triplet = True
        keyWords = level[0]
    else:
        triplet = False
    commands = "# opt=(calcall, tight) " + keyWords + "\n\nOpt\n\n"
    spinLine = " 0 " + str(getSpinMult(mol,"none",trip = triplet)) + "\n"
    inp = str(commands) + spinLine + str(convertMolToGauss(mol))


    f=open("Opt.gjf", "w")
    f.write(inp)
    f.close()
    gaussPath = os.environ['CHEMDYME_GAUSS']
    os.system(os.environ['CHEMDYME_GAUSS'] + " Opt.gjf" )
    try:
        mol,vibs,zpe = readGaussOutput((workPath +"/Opt.log"))
    except:
        print("problem reading gaussian file")
    os.chdir(workPath)
    return mol,vibs,zpe

def getGausTSOut(workPath, outpath, keyWords, rMol, pMol, mol, biMole, QST3):
    path = os.getcwd()
    os.chdir(workPath)
    level = keyWords.split('_')
    if 'triplet' in keyWords:
        level = keyWords.split('_')
        triplet = True
        keyWords = level[0]
    else:
        triplet = False
    if (QST3):
        commands = "# opt=(QST3,calcall,tight) Guess=Always " + keyWords + "\n\nReac\n\n"
        spinLine = " 0 " + str(getSpinMult(rMol,"none",trip = triplet)) + "\n"
        inp = str(commands) + spinLine + str(convertMolToGauss(mol))
        spinLine2 = "Prod\n\n 0 " + str(getSpinMult(rMol,"none",trip = triplet)) + "\n"
        inp = inp + spinLine2 + str(convertMolToGauss(pMol))
        spinLine3 = "TS\n\n 0 " + str(getSpinMult(rMol,"none",trip = triplet)) + "\n"
        inp = inp + spinLine3 + str(convertMolToGauss(mol))
    else:
        commands = "# opt=(ts,calcall,tight, noeigentest) " + keyWords + "\n\nTS\n\n"
        spinLine = " 0 " + str(getSpinMult(rMol,"none")) + "\n"
        inp = str(commands) + spinLine + str(convertMolToGauss(mol))

    f=open("Opt.gjf", "w")
    f.write(inp)
    f.close()
    os.system(os.environ['CHEMDYME_GAUSS'] + " Opt.gjf" )
    try:
        mol,vibs,zpe,imaginaryFreq = readGaussTSOutput("Opt.log")
    except:
        print('Error reading gaussian ts output')
    print("Gaussian ts opt finished. Output copied to " + str(outpath) +"/TS2.log")
    oPath = os.path.normpath(str(outpath)+"/TS.log")
    try:
        shutil.copyfile("Opt.gjf", str(outpath) + "/Data/TS.gjf")
    except:
        shutil.copyfile("Opt.gjf", str(outpath) + "/Data/TS2.gjf")
    if os.path.isfile(oPath):
        shutil.copyfile("Opt.log",str(outpath)+"/Data/TS_2.log")
    else:
        shutil.copyfile("Opt.log",str(outpath)+"/Data/TS.log")
    print("Gaussian TS opt read zpe = " + str(zpe))
    os.chdir(workPath)
    return mol,imaginaryFreq,vibs,zpe,rMol,pMol

def convertMolToGauss(mol):
    atoms = mol.get_atomic_numbers()
    cart = mol.get_positions()
    # Create open babel molecule BABMol
    BABmol = openbabel.OBMol()
    for i in range(0,atoms.size):
        a = BABmol.NewAtom()
        a.SetAtomicNum(int(atoms[i]))
        a.SetVector(float(cart[i,0]), float(cart[i,1]), float(cart[i,2]))

    # Assign bonds and fill out angles and torsions
    BABmol.ConnectTheDots()
    BABmol.FindAngles()
    BABmol.FindTorsions()
    BABmol.PerceiveBondOrders()
    BABmol.SetTitle('')

    #Create converter object to convert from XYZ to cml
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "gjf")
    gjf = (obConversion.WriteString(BABmol))
    # Just get molecular coordinates in gaussian form, dont trust obabel to get the righ spin multiplicity
    gjf = gjf.split('\n')[5:]
    return '\n'.join(gjf)

def readGaussOutput(path):
    vibs = []
    zpe = 0
    inp = open(path, "r")
    for line in inp:
        if re.search("Frequencies", line):
            l = line.split()
            vibs.append(float(l[2]))
            zpe += float(l[2])
            try:
                vibs.append(float(l[3]))
                zpe += float(float(l[3]))
            except:
                pass
            try:
                vibs.append(float(l[4]))
                zpe += float(float(l[4]))
            except:
                pass
    try:
        mol = read(filename=path,format="gaussian-out")
    except:
        print("couldnt read gaussian output")
    zpe *= 0.00012
    zpe /= 2
    return mol,vibs, zpe

def readGaussTSOutput(path):
    vibs = []
    zpe = 0
    inp = open(path, "r")
    for line in inp:
        if re.search("Frequencies", line):
            try:
                l = line.split()
                vibs.append(float(l[2]))
                zpe += float(l[2])
                vibs.append(float(l[3]))
                zpe += float(l[3])
                vibs.append(l[4])
                zpe += float(float(l[4]))
            except:
                pass
        if re.search("Error termination"):
            return 0
    try:
        mol = read(filename=path,format="gaussian-out")
    except:
        print("couldnt read gaussian output")
    if vibs[0] > -250:
        print("GaussianTS has no imaginary Frequency")
        return
    if vibs[1] < 0:
        print("GaussianTS has more than 1 imaginary Frequency")
        return

    zpe -= vibs[0]
    imaginaryFreq = abs((vibs[0]))
    vibs.pop(0)
    zpe *= 0.00012
    zpe /= 2
    return mol,vibs, zpe, imaginaryFreq


#Method to identify the code required for obtaining potential forces.
#This call the correct method from Calculators.py and sets up an ASE calculator on Mol
def setCalc(mol, lab, method, level):
    if method == 'xtb':
        mol = calc.xtb(mol,level)
    if method == 'scine':
        mol = calc.scine(mol,lab,level)
    if method == 'dftb':
        mol = calc.dftb(mol, lab, level)
    if method == 'nwchem':
        mol = calc.nwchem(mol, lab, level)
    if method == 'nwchem2':
        mol = calc.nwchem2(mol, lab, level)
    if method == 'nwchemTS':
        mol = calc.nwchemTS(mol, lab, level)
    if method == 'nwchemOpt':
        mol = calc.nwchemOpt(mol, lab, level)
    if method == 'nwchemFreq':
        mol = calc.nwchemFreq(mol, lab, level)
    if method == 'mopac':
        mol = calc.mopac(mol, lab, level)
    if method == 'mopacTS':
        mol = calc.mopacTS(mol, lab, level)
    if method == 'mopacOpt':
        mol = calc.mopacOpt(mol, lab, level)
    if method == 'mopacFreq':
        mol = calc.mopacFreq(mol, lab, level)
    if method == 'mopacIRC1':
        mol = calc.mopacIRC1(mol, lab, level)
    if method == 'mopacIRC2':
        mol = calc.mopacIRC2(mol, lab, level)
    if method == 'gaussian':
        mol = calc.gaussian(mol, lab, level)
    if method =='openMM':
        mol.set_calculator(OpenMMCalculator(atomTypes=level.atomTypes, input="test.pdb", ASEmol = mol, fileType = level.MMfile))
    return mol








