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
    def __init__(self, cartesians, species, i, glo):
        self.twoStageTS = False
        self.tempBiEne = 0
        self.energyDictionary = {"".join(species): 0}
        self.MiddleSteps = 2
        self.NEBrelax = glo.NEBrelax
        self.NEBsteps = glo.NEBsteps
        self.eneBaseline = 0.0
        self.workingDir = os.getcwd()
        self.procNum = str(i)
        self.lowString = 'Raw/calcLow' + str(i) + '/calc'
        self.lowLev = glo.lowerLevel
        self.lowMeth = glo.lowerMethod
        self.highLev = glo.higherLevel
        self.highMeth = glo.higherMethod
        self.highString = 'Raw/calcHigh' + str(i) + '/calc'
        self.singleString = 'Raw/calcSingle' + str(i) + '/calc'
        self.singleMeth = glo.singleMethod
        self.singleLev = glo.singleLevel
        self.ReacFreqs = []
        self.Reac = Atoms(symbols=species, positions=cartesians)
        self.ReacName = tl.getSMILES(self.Reac, False)
        self.TS = Atoms(symbols=species, positions=cartesians)
        self.TSFreqs = []
        self.TS2 = Atoms(symbols=species, positions=cartesians)
        self.TS2Freqs = []
        self.ProdFreqs = []
        self.Prod = Atoms(symbols=species, positions=cartesians)
        self.ProdName = tl.getSMILES(self.Prod, False)
        self.biProd = Atoms(symbols=species, positions=cartesians)
        self.biProdName = tl.getSMILES(self.Prod, False)
        self.biProdFreqs = []
        self.biReac = Atoms(symbols=species, positions=cartesians)
        self.biReacName = tl.getSMILES(self.Reac, False)
        self.biReacFreqs = []
        self.CombProd = Atoms(symbols=species, positions=cartesians)
        self.CombReac = Atoms(symbols=species, positions=cartesians)
        self.is_bimol_prod = False
        self.is_bimol_reac = False
        self.Reac = tl.setCalc(self.Reac, self.lowString, self.lowMeth, self.lowLev)
        self.Prod = tl.setCalc(self.Prod, self.lowString, self.lowMeth, self.lowLev)
        self.imaginaryFreq = 0.0
        self.imaginaryFreq2 = 0.0
        self.Reac = tl.setCalc(self.TS, self.lowString, self.lowMeth, self.lowLev)
        self.barrierlessReaction = False
        self.forwardBarrier = 0.0
        self.forwardBarrier2 = 0.0
        self.reactantEnergy = 0.0
        self.productEnergy = 0.0
        self.have_reactant = False
        self.inc = glo.dynPrintFreq
        self.dynPrintStart = glo.dynPrintStart
        self.printNEB = glo.printNEB
        self.QTS3 = glo.QTS3
        self.MidPoint = Atoms(symbols=species, positions=cartesians)
        self.checkAltProd = glo.checkAltProd
        self.AltProd = Atoms(symbols=species, positions=cartesians)
        self.is_IntermediateProd = False
        self.TScorrect = False
        self.TS2correct = False

    def compareRandP(self, rmol, pmol):
        # Check if TS links reac and prod
        rmol = tl.setCalc(rmol, self.lowString, self.lowMeth, self.lowLev)
        min = BFGS(rmol)
        try:
            min.run(fmax=0.1, steps=50)
        except:
            min.run(fmax=0.1, steps=50)
        Name = tl.getSMILES(rmol, False).strip('\n\t')
        FullName = Name.split('____')
        if len(FullName) > 1:
            FullName = FullName[0]
        pmol = tl.setCalc(pmol, self.lowString, self.lowMeth, self.lowLev)
        min = BFGS(pmol)
        try:
            min.run(fmax=0.1, steps=50)
        except:
            min.run(fmax=0.1, steps=50)
        Name2 = tl.getSMILES(pmol, False).strip('\n\t')
        FullName2 = Name2.split('____')
        if len(FullName2) > 1:
            FullName2 = FullName2[0]
        if ((FullName == self.ReacName and FullName2 == self.ProdName) or (
                FullName2 == self.ReacName and FullName == self.ProdName)):
            TScorrect = True
        else:
            TScorrect = False
            print('TS1 try does not connect reactants and products')
        return TScorrect

    def characteriseMinExt(self, mol, high):
        # Low level optimisation with BFGS
        os.chdir((self.workingDir))
        mol = tl.setCalc(mol, self.lowString, self.lowMeth, self.lowLev)
        min = BFGS(mol)
        try:
            min.run(fmax=0.1, steps=100)
        except:
            min.run(fmax=0.1, steps=100)
        if high:
            if self.highMeth == "gauss":
                mol, freqs, zpe = tl.getGausOut(self.workingDir + '/Raw/calcHigh' + self.procNum, self.highLev, mol)
                os.chdir((self.workingDir))
            else:
                # Higher level optimisation via some external program
                mol = tl.setCalc(mol, self.highString, self.highMeth + 'Opt', self.highLev)
                try:
                    mol.get_forces()
                except:
                    pass
                mol = tl.getOptGeom(self.workingDir + '/' + 'Raw/calcHigh' + self.procNum + '/', 'none', self.Reac,
                                    self.highMeth)

                # Then calculate frequencies
                os.chdir((self.workingDir + '/Raw/' + self.procNum))
                mol = tl.setCalc(mol, self.highString, self.highMeth + 'Freq', self.highLev)
                try:
                    mol.get_forces()
                except:
                    pass
                freqs, zpe = tl.getFreqs(self.workingDir + '/Raw/' + self.procNum + '/Raw/calcHigh' + self.procNum,
                                         self.highMeth)
                os.chdir((self.workingDir))
        else:
            if self.lowMeth == "gauss":
                mol, freqs, zpe = tl.getGausOut(self.workingDir + '/Raw/calcLow' + self.procNum, self.lowLev, mol)
                os.chdir((self.workingDir))
            else:
                # Higher level optimisation via some external program
                mol = tl.setCalc(mol, self.lowString, self.lowMeth + 'Opt', self.lowLev)
                try:
                    mol.get_forces()
                except:
                    pass
                mol = tl.getOptGeom(self.workingDir + '/Raw/' + 'calcLow' + self.procNum + '/', 'none', self.Reac,
                                    self.lowMeth)

                # Then calculate frequencies
                os.chdir((self.workingDir + '/Raw/' + self.procNum))
                mol = tl.setCalc(mol, self.lowString, self.lowMeth + 'Freq', self.lowLev)
                try:
                    mol.get_forces()
                except:
                    pass
                freqs, zpe = tl.getFreqs(self.workingDir + '/Raw/' + self.procNum + '/Raw/calcLow' + self.procNum,
                                         self.lowMeth)
                os.chdir((self.workingDir))
        # Finally get single point energy
        mol = tl.setCalc(mol, self.singleString, self.singleMeth, self.singleLev)
        energy = mol.get_potential_energy() + zpe
        return freqs, energy, mol

    def characteriseTSExt(self, mol, low, path, QTS3):

        os.chdir((self.workingDir))
        # Higher level optimisation via some external program
        if low:
            print("looking for TS at lower level")
            mol = tl.setCalc(mol, self.lowString, self.lowMeth + 'TS', self.lowLev)
        else:
            mol = tl.setCalc(mol, self.highString, self.highMeth + 'TS', self.highLev)

        try:
            mol.get_forces()
        except:
            pass

        if not low and self.highMeth == "gauss":
            mol, imaginaryFreq, TSFreqs, zpe, rmol, pmol = tl.getGausTSOut(
                self.workingDir + '/Raw/calcHigh' + self.procNum, path, self.highLev, self.CombReac, self.CombProd, mol,
                self.is_bimol_reac, QTS3)
            os.chdir((self.workingDir))
        else:
            print("getting TS geom")
            if low:
                mol = tl.getOptGeom(self.workingDir + '/Raw/' + 'calcLow' + self.procNum + '/', 'none', self.Reac,
                                    self.lowMeth)
            else:
                mol = tl.getOptGeom(self.workingDir + '/Raw/' + 'calcLow' + self.procNum + '/', 'none', self.Reac,
                                    self.lowMeth)

            # Then calculate frequencies
            os.chdir((self.workingDir + '/Raw/' + self.procNum))
            print("getting TS freqs")
            if low:
                mol = tl.setCalc(mol, self.lowString, self.lowMeth + 'Freq', self.lowLev)
            else:
                mol = tl.setCalc(mol, self.highString, self.highMeth + 'Freq', self.highLev)
            try:
                mol.get_forces()
            except:
                pass
            print("reading TS freqs")
            if low:
                try:
                    imaginaryFreq, TSFreqs, zpe, rmol, pmol = tl.getTSFreqs(
                        self.workingDir + '/Raw/' + self.procNum + '/Raw/calcLow' + self.procNum,
                        path + '/TSPreDirectImag', self.lowMeth, self.TS)
                except:
                    os.chdir((self.workingDir))
            else:
                try:
                    imaginaryFreq, TSFreqs, zpe, rmol, pmol = tl.getTSFreqs(
                        self.workingDir + '/Raw/' + self.procNum + '/Raw/calcHigh' + self.procNum,
                        path + '/TSDirectImag', self.highMeth, self.TS)
                except:
                    os.chdir((self.workingDir))
            os.chdir((self.workingDir))

        # Finally get single point energy
        mol = tl.setCalc(mol, self.singleString, self.singleMeth, self.singleLev)
        print(
            "Getting single point energy for reaction" + str(self.ReacName) + "_" + str(self.ProdName) + " TS = " + str(
                mol.get_potential_energy()) + "zpe = " + str(zpe) + "reactant energy = " + str(self.reactantEnergy))
        energy = mol.get_potential_energy() + zpe
        return TSFreqs, imaginaryFreq, zpe, energy, mol, rmol, pmol

    def characteriseTSinternal(self, mol):
        os.chdir((self.workingDir + '/Raw/' + self.procNum))
        if (self.lowMeth == 'nwchem'):
            mol = tl.setCalc(mol, self.lowString, 'nwchem2', self.lowLev)
            self.Reac.get_forces()
        else:
            mol = tl.setCalc(mol, self.lowString, self.lowMeth, self.lowLev)
        vib = Vibrations(mol)
        vib.clean()
        vib.run()
        viblist = vib.get_frequencies()
        print("getting vibs")
        TSFreqs, zpe = tl.getVibString(viblist, False, True)
        print("vibs done " + str(zpe))
        imaginaryFreq = tl.getImageFreq(viblist)
        vib.clean()
        os.chdir((self.workingDir))
        # Finally get single point energy
        mol = tl.setCalc(mol, self.singleString, self.singleMeth, self.singleLev)
        print("Getting single point energy for TS = " + str(mol.get_potential_energy()) + "zpe = " + str(
            zpe) + "reactant energy = " + str(self.reactantEnergy))
        energy = mol.get_potential_energy() + zpe
        return TSFreqs, imaginaryFreq, zpe, energy

    def characteriseMinInternal(self, mol):
        os.chdir((self.workingDir + '/' + '/Raw/' + self.procNum))
        if (self.lowMeth == 'nwchem'):
            mol = tl.setCalc(mol, self.lowString, 'nwchem2', self.lowLev)
            self.Reac.get_forces()
        else:
            mol = tl.setCalc(mol, self.lowString, self.lowMeth, self.lowLev)
        min = BFGS(mol)
        min.run(fmax=0.05, steps=50)
        vib = Vibrations(mol)
        vib.clean()
        vib.run()
        viblist = vib.get_frequencies()
        Freqs, zpe = tl.getVibString(viblist, False, False)
        vib.clean()
        os.chdir((self.workingDir))
        # Finally get single point energy
        mol = tl.setCalc(mol, self.singleString, self.singleMeth, self.singleLev)
        print("Getting single point energy for TS = " + str(mol.get_potential_energy()) + "zpe = " + str(
            zpe) + "reactant energy = " + str(self.reactantEnergy))
        energy = mol.get_potential_energy() + zpe
        return Freqs, energy, mol

    def optReac(self):
        self.is_bimol_reac = False
        self.CombReac = tl.setCalc(self.CombReac, self.lowString, self.lowMeth, self.lowLev)
        self.ReacName = tl.getSMILES(self.CombReac, True)
        FullName = self.ReacName.split('____', 1)
        if len(FullName) > 1:
            self.is_bimol_reac = True
            self.ReacName = FullName[0].strip('\n\t')
            self.biReacName = FullName[1].strip('\n\t')
            self.Reac = tl.getMolFromSmile(self.ReacName)
            self.biReac = tl.getMolFromSmile(self.biReacName)
        else:
            self.Reac = self.CombReac
        try:
            self.ReacFreqs, self.reactantEnergy, self.Reac = self.characteriseMinExt(self.Reac, True)
        except:
            self.ReacFreqs, self.reactantEnergy, self.Reac = self.characteriseMinInternal(self.Reac)
        if self.is_bimol_reac == True:
            try:
                self.biReacFreqs, Ene, self.biReac = self.characteriseMinExt(self.biReac, True)
            except:
                self.biReacFreqs, Ene, self.biReac = self.characteriseMinInternal(self.biReac)
            self.reactantEnergy += Ene

    def optProd(self, cart, alt):
        print('optimising product')
        self.is_bimol_prod = False
        if alt == True:
            self.CombProd = self.AltProd.copy()
        else:
            self.CombProd.set_positions(cart)
        self.CombProd = tl.setCalc(self.CombProd, self.lowString, self.lowMeth, self.lowLev)
        min = BFGS(self.CombProd)
        self.ProdName = tl.getSMILES(self.CombProd, True, partialOpt=True)
        if self.is_bimol_reac == True:
            self.ProdName = self.ProdName.replace('____', 'comp')
        FullName = self.ProdName.split('____', 1)
        if len(FullName) > 1:
            self.is_bimol_prod = True
            self.ProdName = FullName[0].strip('\n\t')
            self.biProdName = FullName[1].strip('\n\t')
            self.Prod = tl.getMolFromSmile(self.ProdName)
            self.biProd = tl.getMolFromSmile(self.biProdName)
        else:
            self.Prod = self.CombProd
        try:
            self.ProdFreqs, self.productEnergy, self.Prod = self.characteriseMinExt(self.Prod, True)
        except:
            self.ProdFreqs, self.productEnergy, self.Prod = self.characteriseMinExt(self.Prod, False)
        if self.is_bimol_prod:
            try:
                self.biProdFreqs, Ene, self.biProd = self.characteriseMinExt(self.biProd, True)
            except:
                self.biProdFreqs, Ene, self.biProd = self.characteriseMinExt(self.biProd, False)
            self.productEnergy += Ene

    def optTSpoint(self, trans, path, MolList, TrajStart, idx):

        self.TS = MolList[TrajStart].copy()
        self.TS = tl.setCalc(self.TS, self.lowString, self.lowMeth, self.lowLev)
        c = FixAtoms(trans)
        self.TS.set_constraint(c)
        min = BFGS(self.TS)
        min.run(fmax=0.05, steps=50)
        os.mkdir(path + '/Data/')
        write(path + '/Data/TSGuess.xyz', self.TS)
        TSGuess = self.TS.copy()
        del self.TS.constraints
        if self.biReac or self.biProd:
            QTS3 = False
        else:
            QTS3 = True

        if (self.twoStageTS):
            try:
                self.TSFreqs, self.imaginaryFreq, zpe, energy, self.TS, rmol, pmol = self.characteriseTSExt(self.TS,
                                                                                                            True, path,
                                                                                                            QTS3)
            except:
                pass
        try:
            self.TSFreqs, self.imaginaryFreq, zpe, energy, self.TS, rmol, pmol = self.characteriseTSExt(self.TS, False,
                                                                                                        path, QTS3)
        except:
            try:
                print("High Level TS opt for TS1 failed looking at lower level")
                self.TSFreqs, self.imaginaryFreq, zpe, energy, self.TS, rmol, pmol = self.characteriseTSExt(self.TS,
                                                                                                            True, path,
                                                                                                            QTS3)
            except:
                pass

        try:
            self.TScorrect = self.compareRandP(rmol, pmol)
        except:
            self.TScorrect = False
            self.TS = TSGuess
            self.TSFreqs, self.imaginaryFreq, zpe, energy = self.characteriseTSinternal(self.TS)

        write(path + '/TS1.xyz', self.TS)

        self.forwardBarrier = energy

        if (self.forwardBarrier < self.reactantEnergy and self.forwardBarrier < self.productEnergy):
            self.barrierlessReaction = True

    def refineTSpoint(self, MolList, TrajStart):

        iMol = MolList[TrajStart - 100].copy()
        iMol = tl.setCalc(iMol, self.lowString, self.lowMeth, self.lowLev)
        startName = tl.getSMILES(iMol, True)
        startName = startName.split('____')
        if len(startName) > 1:
            startName = startName[0]

        # Look for change in optimised geometry along reaction path
        point = 0
        for i in range(TrajStart - 99, TrajStart + 100):
            iMol = MolList[i].copy()
            iMol = tl.setCalc(iMol, self.lowString, self.lowMeth, self.lowLev)
            Name = tl.getSMILES(iMol, True)
            FullName = Name.split('____')
            if len(FullName) > 1:
                FullName = FullName[0]
            if FullName != self.startName and FullName != self.ReacName:
                point = i
                return point
        return TrajStart

    def optDynPath(self, trans, path, MolList, changePoints):

        # Open files for saving IRCdata
        xyzfile = open((path + "/Data/dynPath.xyz"), "w")
        MEP = open((path + "/Data/MEP.txt"), "w")
        orriginal = open((path + "/Data/orriginalPath.txt"), "w")
        dyn = open((path + "/Data/traj.xyz"), "w")

        dynList = []
        end = int(changePoints + self.dynPrintStart)
        length = int(len(MolList))
        if end < length:
            endFrame = end
        else:
            endFrame = length
        for i in range(int(changePoints - self.dynPrintStart), int(endFrame), self.inc):
            iMol = MolList[i].copy()
            tl.printTraj(dyn, iMol)
            iMol = tl.setCalc(iMol, self.lowString, self.lowMeth, self.lowLev)
            orriginal.write(str(i) + ' ' + str(iMol.get_potential_energy()) + '\n')
            c = FixAtoms(trans)
            min = BFGS(iMol)
            try:
                min.run(fmax=0.1, steps=50)
            except:
                min.run(fmax=0.1, steps=1)
            del iMol.constraints
            tl.printTraj(xyzfile, iMol)
            dynList.append(iMol.copy())

        maxEne = -50000
        point = 0
        for i in range(0, len(dynList)):
            dynList[i] = tl.setCalc(dynList[i], self.lowString, self.lowMeth, self.lowLev)
            MEP.write(str(i) + ' ' + str(dynList[i].get_potential_energy()) + '\n')
            if dynList[i].get_potential_energy() > maxEne:
                point = i
                maxEne = dynList[i].get_potential_energy()

        self.TS2 = dynList[point]
        TS2Guess = self.TS2.copy()
        write(path + '/Data/TS2guess.xyz', self.TS2)
        try:
            self.TS2Freqs, self.imaginaryFreq2, zpe, energy, self.TS2, rmol, pmol = self.characteriseTSExt(self.TS2,
                                                                                                           True, path,
                                                                                                           self.QTS3)
        except:
            try:
                self.TS2Freqs, self.imaginaryFreq2, zpe, energy, self.TS2, rmol, pmol = self.characteriseTSExt(self.TS2,
                                                                                                               False,
                                                                                                               path,
                                                                                                               self.QTS3)
            except:
                pass
        try:
            self.TS2correct = self.compareRandP(rmol, pmol)
        except:
            print("TS2 does not connect products")
            self.TS2correct = False
            self.TS2 = TS2Guess
            self.TS2Freqs, self.imaginaryFreq2, zpe, energy = self.characteriseTSinternal(self.TS2)

        write(path + '/TS2.xyz', self.TS2)
        self.forwardBarrier2 = energy

    def optNEB(self, trans, path, changePoints, mols):

        # Open files for saving IRCdata
        xyzfile3 = open((path + "/IRC3.xyz"), "w")
        MEP = open((path + "/MEP.txt"), "w")
        imagesTemp1 = []
        index = changePoints[0] - 100
        molTemp = mols[index].copy()
        imagesTemp1.append(molTemp.copy())
        for i in range(0, 100):
            imagesTemp1.append(mols[changePoints[0] - 100].copy())
        try:
            imagesTemp1.append(mols[changePoints[-1] + 300])
        except:
            imagesTemp1.append(self.CombProd.copy())
        neb1 = NEB(imagesTemp1, k=1.0, remove_rotation_and_translation=True)
        try:
            neb1.interpolate('idpp')
        except:
            neb1.interpolate()

        for i in range(0, len(imagesTemp1)):
            try:
                imagesTemp1[i] = tl.setCalc(imagesTemp1[i], self.lowString, self.lowMeth, self.lowLev)
                for i in range(0, len(trans)):
                    c = FixAtoms(trans)
                    imagesTemp1[i].set_constraint(c)
                min = BFGS(imagesTemp1[i])
                min.run(fmax=0.005, steps=40)
                del imagesTemp1[i].constraints
            except:
                pass

        optimizer = FIRE(neb1)
        try:
            optimizer.run(fmax=0.07, steps=300)
        except:
            pass
            print("passed seccond neb")

        neb1_2 = NEB(imagesTemp1, k=0.01, remove_rotation_and_translation=True)
        try:
            optimizer.run(fmax=0.07, steps=200)
        except:
            pass

        neb2 = NEB(imagesTemp1, climb=True, remove_rotation_and_translation=True)
        try:
            optimizer.run(fmax=0.07, steps=200)
        except:
            pass

        for i in range(0, len(imagesTemp1)):
            tl.printTraj(xyzfile3, imagesTemp1[i])

        print("NEB printed")

        xyzfile3.close()

        point = 0
        maxEne = -50000000

        try:
            for i in range(0, len(imagesTemp1)):
                MEP.write(str(i) + ' ' + str(imagesTemp1[i].get_potential_energy()) + '\n')
                if imagesTemp1[i].get_potential_energy() > maxEne and i > 5:
                    point = i
                    maxEne = imagesTemp1[i].get_potential_energy()
            self.TS2 = imagesTemp1[point]
        except:
            point = 0

        print("TS Climb part")
        write(path + '/TSClimbGuess.xyz', self.TS2)
        try:
            self.TS2Freqs, self.imaginaryFreq2, zpe, energy, self.TS2, rmol, pmol = self.characteriseTSExt(self.TS2,
                                                                                                           False, path,
                                                                                                           self.QTS3)
        except:
            self.TS2Freqs, self.imaginaryFreq2, zpe, energy, self.TS2, rmol, pmol = self.characteriseTSExt(self.TS2,
                                                                                                           True, path,
                                                                                                           self.QTS3)

        self.TScorrect = self.compareRandP(rmol, pmol)
        self.forwardBarrier2 = energy + zpe

    def printTS(self, path):
        write(path + '/TS.xyz', self.TS)

    def printProd(self, path):
        write(path + '/Prod.xyz', self.Prod)
        if self.is_bimol_prod:
            write(path + '/biProd.xyz', self.biProd)

    def printReac(self, path):
        write(path + '/Reac.xyz', self.Reac)
        if self.is_bimol_reac:
            write(path + '/biReac.xyz', self.biReac)

    def newReac(self, path, name, Reac):
        if Reac == True:
            self.Reac = read(path + '/Reac.xyz')
        else:
            self.Reac = read(path + '/Prod.xyz')
        self.ReacName = name
        self.CombReac = self.Reac.copy()
        self.TS = self.Reac.copy()
        self.TSFreqs = []
        self.TS2 = self.Reac.copy()
        self.TS2Freqs = []
        self.ProdFreqs = []
        self.Prod = self.Reac.copy()
        self.ProdName = tl.getSMILES(self.Prod, False)
        self.biProd = self.Reac.copy()
        self.biProdName = tl.getSMILES(self.Prod, False)
        self.biProdFreqs = []
        self.CombProd = self.Reac.copy()
        self.is_bimol_prod = False
        self.is_bimol_reac = False
        self.Reac = tl.setCalc(self.Reac, self.lowString, self.lowMeth, self.lowLev)
        self.TS = tl.setCalc(self.TS, self.lowString, self.lowMeth, self.lowLev)
        self.TS2 = tl.setCalc(self.TS2, self.lowString, self.lowMeth, self.lowLev)
        self.imaginaryFreq = 0.0
        self.imaginaryFreq2 = 0.0
        self.Prod = tl.setCalc(self.Prod, self.lowString, self.lowMeth, self.lowLev)
        self.barrierlessReaction = False
        self.forwardBarrier = 0.0
        self.forwardBarrier2 = 0.0
        self.reactantEnergy = self.Reac.get_potential_energy()
        self.have_reactant = True
        self.is_IntermediateProd = False
        self.AltProd = self.Reac.copy()
        self.TScorrect = False

    def newReacFromSMILE(self, SMILE):
        self.ReacName = SMILE
        SMILE.replace('____', '.')
        SMILE.replace('comp', '.')
        self.Reac = tl.getMolFromSmile(SMILE)
        self.CombReac = self.Reac.copy()
        self.TS = self.Reac.copy()
        self.TSFreqs = []
        self.TS2 = self.Reac.copy()
        self.TS2Freqs = []
        self.ProdFreqs = []
        self.Prod = self.Reac.copy()
        self.ProdName = tl.getSMILES(self.Prod, False)
        self.biProd = self.Reac.copy()
        self.biProdName = tl.getSMILES(self.Prod, False)
        self.biProdFreqs = []
        self.CombProd = self.Reac.copy()
        self.is_bimol_prod = False
        self.is_bimol_reac = False
        self.Reac = tl.setCalc(self.Reac, self.lowString, self.lowString, self.lowString)
        self.TS = tl.setCalc(self.TS, self.lowString, self.lowMeth, self.lowLev)
        self.TS2 = tl.setCalc(self.TS2, self.lowString, self.lowMeth, self.lowLev)
        self.imaginaryFreq = 0.0
        self.imaginaryFreq2 = 0.0
        self.Prod = tl.setCalc(self.Prod, self.lowString, self.lowMeth, self.lowLev)
        self.barrierlessReaction = False
        self.forwardBarrier = 0.0
        self.forwardBarrier2 = 0.0
        self.reactantEnergy = self.Reac.get_potential_energy()
        self.have_reactant = True
        self.is_IntermediateProd = False
        self.AltProd = self.Reac.copy()
        self.TScorrect = False

    def re_init(self, path):
        self.TS = read(path + '/Reac.xyz')
        self.TSFreqs = []
        self.TS2 = read(path + '/Reac.xyz')
        self.TS2Freqs = []
        self.ProdFreqs = []
        self.Prod = read(path + '/Reac.xyz')
        self.ProdName = tl.getSMILES(self.Prod, False).strip('\n\t')
        self.biProd = read(path + '/Reac.xyz')
        self.biProdName = tl.getSMILES(self.Prod, False).strip('\n\t')
        self.biProdFreqs = []
        self.CombProd = read(path + '/Reac.xyz')
        self.is_bimol_prod = False
        self.TS = tl.setCalc(self.TS, self.lowString, self.lowMeth, self.lowLev)
        self.TS2 = tl.setCalc(self.TS2, self.lowString, self.lowMeth, self.lowLev)
        self.imaginaryFreq = 0.0
        self.imaginaryFreq2 = 0.0
        self.Prod = tl.setCalc(self.Prod, self.lowString, self.lowMeth, self.lowLev)
        self.barrierlessReaction = False
        self.forwardBarrier = 0.0
        self.forwardBarrier2 = 0.0
        self.productEnergy = 0.0
        self.TScorrect = False

    def re_init_bi(self, cartesians, species):
        self.TS = Atoms(symbols=species, positions=cartesians)
        self.TSFreqs = []
        self.TS2 = Atoms(symbols=species, positions=cartesians)
        self.TS2Freqs = []
        self.ProdFreqs = []
        self.Prod = Atoms(symbols=species, positions=cartesians)
        self.ProdName = tl.getSMILES(self.Prod, False)
        self.biProd = Atoms(symbols=species, positions=cartesians)
        self.biProdName = tl.getSMILES(self.Prod, False)
        self.biProdFreqs = []
        self.CombProd = Atoms(symbols=species, positions=cartesians)
        self.is_bimol_prod = False
        self.TS = tl.setCalc(self.TS, self.lowString, self.lowMeth, self.lowLev)
        self.TS2 = tl.setCalc(self.TS2, self.lowString, self.lowMeth, self.lowLev)
        self.imaginaryFreq = 0.0
        self.imaginaryFreq2 = 0.0
        self.Prod = tl.setCalc(self.Prod, self.lowString, self.lowMeth, self.lowLev)
        self.barrierlessReaction = False
        self.forwardBarrier = 0.0
        self.forwardBarrier2 = 0.0
        self.productEnergy = 0.0
        self.TScorrect = False
        self.optReac()

    def TempBiEne(self, tempBi):
        tempBi = tl.setCalc(tempBi, self.singleString, self.singleMeth, self.singleLev)
        return tempBi.get_potential_energy()





