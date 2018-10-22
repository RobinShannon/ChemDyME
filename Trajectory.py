import Potentials
import BXDconstraint
import os
from ase import Atoms
import MDIntegrator
import Connectivity
import numpy as np
import Tools as tl
import ConnectTools as ct
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,Stationary, ZeroRotation)
from ase import units
from time import time
from ase.optimize import BFGS
from ase import io as aio

class Trajectory:

    def __init__(self, mol, gl, path, i, Bi):
        self.printSMILES = False
        self.procNum = i
        self.gl = gl
        self.biMolecular = Bi
        self.method = gl.trajMethod
        self.level = gl.trajLevel
        self.mdSteps = gl.mdSteps
        self.printFreq = gl.printFreq
        if self.biMolecular:
            self.initialT = gl.BiTemp
        else:
            self.initialT = gl.LTemp
        self.MDIntegrator =gl.MDIntegrator
        self.timeStep = gl.timeStep * units.fs
        self.eneBXD = gl.eneBXD
        self.comBXD = gl.comBXD
        try:
            self.minCOM = gl.minCOM
        except:
            self.minCOM = 3.0
        try:
            self.fragIdx = gl.fragIndices
        except:
            self.fragIdx = 0
        self.forces = np.zeros(mol.get_positions().shape)
        self.LangFric = gl.LFric
        self.LangTemp = self.initialT
        self.Mol = Atoms(symbols=mol.get_chemical_symbols(), positions = mol.get_positions())
        self.Mol = tl.setCalc(self.Mol,'calcMopac' + str(self.procNum) + '/Traj', self.method, self.level)
        MaxwellBoltzmannDistribution(self.Mol, self.initialT * units.kB)
        self.velocity = self.Mol.get_velocities()
        self.tempReactGeom = mol.copy()
        self.tempProdGeom = mol.copy()
        self.NEBguess = []
        self.TSgeom = mol.copy()
        self.productGeom = self.Mol.get_positions()
        try:
            self.window = gl.reactionWindow
            self.endOnReac = gl.endOnReaction
            self.consistantWindow = gl.reactionWindow
        except:
            pass
        self.savePath = path
        self.ReactionCountDown = 0
        self.MolList = [self.Mol]
        self.KeepTracking = True
        self.tempList = [1000]
        self.window = 2
        self.transindex = np.zeros(3)
        self.numberOfSteps = 0
        self.AssDisoc = False
        self.TSpoint = 0
        self.names=[]
        self.changePoints=[]


    def runTrajectory(self):
        # Create specific directory
        workingDir = os.getcwd()
        newpath = workingDir + '/traj' + str(self.procNum)
        namefile = open(("traj.xyz"), "w")
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        os.chdir(newpath)


        self.numberOfSteps = 0
        consistantChange = 0



        eneBXDon = False
        eBounded= False
        comBounded = False

        # Get potential type
        if (self.method == 'nwchem'):
            self.Mol =tl.setCalc(self.Mol,'calcMopac' + str(self.procNum) + '/Traj', 'nwchem2', self.level)
            self.Mol.get_forces()
        self.Mol =tl.setCalc(self.Mol,'calcMopac' + str(self.procNum) + '/Traj', self.method, self.level)

        #Get MDintegrator type
        if self.MDIntegrator == 'VelocityVerlet':
            mdInt = MDIntegrator.VelocityVerlet(self.forces, self.velocity, self.Mol)
        elif self.MDIntegrator == 'Langevin':
            mdInt = MDIntegrator.Langevin(units.kB * self.LangTemp, self.LangFric, self.forces, self.velocity, self.Mol)



        # Then set up reaction criteria or connectivity map
        con = Connectivity.NunezMartinez(self.Mol)


        # Then set up various BXD procedures
        if self.comBXD or self.biMolecular:
            self.comBXD = True
            comBxd = BXDconstraint.COM(self.Mol,"fixed", 0, self.minCOM, 10000, 10000, self.fragIdx)
        if self.eneBXD:
            eneBXD = BXDconstraint.Energy(self.Mol,"adaptive", -10000, -0.1, 10000, 1000, 10000)


        # Run MD trajectory for specified number of steps
        for i in range(0,self.mdSteps):
            t = time()
            # Get forces and energy from designated potential
            try:
                self.forces = self.Mol.get_forces()
            except:
                try:
                    self.forces = self.Mol.get_forces()
                except:
                    print("forces error")

            self.ene = self.Mol.get_potential_energy()

            if i % 100 == 0:
                tl.printTraj(namefile,self.Mol)
            #Print Smiles? This intermitently checks whether the current sturcture optimises to a new species as identified by the SMILES string
            if self.printSMILES:
                if i % self.printFreq == 0:
                    tempMol = self.Mol.copy()
                    tempMol =tl.setCalc(self.Mol,'calcMopac' + str(self.procNum) + '/Traj', self.method, self.level)
                    Name = tl.getSMILES(tempMol, True)
                    self.names.append(Name)
                    length = len(self.names)
                    if length > 3 and Name == self.names[length - 2] and Name != self.names[length-3]:
                        self.changePoints.append(i-self.printFreq)


            # Update the COM seperation and check whether it is bounded
            if self.comBXD:
                comBxd.update(self.Mol)
                comBounded = comBxd.inversion
                if comBounded is True and self.ReactionCountDown == 0:
                    com_del_phi = comBxd.del_phi

            if self.comBXD and eneBXDon:
                print("Ene = " + str(eneBXD.s[0]) + ' S = ' + str(comBxd.s[0]) + ' step = ' + str(i) + ' process = ' + str(self.procNum) + ' time = ' + str(time()-t) + ' temperature = ' + str(self.Mol.get_temperature()))
            elif self.comBXD:
                print("Ene = " + "NA" + ' S = ' + str(comBxd.s[0]) + ' step = ' + str(i) + ' process = ' + str(self.procNum) + ' time = ' + str(time()-t) + ' temperature = ' + str(self.Mol.get_temperature()))
            elif eneBXDon:
                print("Ene = " + str(eneBXD.s[0]) + 'S = ' + "NA" + ' step = ' + str(i) + ' process = ' + str(self.procNum) + ' time = ' + str(time()-t) + ' temperature = ' + str(self.Mol.get_temperature()))

            #  Now check whether to turn BXDE on
            if self.comBXD:
                if (self.biMolecular or self.comBXD) and comBxd.s[0] < self.minCOM and eneBXDon == False:
                    eneBXDon = True
            elif self.eneBXD:
                eneBXDon = True

            if eneBXDon == True:
                eneBXD.update(self.Mol)
                eBounded = eneBXD.inversion
                e_del_phi = eneBXD.del_phi

            # Perform inversion if required
            if eBounded is True and comBounded is True:
                mdInt.constrain2(e_del_phi, com_del_phi)
            elif eBounded is False and comBounded is True:
                mdInt.constrain(com_del_phi)
            elif eBounded is True and comBounded is False:
                mdInt.constrain(e_del_phi)

            mdInt.mdStep(self.forces, self.timeStep, self.Mol)
            self.numberOfSteps += 1


            self.MolList.append(self.Mol.copy())

            # Check if we are in a reaction countdown
            if self.ReactionCountDown == 0:


                # Update connectivity map to check for reaction
                if eneBXDon:
                    con.update(self.Mol)
                    if con.criteriaMet is True:
                        if consistantChange == 0:
                            self.TSpoint = i
                            self.TSgeom = self.Mol.copy()
                            consistantChange = self.consistantWindow
                        elif consistantChange > 0:
                            if consistantChange == 1:
                                self.ReactionCountDown = self.window
                                consistantChange -= 2
                            else:
                                consistantChange -= 1
                    else:
                        consistantChange = 0
                        con.criteriaMet = False
                    if not self.ReactionCountDown == 0:
                        self.ReactionCountDown -= 1
                    if self.ReactionCountDown == 1:
                        if self.endOnReac is True:
                            self.ReactionCountDown = 0
                            self.productGeom = self.Mol.get_positions()
                            os.chdir(workingDir)
                            break

        os.chdir(workingDir)

    def runBXDEconvergence(self, grainsize, startingE, additionalConstraint, Mol, type, boxes, hitLimit, runs):
        # Create specific directory
        workingDir = os.getcwd()

        self.iterations = 0
        freeFly = False
        eneBXDon = True
        eBounded = False
        comBounded = False
        keepGoing = True

        namefile = []
        for i in range(0,runs):
            namefile.append(open((str(i) +"_box"), "w"))

        # Get potential type
        if not freeFly:
            if (self.method == 'nwchem'):
                self.Mol =tl.setCalc(self.Mol,'calcMopac' + str(self.procNum) + '/Traj', 'nwchem2', self.level)
                self.Mol.get_forces()
            else:
                self.Mol =tl.setCalc(self.Mol,'calcMopac' + str(self.procNum) + '/Traj', self.method, self.level)

            #Get MDintegrator type
            if self.MDIntegrator == 'VelocityVerlet':
                mdInt = MDIntegrator.VelocityVerlet(self.forces, self.velocity, self.Mol)
            elif self.MDIntegrator == 'Langevin':
                mdInt = MDIntegrator.Langevin(units.kB * self.LangTemp, self.LangFric, self.forces, self.velocity, self.Mol)

        else:
            self.forces = 0
        # Then set up various BXD procedures
        #if additionalConstraint:
            #genBxd = BXDconstraint.genBXD(self.Mol,"fixed", Mol, Product, 10000, 10000, 0)

        if (type == "fixed"):
            eneBXD = BXDconstraint.Energy(self.Mol,type, startingE, startingE + grainsize, hitLimit, boxes, 10)
            eneBXD.createFixedBoxes(grainsize)
        else:
            eneBXD = BXDconstraint.Energy(self.Mol,type, startingE, 10000, hitLimit, 1000, boxes)

        # Run MD trajectory for specified number of steps
        while keepGoing:
            t = time()
            # Get forces and energy from designated potential
            try:
                self.forces = self.Mol.get_forces()
            except:
                print('forces error')

            self.ene = self.Mol.get_potential_energy()

            # Update the COM seperation and check whether it is bounded
            #if self.comBXD:
                #genBxd.update(self.Mol)
                #comBounded = genBxd.inversion
                #if comBounded is True and self.ReactionCountDown == 0:
                    #com_del_phi = genBxd.del_constraint(self.Mol)

            if not freeFly:
                eneBXD.update(self.Mol)
                eBounded = eneBXD.inversion
                e_del_phi = eneBXD.del_phi
                print('S ' + str(self.ene)  + ' box ' + str(eneBXD.box) + ' time ' + str(time()-t) + ' temperature ' + str(self.Mol.get_temperature()))
            else:
                eBounded = False

            # Perform inversion if required
            if eBounded is True and comBounded is True:
                mdInt.constrain2(e_del_phi, com_del_phi)
            elif eBounded is False and comBounded is True:
                mdInt.constrain(com_del_phi)
            elif eBounded is True and comBounded is False:
                mdInt.constrain(e_del_phi)

            mdInt.mdStep(self.forces, self.timeStep, self.Mol)
            self.numberOfSteps += 1

            if eneBXD.stuckCount >1 :
                freeFly = True

            if not eneBXD.boxList[eneBXD.box].lower.hit( self.ene, 'down') and not self.ene < eneBXD.boxList[eneBXD.box].upper.hit(self.ene,'up'):
                freeFly = False
            if eneBXD.completeRuns == 1:
                keepGoing = False

        if type == "adaptive":
            keepGoing = True
            eneBXD.reset("fixed", 100)

        while keepGoing:
            t = time()
            # Get forces and energy from designated potential
            try:
                self.forces = self.Mol.get_forces()
            except:
                print('forces error')

            self.ene = self.Mol.get_potential_energy()

            # Update the COM seperation and check whether it is bounded
            #if self.comBXD:
                #genBxd.update(self.Mol)
                #comBounded = genBxd.inversion
                #if comBounded is True and self.ReactionCountDown == 0:
                    #com_del_phi = genBxd.del_constraint(self.Mol)

            if not freeFly:
                eneBXD.update(self.Mol)
                eBounded = eneBXD.inversion
                e_del_phi = eneBXD.del_phi
                print('S ' + str(self.ene)  + ' box ' + str(eneBXD.box) + ' time ' + str(time()-t) + ' temperature ' + str(self.Mol.get_temperature()))
            else:
                eBounded = False

            # Perform inversion if required
            if eBounded is True and comBounded is True:
                mdInt.constrain2(e_del_phi, com_del_phi)
            elif eBounded is False and comBounded is True:
                mdInt.constrain(com_del_phi)
            elif eBounded is True and comBounded is False:
                mdInt.constrain(e_del_phi)

            mdInt.mdStep(self.forces, self.timeStep, self.Mol)
            self.numberOfSteps += 1

            if self.iterations % 10 == 0:
                tl.printTraj(namefile[self.box],self.Mol)

            if eneBXD.stuckCount >1 :
                freeFly = True

            if not eneBXD.boxList[eneBXD.box].lower.hit( self.ene, 'down') and not self.ene < eneBXD.boxList[eneBXD.box].upper.hit(self.ene,'up'):
                freeFly = False
            if eneBXD.completeRuns == runs:
                keepGoing = False

        eneBXD.gatherData(False,units.kB * self.LangTemp)

    def runGenBXD(self, Reac, Prod, maxHits, adapMax, pathType, path,bonds):

        self.iterations = 0

        eneBXDon = True
        eBounded = False
        comBounded = False
        keepGoing = True

        workingDir = os.getcwd()
        newpath = workingDir + '/gen' + str(self.procNum)
        file = open("geo.xyz","w")

        # Get potential type
        if (self.method == 'nwchem'):
            self.Mol =tl.setCalc(self.Mol,'calcMopac' + str(self.procNum) + '/Traj', 'nwchem2', self.level)
            self.Mol.get_forces()
        self.Mol =tl.setCalc(self.Mol,'calcMopac' + str(self.procNum) + '/Traj', self.method, self.level)

        #Get MDintegrator type
        if self.MDIntegrator == 'VelocityVerlet':
            mdInt = MDIntegrator.VelocityVerlet(self.forces, self.velocity, self.Mol)
        elif self.MDIntegrator == 'Langevin':
            mdInt = MDIntegrator.Langevin(units.kB * self.LangTemp, self.LangFric, self.forces, self.velocity, self.Mol)

        # Then set up reaction criteria or connectivity map
        con = Connectivity.NunezMartinez(self.Mol)



        #set up adaptive bxd
        BXD = BXDconstraint.genBXD(self.Mol,"adaptive", Reac, Prod, maxHits, adapMax, bonds,path,pathType)


        # Run MD trajectory for specified number of steps
        while keepGoing:
            t = time()
            # Get forces and energy from designated potential
            try:
                self.forces = self.Mol.get_forces()
            except:
                print('forces error')


            BXD.update(self.Mol)
            eBounded = BXD.inversion
            if self.MDIntegrator == 'VelocityVerlet':
                print('S ' + str(BXD.s[2])  + ' box ' + str(BXD.box) + ' time ' + str(time()-t) + ' Epot ' + str(self.Mol.get_potential_energy() + self.Mol.get_kinetic_energy()))
            else:
                print('S ' + str(BXD.s[2])  + ' box ' + str(BXD.box) + ' time ' + str(time()-t) + ' temperature ' + str(self.Mol.get_temperature()))

            # Perform inversion if required
            if eBounded is True:
                mdInt.constrain(BXD.del_phi)

            mdInt.mdStep(self.forces, self.timeStep, self.Mol)
            self.numberOfSteps += 1

            if self.iterations % 100 == 0:
                tl.printTraj(file,self.Mol)

            self.iterations += 1;

            if BXD.completeRuns == 1:
                keepGoing = False

        keepGoing = True
        BXD.reset("fixed", 100)

        # Rerun BXD with new adaptively set bounds to converge box to box statistics
        while keepGoing:
            t = time()
            # Get forces and energy from designated potential
            try:
                self.forces = self.Mol.get_forces()
            except:
                print('forces error')


            BXD.update(self.Mol)
            eBounded = BXD.inversion
            if self.MDIntegrator == 'VelocityVerlet':
                print('S ' + str(BXD.s[2])  + ' box ' + str(BXD.box) + ' time ' + str(time()-t) + ' Epot ' + str(self.Mol.get_potential_energy() + self.Mol.get_kinetic_energy()))
            else:
                print('S ' + str(BXD.s[2])  + ' box ' + str(BXD.box) + ' time ' + str(time()-t) + ' temperature ' + str(self.Mol.get_temperature()))

            # Perform inversion if required
            if eBounded is True:
                mdInt.constrain(BXD.del_phi)

            mdInt.mdStep(self.forces, self.timeStep, self.Mol)
            self.numberOfSteps += 1

            if BXD.completeRuns == 1:
                keepGoing = False

        BXD.gatherData(False,units.kB * self.LangTemp)