import ChemDyME.BXDconstraint
import os
from ase import Atoms
import ChemDyME.MDIntegrator
import ChemDyME.Connectivity
import numpy as np
import ChemDyME.Tools as tl
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,Stationary, ZeroRotation)
from ase import units
from time import process_time


class Trajectory:

    def __init__(self, mol, gl, path, i, Bi):
        self.geom_print = gl.full_geom_print
        self.printSMILES = False
        self.procNum = i
        self.gl = gl
        self.biMolecular = Bi
        self.method = gl.trajMethod
        self.level = gl.trajLevel
        self.mdSteps = gl.mdSteps
        self.printFreq = gl.printFreq
        self.mixedTimestep = gl.mixedTimestep
        if self.biMolecular:
            try:
                self.initialT = gl.BiTemp
            except:
                print("No specific bimolecular temperature set, using the Langevin temperature instead")
                self.initialT = gl.LTemp
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
        if self.method == "openMM":
            self.Mol = tl.setCalc(self.Mol,'calcMopac' + str(self.procNum) + '/Traj', self.method, gl)
        else:
            self.Mol = tl.setCalc(self.Mol,'calcMopac' + str(self.procNum) + '/Traj', self.method, self.level)
        MaxwellBoltzmannDistribution(self.Mol, self.initialT * units.kB, force_temp=True)
        Stationary(self.Mol)
        ZeroRotation(self.Mol)
        self.velocity = self.Mol.get_velocities()
        print('initialising velocities for process ' + str(i) + '\n')
        print(str(self.velocity) + '\n')
        self.tempReactGeom = mol.copy()
        self.tempProdGeom = mol.copy()
        self.NEBguess = []
        self.TSgeom = mol.copy()
        self.productGeom = self.Mol.get_positions()
        try:
            self.window = gl.reactionWindow
            self.endOnReac = gl.endOnReaction
            self.consistantWindow = 10
        except:
            pass
        if self.biMolecular:
            self.consistantWindow = 5
        self.savePath = path
        self.ReactionCountDown = 0
        self.MolList = []
        self.KeepTracking = True
        self.tempList = [1000]
        self.transindex = np.zeros(3)
        self.numberOfSteps = 0
        self.AssDisoc = False
        self.TSpoint = 0
        self.names=[]
        self.changePoints=[]
        self.adaptiveSteps = gl.maxAdapSteps


    def runTrajectory(self):
        # Create specific directory
        fails = 0
        rmol= self.Mol.copy()
        rmol =tl.setCalc(rmol, 'Traj_' + str(self.procNum), self.method, self.level)
        reac_smile = tl.getSMILES(rmol,True)
        workingDir = os.getcwd()
        newpath = workingDir + '/Raw/traj' + str(self.procNum)
        print("making directory " + newpath)


        if not os.path.exists(newpath):
            os.makedirs(newpath)
        os.chdir(newpath)
        file_name = str(newpath + "/totaltraj.xyz")
        if os.path.isfile(file_name):
            expand = 1
            while True:
                expand += 1
                new_file_name = file_name.split(".xyz")[0] + str(expand) + ".xyz"
                if os.path.isfile(new_file_name):
                    continue
                else:
                    file_name = new_file_name
                    break

        namefile = open((file_name), "a")

        self.numberOfSteps = 0
        consistantChange = 0

        #Multiple stepsizes
        self.smallStep = self.timeStep
        timeStep = self.timeStep



        eneBXDon = False
        eBounded= False
        comBounded = False

        # Get potential type
        if (self.method == 'nwchem'):
            self.Mol =tl.setCalc(self.Mol,'Traj_' + str(self.procNum), 'nwchem2', self.level)
            self.Mol.get_forces()
        self.Mol =tl.setCalc(self.Mol, 'Traj_' + str(self.procNum), self.method, self.level)
        print("getting first forces")
        try:
            self.forces = self.Mol.get_forces()
        except:
            try:
                self.forces = self.Mol.get_forces()
            except:
                print("forces error")
                self.forces = np.zeros(len(self.forces.shape))

        #Get MDintegrator type
        if self.MDIntegrator == 'VelocityVerlet':
            mdInt = ChemDyME.MDIntegrator.VelocityVerlet(self.forces, self.velocity, self.Mol)
        elif self.MDIntegrator == 'Langevin':
            mdInt = ChemDyME.MDIntegrator.Langevin(units.kB * self.LangTemp, self.LangFric, self.forces, self.velocity, self.Mol,timeStep)



        # Then set up reaction criteria or connectivity map
        con = ChemDyME.Connectivity.NunezMartinez(self.Mol)


        # Then set up various BXD procedures
        if self.comBXD or self.biMolecular:
            self.comBXD = True
            if self.mixedTimestep == True:
                mdInt.reset(self.timeStep * 10)
            comBxd = ChemDyME.BXDconstraint.COM(self.Mol, 0, self.minCOM, hitLimit = 100000, activeS = self.fragIdx, runType="fixed")
        if self.eneBXD:
            eneBXD = ChemDyME.BXDconstraint.Energy(self.Mol, -10000, 10000, hitLimit = 1, adapMax = self.adaptiveSteps, runType="adaptive")


        # Run MD trajectory for specified number of steps
        for i in range(0,self.mdSteps):
            t = process_time()

            try:
                self.ene = self.Mol.get_potential_energy()
            except:
                pass



            # Update the COM seperation and check whether it is bounded
            if self.comBXD:
                comBxd.update(self.Mol)
                comBounded = comBxd.inversion


            if self.comBXD and eneBXDon and i % self.printFreq == 0:
                print("Ene = " + str(eneBXD.s[0]) + ' S = ' + str(comBxd.s[0]) + ' step = ' + str(i) + ' process = ' + str(self.procNum) + ' time = ' + str(process_time()-t) + ' temperature = ' + str(self.Mol.get_temperature()))
            elif self.comBXD and i % self.printFreq == 0 :
                print("Ene = " + "NA" + ' S = ' + str(comBxd.s[0]) + ' step = ' + str(i) + ' process = ' + str(self.procNum) + ' time = ' + str(process_time()-t) + ' temperature = ' + str(self.Mol.get_temperature()))
            elif eneBXDon and i % self.printFreq == 0:
                print("Ene = " + str(self.Mol.get_potential_energy()) + ' box = ' + str(eneBXD.box) + ' step = ' + str(i) + ' process = ' + str(self.procNum) + ' time = ' + str(process_time()-t) + ' temperature = ' + str(self.Mol.get_temperature()) + ' Etot ' + str(self.Mol.get_potential_energy() + self.Mol.get_kinetic_energy()))

            if i % self.printFreq == 0 and self.geom_print is True:
                tl.printTraj(namefile, self.Mol.copy())


            #  Now check whether to turn BXDE on
            if self.comBXD:
                if (self.biMolecular or self.comBXD) and comBxd.s[0] < self.minCOM and eneBXDon == False:
                    if self.mixedTimestep == True:
                        mdInt.reset(self.timeStep)
                    eneBXDon = False
            elif self.eneBXD and self.ReactionCountDown == 0:
                eneBXDon = True

            if eneBXDon == True:
                eneBXD.update(self.Mol)
                eBounded = eneBXD.inversion

            if comBounded is True and self.ReactionCountDown == 0:
                self.Mol.set_positions(mdInt.old_positions)
                com_del_phi = comBxd.del_constraint(self.Mol)
                if comBxd.stuck == True  and comBxd.s[0] < self.minCOM:
                    comBxd.stuckFix()

            if eBounded:
                self.Mol.set_positions(mdInt.old_positions)
                try:
                    e_del_phi = eneBXD.del_constraint(self.Mol)
                except:
                    pass

            # Perform inversion if required
            if eBounded is True and comBounded is True:
                mdInt.constrain2(e_del_phi, com_del_phi)
            elif eBounded is False and comBounded is True:
                mdInt.constrain(com_del_phi)
            elif eBounded is True and comBounded is False:
                mdInt.constrain(e_del_phi)
                timeStep = self.smallStep
            else:
                self.numberOfSteps += 1
                timeStep = self.timeStep


            mdInt.md_step_pos(self.forces, timeStep, self.Mol)
            try:
                self.forces = self.Mol.get_forces()
            except:
                try:
                    self.forces = self.Mol.get_forces()
                except:
                    print("forces error")
                    self.forces = np.zeros(len(self.forces.shape))

            mdInt.md_step_vel(self.forces, timeStep, self.Mol)

            self.MolList.append(self.Mol.copy())

            # Update connectivity map to check for reaction
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
                if consistantChange > 0:
                    consistantChange = 0
                con.criteriaMet = False


            if not self.ReactionCountDown == 0:
                self.ReactionCountDown -= 1
            if self.ReactionCountDown == 1:
                pmol = self.Mol.copy()
                pmol = tl.setCalc(pmol, 'Traj_' + str(self.procNum), self.method, self.level)
                prod_smile = tl.getSMILES(pmol, True, True)
                if self.endOnReac is True and prod_smile != reac_smile:
                    self.ReactionCountDown = 0
                    self.productGeom = self.Mol.get_positions()
                    os.chdir(workingDir)
                    break
                elif fails < 10:
                    self.ReactionCountDown = self.window
                    fails +=1
                else:
                    fails = 0
                    self.ReactionCountDown = 0
                    consistantChange = 0
        namefile.close()
        os.chdir(workingDir)


    def runBXDEconvergence(self, maxHits,maxAdapSteps,eneAdaptive, decorrelationSteps, histogramLevel, runsThough, numberOfBoxes, grainSize):

        # Create specific directory
        keepGoing = True

        freeFly = False

        workingDir = os.getcwd()
        file = open("geo.xyz","w")

        # Get potential type
        if (self.method == 'nwchem'):
            self.Mol =tl.setCalc(self.Mol,'calcMopac' + str(self.procNum) + '/Traj', 'nwchem2', self.level)
            self.Mol.get_forces()


        #Get MDintegrator type
        if self.MDIntegrator == 'VelocityVerlet':
            mdInt = ChemDyME.MDIntegrator.VelocityVerlet(self.forces, self.velocity, self.Mol)
        elif self.MDIntegrator == 'Langevin':
            mdInt = ChemDyME.MDIntegrator.Langevin(units.kB * self.LangTemp, self.LangFric, self.forces, self.velocity, self.Mol, self.timeStep)

        if (eneAdaptive == False):
            BXD = ChemDyME.BXDconstraint.Energy(self.Mol, -10000, 10000, runType="fixed", numberOfBoxes = numberOfBoxes, hitLimit = maxHits )
            BXD.createFixedBoxes(grainSize)
        else:
            BXD = ChemDyME.BXDconstraint.Energy(self.Mol, -10000, 10000, hitLimit = 1, adapMax = maxAdapSteps, runType="adaptive", numberOfBoxes = numberOfBoxes, decorrelationSteps = decorrelationSteps, hist = histogramLevel)

        #Check whether a list of bounds is present? If so read adaptive boundaries from previous run
        if os.path.isfile("BXDbounds565.txt"):
            BXD.readExisitingBoundaries("BXDbounds.txt")
            BXD.runType = 'fixed'
        else:
            BXDfile = open("BXDbounds.txt", "w")

        # Get forces from designated potential
        try:
            self.forces = self.Mol.get_forces()
        except:
            print('forces error')


        # Get potential type
        if (self.method == 'nwchem'):
            self.Mol =tl.setCalc(self.Mol,'calcMopac' + str(self.procNum) + '/Traj', 'nwchem2', self.level)
            self.Mol.get_forces()
        else:
            self.Mol =tl.setCalc(self.Mol,'calcMopac' + str(self.procNum) + '/Traj', self.method, self.level)

        #Get MDintegrator type
        if self.MDIntegrator == 'VelocityVerlet':
            mdInt = ChemDyME.MDIntegrator.VelocityVerlet(self.forces, self.velocity, self.Mol)
        elif self.MDIntegrator == 'Langevin':
            mdInt = ChemDyME.MDIntegrator.Langevin(units.kB * self.LangTemp, self.LangFric, self.forces, self.velocity, self.Mol, self.timeStep)

        # Run MD trajectory for specified number of steps
        while keepGoing:

            if not freeFly:
                BXD.update(self.Mol)
                eBounded = BXD.inversion
                if eBounded:
                    self.Mol.set_positions(mdInt.oldPos)
                    e_del_phi = BXD.del_constraint(self.Mol)
                print('S ' + str(BXD.s[0])  + ' box ' + str(BXD.box) + ' temperature ' + str(self.Mol.get_temperature()))
            else:
                eBounded = False

            # Perform inversion if required
            if eBounded is True:
                mdInt.constrain(e_del_phi)


            mdInt.mdStepPos(self.forces, self.timeStep, self.Mol)
            t = process_time()
            # Get forces and energy from designated potential
            try:
                self.forces = self.Mol.get_forces()
            except:
                print('forces error')
            self.numberOfSteps += 1
            if freeFly:
                self.forces = np.zeros((self.forces.shape))
            mdInt.mdStepVel(self.forces, self.timeStep, self.Mol)

            if BXD.stuck:
                freeFly = True

            try:
                ene = self.Mol.get_potential_energy()
            except:
                pass

            if not BXD.boxList[BXD.box].lower.hit( BXD.s, 'down') and not BXD.boxList[BXD.box].upper.hit(BXD.s,'up'):
                freeFly = False
            if BXD.completeRuns == 1:
                keepGoing = False

        if type == "adaptive":
            keepGoing = True
            BXD.printBounds(BXDfile)
            BXD.reset("fixed", maxHits)

        while keepGoing:
            t = process_time()
            # Get forces and energy from designated potential
            try:
                self.forces = self.Mol.get_forces()
            except:
                print('forces error')

            self.ene = self.Mol.get_potential_energy()

            if not freeFly:
                BXD.update(self.Mol)
                eBounded = BXD.inversion
                e_del_phi = BXD.del_phi
                print('S ' + str(self.ene)  + ' box ' + str(BXD.box) + ' time ' + str(process_time()-t) + ' temperature ' + str(self.Mol.get_temperature()))
            else:
                eBounded = False

            # Perform inversion if required

            if eBounded is True:
                mdInt.constrain(e_del_phi)

            mdInt.mdStep(self.forces, self.timeStep, self.Mol)
            self.numberOfSteps += 1

            if BXD.stuckCount > 1 :
                freeFly = True

            if not BXD.boxList[BXD.box].lower.hit( self.ene, 'down') and not self.ene < BXD.boxList[BXD.box].upper.hit(self.ene,'up'):
                freeFly = False
            if BXD.completeRuns == runsThough:
                keepGoing = False

        BXD.gatherData(False,units.kB * self.LangTemp)

    def runGenBXD(self, Reac, Prod, maxHits, adapMax, pathType, path, bonds, decSteps, histogramBins, pathLength, fixToPath, pathDistCutOff, epsilon):

        self.iterations = 0

        keepGoing = True

        workingDir = os.getcwd()
        file = open("geo.xyz","w")
        sfile = open("plotData.txt", "w")


        # Get potential type
        if (self.method == 'nwchem'):
            self.Mol =tl.setCalc(self.Mol,'calcMopac' + str(self.procNum) + '/Traj', 'nwchem2', self.level)
            self.Mol.get_forces()


        #Get MDintegrator type
        if self.MDIntegrator == 'VelocityVerlet':
            mdInt = ChemDyME.MDIntegrator.VelocityVerlet(self.forces, self.velocity, self.Mol)
        elif self.MDIntegrator == 'Langevin':
            mdInt = ChemDyME.MDIntegrator.Langevin(units.kB * self.LangTemp, self.LangFric, self.forces, self.velocity, self.Mol, self.timeStep)

        #Check whether a list of bounds is present? If so read adaptive boundaries from previous run
        BXD = ChemDyME.BXDconstraint.genBXD(self.Mol, Reac, Prod, adapMax = adapMax, activeS = bonds, path = path, pathType = pathType, decorrelationSteps = decSteps, runType = 'adaptive', hitLimit = 1, hist = histogramBins, endDistance=pathLength, fixToPath=fixToPath, pathDistCutOff=pathDistCutOff, epsilon=epsilon )
        #Check whether a list of bounds is present? If so read adaptive boundaries from previous run
        if os.path.isfile("BXDbounds.txt"):
            BXD.readExisitingBoundaries("BXDbounds.txt")
            BXD.reset("fixed", maxHits)

        else:
            BXDfile = open("BXDbounds.txt", "w")

        # Get forces from designated potential
        try:
            self.forces = self.Mol.get_forces()
        except:
            print('forces error')

        # Run MD trajectory for specified number of steps
        while keepGoing:

            t = process_time()
            bxdt = process_time()
            BXD.update(self.Mol)
            bxdte = process_time()
            eBounded = BXD.inversion
            if eBounded:
                self.Mol.set_positions(mdInt.oldPos)
                bxd_del_phi = BXD.del_constraint(self.Mol)

            if self.MDIntegrator == 'VelocityVerlet' and not eBounded:
                print('S ' + str(BXD.s[2])  + ' box ' + str(BXD.box) + ' time ' + str(process_time()-t) + ' Etot ' + str(self.Mol.get_potential_energy() + self.Mol.get_kinetic_energy()))
            elif self.iterations % self.printFreq == 0 and not eBounded:
                print('pathNode = ' +str(BXD.pathNode) + ' distFromPath = ' + str(BXD.distanceToPath) + ' project = ' +str(BXD.s[2]) + ' S ' + str(BXD.s[0]) + " hits " + str(BXD.boxList[BXD.box].upper.hits) + ' ' + str(BXD.boxList[BXD.box].lower.hits) + " points in box " + str(len(BXD.boxList[BXD.box].data))  + ' box ' + str(BXD.box) + ' time ' + str(process_time()-t) + ' temperature ' + str(self.Mol.get_temperature()))

            if self.iterations % (self.printFreq/100) == 0:
                sfile.write('S \t=\t' + str(BXD.s[0]) + '\tbox\t=\t' + str(BXD.box) + " hits " + str(BXD.boxList[BXD.box].upper.hits) + ' ' + str(BXD.boxList[BXD.box].lower.hits) + "\n")
                sfile.flush()

            # Perform inversion if required
            if eBounded is True:
                mdInt.constrain(bxd_del_phi)

            mdpt = process_time()
            mdInt.mdStepPos(self.forces, self.timeStep, self.Mol)
            mdpte = process_time()
            # Get forces from designated potential
            ft = process_time()
            try:
                self.forces = self.Mol.get_forces()
            except:
                print('forces error')
            fte = process_time()
            mdvt = process_time()
            mdInt.mdStepVel(self.forces, self.timeStep, self.Mol)
            mdvte = process_time()

            self.numberOfSteps += 1

            if self.iterations % self.printFreq == 0:
                tl.printTraj(file,self.Mol)
            self.iterations += 1

            #print( "Forces time = " + str(fte - ft) + " BXDtime = " + str(bxdte - bxdt) + " MDP time = " + str(mdpte - mdpt) + "MDV time = " + str(mdvte - mdvt))

            #check if one full run is complete, if so stop the adaptive search
            if BXD.completeRuns == 1:
                keepGoing = False

        #Check whether inital run and adaptive box placement run, save boundaries and rest all bxd boundaries ready for convergence
        if BXD.runType == "adaptive":
            # reset all box data other than boundary positions and prepare to run full BXD
            # print boundaries to file
            keepGoing = True
            BXD.printBounds(BXDfile)
            BXD.reset("fixed", maxHits)

        file.close()

        # Rerun BXD with new adaptively set bounds to converge box to box statistics
        # Get forces and energy from designated potential
        try:
            self.forces = self.Mol.get_forces()
        except:
            print('forces error')

        while keepGoing:
            t = process_time()

            BXD.update(self.Mol)
            eBounded = BXD.inversion
            if eBounded:
                self.Mol.set_positions(mdInt.oldPos)

            eBounded = BXD.inversion
            if self.MDIntegrator == 'VelocityVerlet' and self.iterations % self.printFreq == 0:
                print('S ' + str(BXD.s[2])  + ' box ' + str(BXD.box) + ' time ' + str(process_time()-t) + ' Epot ' + str(self.Mol.get_potential_energy() + self.Mol.get_kinetic_energy()))
            elif self.iterations % self.printFreq == 0:
                print('pathNode = ' +str(BXD.pathNode) + ' project = ' +str(BXD.s[2]) + ' S ' + str(BXD.s[0]) + " hits " + str(BXD.boxList[BXD.box].upper.hits) + ' ' + str(BXD.boxList[BXD.box].lower.hits) + " points in box " + str(len(BXD.boxList[BXD.box].data))  + ' box ' + str(BXD.box) + ' time ' + str(process_time()-t) + ' temperature ' + str(self.Mol.get_temperature()))

            # Perform inversion if required
            if eBounded is True:
                mdInt.constrain(BXD.del_phi)

            mdInt.mdStepPos(self.forces, self.timeStep, self.Mol)
            # Get forces from designated potential
            try:
                self.forces = self.Mol.get_forces()
            except:
                print('forces error')
            mdInt.mdStepVel(self.forces, self.timeStep, self.Mol)

            self.numberOfSteps += 1

            if BXD.completeRuns == 1:
                keepGoing = False

        #Look for exsisting results files from previous runs and open a new results file numbered sequentially from those that exsist already
        i = 0
        while(os.path.isfile("BXDprofile" + str(i) + ".txt")):
            i += 1
        BXDprofile = open("BXDprofile" + str(i) + ".txt", "w")
        BXDrawData = open("BXDrawData" + str(i) + ".txt", "w")
        BXDlowRes = open("BXDlowResProfile" + str(i) + ".txt" , "w")
        BXD.gatherData(BXDprofile, BXDrawData, self.LangTemp, BXDlowRes)
