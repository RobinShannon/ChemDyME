import re
import numpy as np
import Tools as tl
from io import StringIO

# Class to store global variables from the input file
class Globals:

    def __init__(self, path):
        self.BiList = []
        mpath = path + '/inp.txt'

        self.InitialBi = False
        self.mixedTimestep = False

        self.RunType = "MechGen"

        self.printNEB = False
        self.checkAltProd = False

        self.eneBXD = False
        self.comBXD = False
        self.printDynPath = False
        self.printNEB = False
        self.maxAdapSteps  = 0
        self.eneBoxes = 0
        self.grainSize = 0
        self.numberOfBoxes = 10

        self.ReactIters = 1
        self.NEBrelax = False
        self.GenBXDrelax = False
        self.NEBsteps = 3
        self.printFreq = 10

        self.QTS3 = False


        # BXD defaults
        self.decorrelationSteps = 10
        self.histogramLevel = 1
        self.principalCoordinates = []

        # Open Input
        inp = open(mpath, "r")
        geom = open(mpath,"r").readlines()


        self.printString = ""

        #Generate empty list to hold mm3 types if required
        self.atomTypes = []
        self.MMfile = 'test.pdb'

        words = geom[1].split()
        # Check if the first word is Atoms, if so the starting molecule is specified in the input file
        if words[0] == "Atoms":
            self.numberOfAtoms = int(words[2])
            size = self.numberOfAtoms

            self.species = np.zeros(size, dtype='str')
            self.cartesians = np.zeros((size,3), dtype='float')
            self.masses = np.zeros((size), dtype='float')

            for i in range(3,size+3):
                # Get index for species and cartesian arrays
                j = i - 3
                words = geom[i].split()
                self.species[j] = words[0]
                self.cartesians[j][0] = float(words[1])
                self.cartesians[j][1] = float(words[2])
                self.cartesians[j][2] = float(words[3])

            for i in range (0,size):
                if self.species[i] == 'C':
                    self.masses[i] = 12.0107
                if self.species[i] == 'O':
                    self.masses[i] = 15.9994
                if self.species[i] == 'H':
                    self.masses[i] = 1.00794
                if self.species[i] == 'N':
                    self.masses[i] = 12.0107
            self.StartType = "specified"

        # Loop over lines
        for line in inp:
            #GenBXD input;
            #Start type is the starting reactant unless specified in already. StartType is a SMILES string or file path to be read in
            if re.search("Start", line):
                self.StartType = str(line.split()[2])
                self.Start = str(line.split()[3])
            # End type specifies the target product for the GenBXD run
            if re.search("End", line):
                self.EndType = str(line.split()[2])
                self.End = str(line.split()[3])
            if re.search("MMfile", line):
                self.MMfile = str(line.split()[2])
            # PathType defines the prgress along Traj. It is either simpleDistance (distance from boundary), distance (cumulative version of simple distance)
            # linear (projection onto linear path between start and end) or curve (projection onto some trajectory or path)
            if re.search("PathType", line):
                self.PathType = str(line.split()[2])
                if self.PathType == 'curve' or self.PathType == 'gates':
                    self.PathFile = str(line.split()[3])
            # Determines the collective variables for GenBXD
            if re.search("CollectiveVarType", line):
                self.CollectiveVarType = str(line.split()[2])
                if self.CollectiveVarType == 'specified':
                    uVar = line.split()[3]
                    uVar = uVar.replace("n","\n")
                    c = StringIO(uVar)
                    self.CollectiveVar = np.loadtxt(c, delimiter=',')
                elif self.CollectiveVarType == 'file':
                    i = 4
                    while i < len(line.split()):
                        array = self.readPrincipalComponents(line.split()[i], line.split()[3])
                        self.principalCoordinates.append(array)
                        i+=1
                elif self.CollectiveVarType != 'changedBonds':
                    self.CollectiveVarTraj = str(line.split()[3])



            if re.search("RunType", line):
                self.RunType = str(line.split()[2])
            if re.search("NEBrelax", line):
                self.NEBrelax = True
            if re.search("printNEB", line):
                self.printNEB = True
            if re.search("printDynPath", line):
                self.printDynPath = True
                self.dynPrintFreq = int(line.split()[1])
                self.dynPrintStart = int(line.split()[2])
            if re.search("QTS3", line):
                self.QTS3 = True
            if re.search("checkAltProd", line):
                self.checkAltProd = True
            if re.search("NEBsteps", line):
                self.NEBsteps = int(line.split()[2])
            if re.search("name", line):
                self.dirName = line.split()[2]
            if re.search("cores", line):
                self.cores =int(line.split()[2])
            if re.search("ReactionIterations", line):
                self.ReactIters =int(line.split()[2])
            if re.search("trajectoryMethod1", line):
                self.trajMethod1 = line.split()[2]
                self.trajLevel1 = line.split()[3]
                self.trajMethod2 = self.trajMethod1
                self.trajLevel2 = self.trajLevel1
            if re.search("trajectoryMethod2", line):
                self.trajMethod2 = line.split()[2]
                self.trajLevel2 = line.split()[3]
            if re.search("lowLevel", line):
                self.lowerMethod = line.split()[2]
                self.lowerLevel = line.split()[3]
            if re.search("highLevel", line):
                self.higherMethod = line.split()[2]
                self.higherLevel = line.split()[3]
            if re.search("singleLevel", line):
                self.singleMethod = line.split()[2]
                self.singleLevel = line.split()[3]
            if re.search("TrajectoryInitialTemp", line):
                self.trajInitT = int(line.split()[2])
            if re.search("MDIntegrator", line):
                self.MDIntegrator = line.split()[2]
            if re.search("mdSteps", line):
                self.mdSteps = int(line.split()[2])
            if re.search("printFreq", line):
                self.printFreq = int(line.split()[2])
            if re.search("timeStep", line):
                self.timeStep = float(line.split()[2])
            if re.search("endOnReaction", line):
                self.endOnReaction = bool(line.split()[2])
            if re.search("reactionWindow", line):
                self.reactionWindow = int(line.split()[2])
            if re.search("BiReactant", line):
                self.InitialBi = True
            if re.search("LangFriction", line):
                self.LFric = float(line.split()[2])
            if re.search("LangTemperature", line):
                self.LTemp = float(line.split()[2])
            if re.search("BiList", line):
                BiListStrings = [str(line.split()[2])]
                for x in BiListStrings:
                    self.BiList.append(tl.getMolFromSmile(x))
            if re.search("BiRate", line):
                self.BiRates = [str(line.split()[2])]
            if re.search("BiTemp", line):
                self.BiTemp = float(line.split()[2])

            # BXD input
            if re.search("eneBXD", line):
                if (line.split()[2]) == 'True':
                    self.eneBXD = True
                if (line.split()[2])== 'False':
                    self.eneBXD = False
            if self.eneBXD is True:
                if re.search("eneUpper", line):
                    self.eneUpper = float(line.split()[2])
                if re.search("eneLower", line):
                    self.eneLower = float(line.split()[2])
                if re.search("eneAdaptive", line):
                    if (line.split()[2]) == 'True':
                        self.eneAdaptive = True
                    if (line.split()[2])== 'False':
                        self.eneAdaptive = False
                if re.search("eneMax", line):
                    self.eneMax = float(line.split()[2])
                if re.search('eneEvents', line):
                    self.eneEvents = int(line.split()[2])
                if re.search('eneBoxes', line):
                    self.eneBoxes = int(line.split()[2])
            if re.search('decorrelationSteps', line):
                self.decorrelationSteps = int(line.split()[2])
            if re.search('histogramLevel', line):
                self.histogramLevel = int(line.split()[2])
            if re.search("maxHits",line):
                    self.maxHits  = int(line.split()[2])
            if re.search("runsThrough",line):
                    self.runsThrough  = int(line.split()[2])
            if re.search("adaptiveSteps",line):
                    self.maxAdapSteps  = int(line.split()[2])
            if re.search("grainSize",line):
                    self.grainSize  = int(line.split()[2])
            if re.search("numberOfBoxes",line):
                    self.numberOfBoxes  = int(line.split()[2])
            if re.search("comBXD", line):
                if (line.split()[2]) == 'True':
                    self.comBXD = True
                if (line.split()[2])== 'False':
                    self.comBXD = False
            if self.comBXD is True or self.InitialBi is True:
                if re.search("minCOM", line):
                    self.minCOM = float(line.split()[2])
                if re.search("fragment_indices", line):
                    self.fragIndices = [int(line.split()[2]),int(line.split()[3])]

            # Geometry opt
            if re.search("GeomPot", line):
                self.GeomPot1 = line.split()[2]


            # Single point ene
            if re.search("singlePoint", line):
                self.singlePoint = line.split()[2]

            # Master Equation
            if re.search("maxSimulationTime", line):
                self.maxSimulationTime = float(line.split()[2])
            if re.search("equilThreshold", line):
                self.equilThreshold = int(line.split()[2])

            # Look for mm3 types
            if re.search("AtomTypes", line):
                self.atomTypes = line.split()[2]
                self.atomTypes = self.atomTypes.split(",")
                for i in range (0,len(self.atomTypes)):
                    self.atomTypes[i] = float(self.atomTypes[i])

        self.trajMethod = self.trajMethod1
        self.trajLevel = self.trajLevel1

    def readPrincipalComponents(self, inp, terms):
        #Get number of lines in file
        num_lines = sum(1 for line in open(inp))
        # size of array is min of specified lines to be read and the actual size
        size = min(num_lines,int(terms))
        array = np.zeros((size,3))
        i = 0
        with open(inp) as f:
            lines = f.read().splitlines()
        for i in range(1,size+1):
            array[i-1][0] = int(lines[i].split('\t')[1])
            array[i-1][1] = int(lines[i].split('\t')[2])
            array[i-1][2] = float(lines[i].split('\t')[0])
        return array

