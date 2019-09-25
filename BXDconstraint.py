from abc import ABCMeta, abstractmethod
import ConnectTools  as ct
import numpy as np


# Class to track constraints and calculate required derivatives for BXD constraints
# Inversion procedure occurs in MDintegrator class
class Constraint:
    def __init__(self, mol, start,  end, hitLimit = 100, adapMax = 100, activeS = [], topBox = 500, hist = 1, decorrelationSteps = 10, path = 0, pathType = 'linear',runType = 'adaptive', stuckLimit = 20, numberOfBoxes = 10000, endType = 'RMSD', endDistance = 100000000, fixToPath = False, pathDistCutOff = 100, epsilon = 0.95 ):
        self.decorrelationSteps = decorrelationSteps
        self.pathStuckCountdown = 0
        self.boundFile = open("bounds.txt","w")
        self.adaptiveEnd = False
        self.fixToPath = fixToPath
        self.distanceToPath = 0
        self.stepsSinceAnyBoundaryHit = 0
        self.pathDistCutOff = pathDistCutOff
        self.endType = endType
        self.endDistance = endDistance
        self.runType = runType
        self.stuckLimit = stuckLimit
        if (self.runType == "adaptive"):
            self.adaptive = True
        else:
            self.adaptive = False
        self.epsilon = epsilon
        self.mol = mol
        self.histogramSize = int(adapMax / hist)
        self.adapMax = adapMax
        self.del_phi = 0
        self.reverse = False
        self.boxList = []
        self.box = 0
        self.hitLimit = hitLimit
        self.start, self.end = start, end
        self.completeRuns = 0
        self.activeS = activeS
        self.oldS = 0
        self.s = 0
        self.oldMol = 0
        self.oldDistanceToPath = 0
        self.inversion = False
        self.stuckCount = 0
        self.stuck = False
        self.path = path
        self.boundHit = "none"
        self.pathType = pathType
        self.sInd = 0
        self.reachedEnd = True
        self.pathNode = False
        self.numberOfBoxes = numberOfBoxes
        if self.__class__.__name__ == 'genBXD':
            start,end = self.convertCartToBound(self.start,self.end)
        else:
            start,end = self.convertStoBound(self.start,self.end)
        self.boxList.append(self.getDefaultBox(start, end))
        self.totalGibbs = 0

    def createFixedBoxes(self, grainSize):
        lower = self.end
        for i in range(0,self.numberOfBoxes):
            upper = lower + grainSize
            start,end = self.convertStoBound(lower,upper)
            self.boxList.append(self.getDefaultBox(start, end))
            lower = upper


    def reset(self, type, hitLimit):
        self.box = 0
        self.runType = type
        self.adaptive = False
        self.hitLimit = hitLimit
        self.completeRuns = 0
        self.totalGibbs = 0
        self.reverse = False
        self.numberOfBoxes = len(self.boxList)
        for box in self.boxList:
            box.type = 'fixed'
            box.data = []
            box.upper.rates = []
            box.upper.stepsSinceLastHit = 0
            box.upper.transparent = False
            box.upper.stuckCount = 0
            box.upper.invisible = False
            box.lower.rates = []
            box.lower.transparent = False
            box.lower.stepsSinceLastHit = 0
            box.lower.stuckCount = 0
            box.lower.invisible = False

    def printBounds(self, file):
        file.write("BXD boundary list \n\n")
        file.write("Boundary\t" + str(0) + "\tD\t=\t" + str(self.boxList[0].lower.D) + "\tn\t=\t" + str(self.boxList[0].lower.norm) + "\n" )
        for i in range(0, len(self.boxList)):
            file.write("Boundary\t" + str(i+1) + "\tD\t=\t" + str(self.boxList[i].upper.D) + "\tn\t=\t" + str(self.boxList[i].upper.norm) + "\tS\t=\t" + str(self.boxList[i].upper.Spoint) + "\n" )
        file.close()



    def readExisitingBoundaries(self,file):
        self.runType = "fixed"
        lines = open(file,"r").readlines()
        for i in range(2, len(lines)-1):
            words = lines[i].split("\t")
            dLower = (float(words[4]))
            nL = (words[7]).strip("[]\n")
            normLower = (nL.split())
            for l in range(0,len(normLower)):
                normLower[l] = float(normLower[l])
            lowerBound = bxdBound(normLower,dLower)
            words = lines[i+1].split("\t")
            dUpper = (float(words[4]))
            nU = (words[7]).strip("[]\n")
            normUpper = (nU.split())
            for l2 in range(0,len(normUpper)):
                normUpper[l2] = float(normUpper[l2])
            upperBound = bxdBound(normUpper,dUpper)
            box = bxdBox(lowerBound,upperBound, "fixed", True)
            box.type = "fixed"
            self.boxList.append(box)
        self.boxList.pop(0)

    def gatherData(self, path, rawPath, T,lowResPath):
        # Multiply T by the gas constant in kJ/mol
        T *= (8.314 / 1000)
        profile = []
        totalProb = 0
        i = 0
        for box in self.boxList:
            rawPath.write(
                "box " + str(i) + " Steps in box = " + str(len(box.data)) + " Hits at lower boundary = " + str(
                    box.lower.hits) + " Hits at upper boundary = " + str(box.upper.hits) + "\n")
            rawPath.write("box " + str(i) + " Histogram " + "\n")
            s, dens = box.getFullHistogram()
            for j in range(0, len(dens)):
                rawPath.write("\t" + "S =" + str(s[j + 1]) + " density " + str(dens[j]) + "\n")
            boxfile = open("box" + str(i), "w")
            data = [d[1] for d in box.data]
            for d in data:
                boxfile.write(str(d) + "\n")
            boxfile.close()

        rawPath.close()
        for box in self.boxList:
            box.upper.averageRate = box.upper.hits / len(box.data)
            try:
                box.lower.averageRate = box.lower.hits / len(box.data)
            except:
                box.lower.averageRate = 0
        for i in range(0, len(self.boxList) - 1):
            if i == 0:
                self.boxList[i].Gibbs = 0
            Keq = self.boxList[i].upper.averageRate / self.boxList[i + 1].lower.averageRate
            deltaG = -1.0 * np.log(Keq) * T
            self.boxList[i + 1].Gibbs = deltaG + self.boxList[i].Gibbs


        for i in range(0, len(self.boxList)):
            self.boxList[i].eqPopulation = np.exp(-1.0 * (self.boxList[i].Gibbs / T))
            totalProb += self.boxList[i].eqPopulation

        for i in range(0, len(self.boxList)):
            self.boxList[i].eqPopulation /= totalProb

        lastS = 0
        for i in range(0, len(self.boxList)):
            s, dens = self.boxList[i].getFullHistogram()
            width = s[1] - s[0]
            lowResPath.write("S = " + str(lastS) + " G = " + str(self.boxList[i].Gibbs) + "\n")
            for j in range(0, len(dens)):
                d = float(dens[j]) / float(len(self.boxList[i].data))
                p = d * self.boxList[i].eqPopulation
                mainp = -1.0 * np.log(p / width) * T
                altp = -1.0 * np.log(p) * T
                s_path = s[j] + lastS - (width / 2.0)
                profile.append((s_path, p))
                path.write("S = " + str(s_path) + " G = " + str(mainp) + " altG = " + str(altp) + "\n")
            lastS += s[len(s) - 1] - width / 2.0

    @abstractmethod
    def update(self, mol):
        #update current and previous s(r) values
        self.s = self.getS(mol)
        self.inversion = False
        self.boundHit = "none"

        #Countdown to stop getting stuck at a boundary
        if self.pathStuckCountdown > 0:
            self.pathStuckCountdown -= 1

        # If you are doing and energy convergence run and you get stuck in a well this code allows escape
        if self.__class__.__name__ == 'Energy' and self.reverse and len(self.boxList[self.box].data) > ((self.completeRuns+1) * 5000):
            self.reverse = False
            self.completeRuns += 1


        if self.adaptive and self.s[2] > self.endDistance and self.boxList[self.box].type != "adap" and self.reverse == False:
            self.reverse = True
            self.boxList[self.box].type = "adap"
            del self.boxList[-1]
            self.boxList[self.box].data = []
            self.boxList[self.box].upper.transparent = False

        # Check whether we are in an adaptive sampling regime.
        # If so updateBoxAdap checks current number of samples and controls new box placement if neccessary
        if self.boxList[self.box].type == "adap":
            if self.box == self.numberOfBoxes:
                self.reverse = True
                self.boxList[self.box].type = "fixed"
            else:
                self.updateBoxAdap()

        if self.adaptive and self.boxList[self.box].type == "normal" and len(self.boxList[self.box].data) > 2 * self.adapMax:
            print('reassigning boundary')
            self.reassignBoundary()

        # Check if box is shrinking in order to pull two fragments together
        if self.boxList[self.box].type == "shrinking":
            if self.oldS == 0:
                self.oldS = (self.s[0]+1,self.s[1]+1,self.s[2]+1)
            #First check whether we are inside the box
            if not self.boxList[self.box].upper.hit(self.s, "up"):
                # if so box type is now fixed
                self.boxList[self.box].type = "fixed"
                self.boxList[self.box].active = "True"
            # If we are outside the box and moving further away then invert velocities
            elif self.s[0] > self.oldS[0]:
                self.inversion = True
            else:
                self.inversion = False

        if self.boxList[self.box].active and self.inversion == False:
            #Check whether boundary has been hit and whether an inversion is required
            self.inversion = self.boundaryCheck(mol)
            self.updateBounds()

        if self.inversion:
            self.stuckCount += 1
            self.stepsSinceAnyBoundaryHit = 0
        else:
            self.stepsSinceAnyBoundaryHit += 1
            self.stuckCount = 0
            self.stuck  = False
            self.oldS = self.s
            self.boxList[self.box].upper.stepSinceHit += 1
            self.boxList[self.box].lower.stepSinceHit += 1
            if (self.s[1] >= 0 or not self.__class__.__name__ == 'genBXD') and (self.pathNode is False or self.distanceToPath <=self.pathDistCutOff[self.pathNode]):
                self.boxList[self.box].data.append(self.s)

        if self.stuckCount > self.stuckLimit:
            self.stuck = True
            self.stuckCount = 0


    def updateBoxAdap(self):
        # if the box is adap mode there is no upper bound, check whether adap mode should end
        if len(self.boxList[self.box].data) > self.adapMax:
            # If adaptive sampling has ended then add a boundary based up sampled data
            self.boxList[self.box].type = "normal"
            if not self.reverse:
                self.boxList[self.box].getSExtremes(self.histogramSize, self.epsilon)
                if self.box > 0:
                    bottom = self.boxList[self.box].bot
                else:
                    try:
                        bottom = self.getS(self.start)[0]
                    except:
                        bottom = self.start
                top = self.boxList[self.box].top
                if self.__class__.__name__ == 'genBXD':
                    b1 = self.convertStoBound(bottom,top)
                    b2 = self.convertStoBound(bottom,top)
                else:
                    b2,b1 = self.convertStoBound(bottom,top)
                    b2 = b1
                b3 = bxdBound(self.boxList[self.box].upper.norm,self.boxList[self.box].upper.D)
                b3.invisible = True
                self.boxList[self.box].upper = b1
                self.boxList[self.box].upper.transparent = True
                newBox = self.getDefaultBox(b2,b3)
                self.boxList.append(newBox)
            elif self.reverse and self.__class__.__name__ == 'genBXD':
                # at this point we partition the box into two and insert a new box at the correct point in the boxList
                self.boxList[self.box].getSExtremesReverse(self.histogramSize,self.epsilon)
                bottom = self.boxList[self.box].bot
                try:
                    top = self.boxList[self.box].top
                except:
                    top = self.boxList[self.box].top
                b1 = self.convertStoBound(bottom,top)
                b2 = self.convertStoBound(bottom,top)
                b3 = bxdBound(self.boxList[self.box].lower.norm,self.boxList[self.box].lower.D)
                self.boxList[self.box].lower = b1
                newBox = self.getDefaultBox(b3,b2)
                self.boxList.insert(self.box, newBox)
                self.boxList[self.box].active = True
                self.boxList[self.box].upper.transparent = False
                self.box+= 1
                self.boxList[self.box].lower.transparent = True
            else:
                self.completeRuns += 1

    @abstractmethod
    def stuckFix(self):
        pass

    def reassignBoundary(self):
        fix = self.fixToPath
        self.fixToPath = False
        if self.reverse:
            self.boxList[self.box].getSExtremesReverse(self.histogramSize, self.epsilon)
        else:
            self.boxList[self.box].getSExtremes(self.histogramSize, self.epsilon)
        bottom = self.boxList[self.box].bot
        top = self.boxList[self.box].top
        b = self.convertStoBound(bottom, top)
        b2 = self.convertStoBound(bottom, top)
        b.transparent = True
        if self.reverse:
            self.boxList[self.box].lower = b
            self.boxList[self.box -1].upper = b2
        else:
            self.boxList[self.box].upper = b
            self.boxList[self.box+1].lower = b2
        self.fixToPath = fix
        self.boxList[self.box].data = []


    def boundaryCheck(self,mol):
        if self.pathNode is False:
            distCutOff = 100
        else:
            distCutOff = self.pathDistCutOff[self.pathNode]
        if self.distanceToPath >= distCutOff and self.distanceToPath > self.oldDistanceToPath:
            self.boundHit = "path"
            self.oldDistanceToPath = self.distanceToPath
            return True
        self.oldDistanceToPath = self.distanceToPath
        #Check for hit against upper boundary
        if self.boxList[self.box].upper.hit(self.s, "up"):
            if self.boxList[self.box].upper.transparent and self.distanceToPath <= distCutOff:
                self.boxList[self.box].upper.transparent = False
                if self.adaptive:
                    self.boxList[self.box].upper.hits = 0
                    self.boxList[self.box].lower.hits = 0
                    self.boxList[self.box].data = []
                self.box += 1
                self.boxList[self.box].lower.stepSinceHit = 0
                self.boxList[self.box].upper.stepSinceHit = 0
                if self.adaptive == False and self.box == (len(self.boxList) - 1) and self.reverse == False:
                    self.reverse = True
                return False
            elif self.distanceToPath <= distCutOff:
                if self.stepsSinceAnyBoundaryHit > self.decorrelationSteps:
                    self.boxList[self.box].upper.hits += 1
                    self.boxList[self.box].upper.rates.append(self.boxList[self.box].upper.stepSinceHit)
                self.boxList[self.box].upper.stepSinceHit = 0
                if self.boxList[self.box].type == "adap" and self.reverse == False:
                    self.reverse = True
                    self.boxList[self.box].type = "adap"
                    self.boxList[self.box].data = []
                    self.boxList[self.box].lower.transparent = True
                self.boundHit = "upper"
                return True
            else:
                self.boundHit = "upper"
                return True
        elif self.boxList[self.box].lower.hit(self.s, "down"):
            if self.boxList[self.box].lower.transparent and self.distanceToPath <= distCutOff:
                self.boxList[self.box].lower.transparent = False
                if self.adaptive:
                    self.boxList[self.box].upper.hits = 0
                    self.boxList[self.box].lower.hits = 0
                    self.boxList[self.box].data = []
                self.box -=1
                self.boxList[self.box].lower.stepSinceHit = 0
                self.boxList[self.box].upper.stepSinceHit = 0
                if self.box == 0:
                    self.reverse = False
                    self.completeRuns += 1
                if self.adaptive:
                    self.boxList[self.box].type = "adap"
                    self.boxList[self.box].data = []
                    self.boxList[self.box].lower.transparent = True
                return False
            elif self.distanceToPath <= distCutOff:
                if self.stepsSinceAnyBoundaryHit > self.decorrelationSteps:
                    self.boxList[self.box].lower.hits += 1
                    self.boxList[self.box].lower.rates.append(self.boxList[self.box].lower.stepSinceHit)
                self.boxList[self.box].lower.stepSinceHit = 0
                self.boundHit = "lower"
                return True
            else:
                self.boundHit = "lower"
                return True
        else:
            return False

    #Determine whether boundary should be transparent or reflective
    def updateBounds(self):
        if ((self.reverse) and (self.criteriaMet(self.boxList[self.box].lower))):
            self.boxList[self.box].lower.transparent = True
        elif ((self.reverse == False) and (self.criteriaMet(self.boxList[self.box].upper))):
            self.boxList[self.box].upper.transparent = True


    # Check whether the criteria for box convergence has been met
    @abstractmethod
    def criteriaMet(self, boundary):
        return boundary.hits >= self.hitLimit


    @abstractmethod
    def getS(self, mol):
        pass

    @abstractmethod
    def del_constraint(self,mol):
        pass

    @abstractmethod
    def getDefaultBox(self, lower, upper):
        pass

    @abstractmethod
    def reachedTop(self):
        return False

    def convertStoBound(self, lower, upper):
        b1 = bxdBound(1,-lower)
        b2 = bxdBound(1,-upper)
        return b1, b2

class Energy(Constraint):

    

    def del_constraint(self, mol):
        self.del_phi = mol.get_forces()
        return self.del_phi


    def getS(self, mol):
        return [mol.get_potential_energy(),mol.get_potential_energy(),mol.get_potential_energy()]


    def getDefaultBox(self,lower, upper):
        if self.runType == "adaptive":
            b = bxdBox(lower,upper,"adap",True)
        else:
            b = bxdBox(lower, upper,"fixed",True)
        return b

    def reachedTop(self):
        if self.box >= self.numberOfBoxes:
            return True
        else:
            return False


    def stuckFix(self):
        self.boxList[self.box].lower.D += 0.1


    def convertStoBound(self, lower, upper):
        b1 = bxdBound(1,-lower)
        b2 = bxdBound(1,-upper)
        return b1, b2

class COM(Constraint):


    def getS(self, mol):
        self.COM = ct.getCOMdist(mol, self.activeS)
        return (self.COM,self.COM,self.COM)
    
    

    def del_constraint(self, mol):
        self.del_phi = ct.getCOMdel(mol, self.activeS)
        return self.del_phi


    def getDefaultBox(self,lower, upper):
        b = bxdBox(lower,upper,"shrinking",False)
        return b


    def stuckFix(self):
        self.boxList[self.box].upper.D -= 0.2
        self.boxList[self.box].type = "shrinking"

class genBXD(Constraint):

    def convertStoBound(self, s1, s2):
        if self.fixToPath == False:
            b = self.convertStoBoundGeneral(s1,s2)
        else:
            s1vec = ct.projectPointOnPath(s1, self.path, self.pathType,self.boxList[self.box].lower.norm,self.boxList[self.box].lower.D, self.reac, self.pathNode, self.reverse)
            s2vec = ct.projectPointOnPath(s2, self.path, self.pathType,self.boxList[self.box].lower.norm,self.boxList[self.box].lower.D, self.reac, self.pathNode, self.reverse)
            if self.reverse:
                b = self.convertStoBoundOnPath(s1vec[1],s1vec[2],s1)
            else:
                b = self.convertStoBoundOnPath(s2vec[1],s2vec[2],s2)
        return b

    def convertStoBoundGeneral(self, s1, s2):
        n2 = (s2 - s1) / np.linalg.norm(s1-s2)
        if self.reverse:
            D2 = -1*np.vdot(n2,s1)
            plength = s1
        else:
            D2 = -1*np.vdot(n2,s2)
            plength = s2
        b2 = bxdBound(n2,D2)
        b2.Spoint = plength
        #self.boundFile.write("pathNode\t=\t" + str(pathNode) + "\tcurrentNode\t=\t" +str(self.pathNode) + "Box\t=\t" + str(self.box) + "\tReverse\t=\t" + str(self.reverse) + "\tn\t=\t" + str(n) + "\tD\t=\t" + str(D) + "\tsPoint\t=\t" + str(plength) +  "\taltSPoint\t=\t" + str(s) + "\n")
        self.boundFile.flush()
        return b2

    def convertStoBoundOnPath(self, proj, pathNode, s):
        #If at last node in path then consider the line in last segment
        if pathNode == (len(self.path)-1) or pathNode == len(self.path):
            segmentStart = self.path[pathNode -1][0]
            segmentEnd = self.path[pathNode][0]
            #Total path distance up to current segment
            TotalDistance = self.path[pathNode][1]
        else:
            segmentStart = self.path[pathNode][0]
            segmentEnd = self.path[pathNode + 1][0]
            #Total path distance up to current segment
            TotalDistance = self.path[pathNode][1]
        # Projection along linear segment only
        SegDistance = proj - TotalDistance
        # Get vector for and length of linear segment
        vec = segmentEnd - segmentStart
        length = np.linalg.norm(vec)
        #finally get distance of projected point along vec
        plength = segmentStart + ((SegDistance / length) * vec)
        n = (segmentEnd - segmentStart) / np.linalg.norm(segmentEnd - segmentStart)
        D = -1 * np.vdot(n, s)
        b = bxdBound(n,D)
        b.Spoint = plength
        self.boundFile.write("pathNode\t=\t" + str(pathNode) + "\tcurrentNode\t=\t" +str(self.pathNode) + "Box\t=\t" + str(self.box) + "\tReverse\t=\t" + str(self.reverse) + "\tn\t=\t" + str(n) + "\tD\t=\t" + str(D) + "\tsPoint\t=\t" + str(plength) +  "\taltSPoint\t=\t" + str(s) + "\n")
        self.boundFile.flush()
        return b

    def getS(self, mol):
        S,Sdist,project,node,dist = self.convertToS(mol,self.activeS)
        self.pathNode = node
        self.distanceToPath = dist
        return S,Sdist,project


    def del_constraint(self, mol):
        if self.boundHit == "lower":
            n = self.boxList[self.box].lower.norm
        elif self.boundHit == "upper":
            n = self.boxList[self.box].upper.norm
        elif self.boundHit == "path":
            if self.pathNode == (len(self.path) - 1) or self.pathNode == len(self.path):
                segmentStart = self.path[self.pathNode - 1][0]
                segmentEnd = self.path[self.pathNode][0]
                # Total path distance up to current segment
                TotalDistance = self.path[self.pathNode][1]
            else:
                segmentStart = self.path[self.pathNode][0]
                segmentEnd = self.path[self.pathNode + 1][0]
                # Total path distance up to current segment
                TotalDistance = self.path[self.pathNode][1]
            # Projection along linear segment only
            SegDistance = self.s[2]- TotalDistance
            # Get vector for and length of linear segment
            vec = segmentEnd - segmentStart
            length = np.linalg.norm(vec)
            # finally get distance of projected point along vec
            plength = segmentStart + ((SegDistance / length) * vec)
            perpendicularPath = (self.s[0] - plength)
            n = perpendicularPath / np.linalg.norm(perpendicularPath)
        self.del_phi = ct.genBXDDel(mol,self.s[0],self.sInd,n)
        return self.del_phi


    def getDefaultBox(self,lower,upper):
        b = bxdBox(lower,upper,"adap",True)
        return b

    def reachedTop(self):
        if self.adaptive == False and self.box == (len(self.boxList)-1):
            return True
        elif self.adaptive == True and self.s[2] > self.endDistance:
            return True
        else:
            return False


    def convertToS(self, mol, activeS):
        # First get S
        S = ct.getDistMatrix(mol,activeS)
        if self.sInd == 0:
            self.sInd = S[1]
            self.reac = S[0]
        try:
            Snorm, project, node, dist = ct.projectPointOnPath(S[0],self.path, self.pathType,self.boxList[self.box].lower.norm,self.boxList[self.box].lower.D,self.reac, self.pathNode, self.reverse )
        except:
            Snorm = 0
            project = 0
            node = False
            dist = 0
        return S[0],Snorm, project, node, dist

    def definePath(self, path, type):
        self.path = path
        self.pathType = type

    #Function to get two boundary based upon some starting and ending configurations.
    # If you allready have the lower bound this can be adapted to return just the upper
    def convertCartToBound(self, lower, upper):
        # Get vector of colective variables for upper and lower boundary
        s1,norm,proj = self.getS(lower)
        s2,norm,proj = self.getS(upper)
        # Define the unit norm for each boundary
        n1 = (s2 - s1) / np.linalg.norm(s1-s2)
        n2 = (s2 - s1) / np.linalg.norm(s1-s2)
        # Then get D for each boundary i.e -sigma(n*s) =  D
        D1 = -1*np.vdot(n1,s1)
        D2 = -1*np.vdot(n2,s2)
        # Return bounds
        b1 = bxdBound(n1,D1)
        b2 = bxdBound(n2,D2)
        #Make bound 2 invisible from being hit
        b2.invisible = True
        return b1, b2


class bxdBox:

    def __init__(self, lower, upper, type, act):
        self.upper =  upper
        self.lower = lower
        self.type = type
        self.active = act
        #store all s values
        self.data = []
        self.topData = []
        self.top = 0
        self.botData = []
        self.bot = 0
        self.eqPopulation = 0
        self.Gibbs = 0


    def getSExtremes(self,b,eps):
        self.topData = []
        self.botData = []
        data = [d[2] for d in self.data]
        hist, edges = np.histogram(data, bins=b)
        cumProb = 0
        limit = 0
        for h in range(0, len(hist)):
            cumProb += hist[h]/len(data)
            if cumProb > eps:
                limit = h
                break
        if limit == 0:
            limit = len(data)-1
        for d in self.data:
            if d[2] > edges[limit] and d[2] <= edges[limit + 1]:
                self.topData.append(d[0])
        self.top = np.mean(self.topData,axis=0)
        for d in self.data:
            if d[2] >= edges[0] and d[2]< edges[1]:
                self.botData.append(d[0])
        self.bot = np.mean(self.botData,axis=0)

    def getSExtremesReverse(self, b, eps):
        self.topData = []
        self.botData = []
        data = [d[2] for d in self.data]
        hist, edges = np.histogram(data, bins=b)
        cumProb = 0
        limit = 0
        for h in range(0, len(hist)):
            cumProb += hist[h] / len(data)
            if cumProb > (1-eps):
                limit = h
                break
        if limit == 0:
            limit = len(data) - 1
        for d in self.data:
            if d[2] > edges[-2] and d[2] <= edges[-1]:
                self.topData.append(d[0])
        self.top = np.mean(self.topData, axis=0)
        for d in self.data:
            if d[2] >= edges[limit] and d[2] < edges[limit+1]:
                self.botData.append(d[0])
        self.bot = np.mean(self.botData, axis=0)

    def getFullHistogram(self):
        del self.data[0]
        data = [d[1] for d in self.data]
        top = max(data)
        edges = []
        for i in range(0,11):
            edges.append(i*(top/10))
        hist = np.zeros(10)
        for d in data:
            for j in range(0,10):
                if d > edges[j] and d <= edges[j+1]:
                    hist[j] += 1
        return edges, hist

class bxdBound:

    def __init__(self, norm, D):
        self.norm = norm
        self.D = D
        self.hits = 0
        self.stuckCount = 0
        self.transparent = False
        self.stepSinceHit = 0
        self.rates = []
        self.averageRate = 0
        self.rateError = 0
        self.invisible = False
        self.Spoint = 0

    def hit(self, s, bound):
        if self.invisible:
            return False
        try:
            s = s[0]
        except:
            s = s
        coord = np.vdot(s,self.norm) + self.D
        if bound == "up" and coord > 0:
            return True
        elif bound == "down" and coord < 0:
            return True
        else:
            return False

    def averageRates(self):
        self.averageRate = np.mean(self.rates)
        self.rateError = np.std(self.rates)
