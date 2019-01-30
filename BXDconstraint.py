from abc import ABCMeta, abstractmethod
import ConnectTools  as ct
import numpy as np


# Class to track constraints and calculate required derivatives for BXD constraints
# Inversion procedure occurs in MDintegrator class
class Constraint:
    @abstractmethod
    def __init__(self, mol, start,  end, hitLimit = 100, adapMax = 100, activeS = [], topBox = 500, hist = 1, decorrelationSteps = 10, path = 0, pathType = 'linear',runType = 'adaptive', stuckLimit = 5, numberOfBoxes = 10, endType = 'RMSD' ):
        self.decorrelationSteps = decorrelationSteps
        self.endType = endType
        self.runType = runType
        self.stuckLimit = stuckLimit
        if (self.runType == "adaptive"):
            self.adaptive = True
        else:
            self.adaptive = False
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
        self.inversion = False
        self.stuckCount = 0
        self.stuck = False
        self.path = path
        self.boundHit = "none"
        self.pathType = pathType
        self.sInd = 0
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
        for box in self.boxList:
            box.type = 'fixed'
            box.data = []
            box.upper.rates = []
            box.upper.stepsSinceLastHit = 0
            box.upper.transparent = False
            box.upper.stuckCount = 0
            box.lower.rates = []
            box.lower.transparent = False
            box.lower.stepsSinceLastHit = 0
            box.lower.stuckCount = 0

    def printBounds(self, file):
        file.write("BXD boundary list \n\n")
        file.write("Boundary\t" + str(0) + "\tD\t=\t" + str(self.boxList[0].lower.D) + "\tn\t=\t" + str(self.boxList[0].lower.norm) + "\n" )
        for i in range(0, len(self.boxList)):
            file.write("Boundary\t" + str(i+1) + "\tD\t=\t" + str(self.boxList[i].upper.D) + "\tn\t=\t" + str(self.boxList[i].upper.norm) + "\n" )
        file.close()



    def readExisitingBoundaries(self,file):
        self.runType = "fixed"
        lines = open(file,"r").readlines()
        for i in range(2, len(lines)-1):
            words = lines[i].split("\t")
            dLower = (float(words[4]))
            nL = (words[7]).strip("[]\n")
            normLower = (nL.split(","))
            for l in range(0,len(normLower)):
                normLower[l] = float(normLower[l])
            lowerBound = bxdBound(normLower,dLower)
            words = lines[i+1].split("\t")
            dUpper = (float(words[4]))
            nU = (words[7]).strip("[]\n")
            normUpper = (nU.split(","))
            for l2 in range(0,len(normUpper)):
                normUpper[l2] = float(normUpper[l2])
            upperBound = bxdBound(normUpper,dUpper)
            box = bxdBox(lowerBound,upperBound, "fixed", True)
            self.boxList.append(box)

    def gatherData(self, path, rawPath):
        profile = []
        totalProb = 0
        i = 0
        for box in self.boxList:
            rawPath.write( "box " + str(i) + " Steps in box = " + str(len(box.data)) + " Hits at lower boundary = " + str(box.lower.hits) + " Hits at upper boundary = " + str(box.upper.hits) + "\n")
            rawPath.write("box " + str(i) + " Histogram " + "\n")
            s,dens = self.boxList[self.box].getFullHistogram()
            for j in range(0,len(dens)):
                rawPath.write( "\t" + "S =" + str(s[j+1]) + " density " +  str(dens[j]) + "\n")
            i += 1
        rawPath.close()
        for box in self.boxList:
            box.upper.averageRate = len(box.data) / box.upper.hits
            try:
                box.lower.averageRate = len(box.data) / box.lower.hits
            except:
                box.lower.averageRate = 0
        for i in range(0,len(self.boxList)-1):
            if i == 0:
                self.boxList[i].Gibbs = 0
            deltaG = self.boxList[i].upper.averageRate / self.boxList[i+1].lower.averageRate
            deltaG = -1 * np.log(deltaG)
            self.boxList[i+1].Gibbs = deltaG + self.boxList[i].Gibbs

        for i in range(0,len(self.boxList)):
            self.boxList[i].eqPopulation =  np.exp(-(self.boxList[i].Gibbs))
            totalProb += self.boxList[i].eqPopulation

        for i in range(0,len(self.boxList)):
            self.boxList[i].eqPopulation /=  totalProb

        lastS = 0
        for i in range(0,len(self.boxList)):
            s,dens = self.boxList[self.box].getFullHistogram()
            width = s[1] - s[0]
            width2 = s[2] - s[1]
            for j in range(0,len(dens)):
                d= float(dens[j]) / float(len(self.boxList[i].data))
                p = d * self.boxList[i].eqPopulation
                p = -1.0 * np.log(p/width)
                profile.append((s,p))
                path.write( "S = " + str(s[j] + lastS) + " G = " + str(p) + "\n")
            lastS += s[len(s)-1]

    @abstractmethod
    def update(self, mol):
        #update current and previous s(r) values
        self.s = self.getS(mol)
        self.inversion = False
        self.boundHit = "none"



        # If you are doing and energy convergence run and you get stuck in a well this code allows escape
        if self.__class__.__name__ == 'Energy' and self.reverse and len(self.boxList[self.box].data) > ((self.completeRuns+1) * 5000):
            self.reverse = False
            self.completeRuns += 1

        # Check whether we are in an adaptive sampling regime.
        # If so updateBoxAdap checks current number of samples and controls new box placement if neccessary
        if self.boxList[self.box].type == "adap":
            self.updateBoxAdap()

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
        else:
            self.stuckCount = 0
            self.stuck  = False
            self.oldS = self.s
            self.boxList[self.box].upper.stepSinceHit += 1
            self.boxList[self.box].lower.stepSinceHit += 1
            if self.s[1] >= 0 or not self.__class__.__name__ == 'genBXD':
                self.boxList[self.box].data.append(self.s)

        if self.stuckCount > self.stuckLimit:
            self.stuck = True
            self.stuckCount = 0

    @abstractmethod
    def updateBoxAdap(self):
        # if the box is adap mode there is no upper bound, check whether adap mode should end
        if len(self.boxList[self.box].data) > self.adapMax:
            # If adaptive sampling has ended then add a boundary based up sampled data
            self.boxList[self.box].type = "normal"
            if not self.reverse:
                self.boxList[self.box].getSExtremes(self.histogramSize)
                if self.box > 0:
                    bottom = self.boxList[self.box].bot
                else:
                    try:
                        bottom = self.getS(self.start)[0]
                    except:
                        bottom = self.start
                top = self.boxList[self.box].top
                if self.__class__.__name__ == 'genBXD':
                    b1 = self.convertSToBound(bottom,top)
                    b2 = self.convertSToBound(bottom,top)
                else:
                    b2,b1 = self.convertSToBound(bottom,top)
                    b2 = b1
                b3 = bxdBound(self.boxList[self.box].upper.norm,self.boxList[self.box].upper.D)
                self.boxList[self.box].upper = b1
                self.boxList[self.box].upper.transparent = True
                newBox = self.getDefaultBox(b2,b3)
                self.boxList.append(newBox)
            elif self.reverse and self.__class__.__name__ == 'genBXD':
                # at this point we partition the box into two and insert a new box at the correct point in the boxList
                self.boxList[self.box].getSExtremes(self.histogramSize)
                bottom = self.boxList[self.box].bot
                try:
                    top = self.boxList[self.box].top
                except:
                    top = self.boxList[self.box].top
                b1 = self.convertSToBound(bottom,top)
                b2 = self.convertSToBound(bottom,top)
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

    @abstractmethod
    def boundaryCheck(self,mol):
        #Check for hit against upper boundary
        if self.boxList[self.box].upper.hit(self.s, "up"):
            if self.boxList[self.box].upper.transparent:
                self.boxList[self.box].upper.transparent = False
                if self.adaptive:
                    self.boxList[self.box].upper.hits = 0
                    self.boxList[self.box].lower.hits = 0
                self.box += 1
                self.boxList[self.box].lower.stepSinceHit = 0
                self.boxList[self.box].upper.stepSinceHit = 0
                if self.reachedTop() and self.reverse == False:
                    self.reverse = True
                return False
            else:
                if self.boxList[self.box].upper.stepSinceHit > self.decorrelationSteps:
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
        elif self.boxList[self.box].lower.hit(self.s, "down"):
            if self.boxList[self.box].lower.transparent:
                self.boxList[self.box].lower.transparent = False
                if self.adaptive:
                    self.boxList[self.box].upper.hits = 0
                    self.boxList[self.box].lower.hits = 0
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
            else:
                if self.boxList[self.box].lower.stepSinceHit > self.decorrelationSteps:
                    self.boxList[self.box].lower.hits += 1
                    self.boxList[self.box].lower.rates.append(self.boxList[self.box].lower.stepSinceHit)
                self.boxList[self.box].lower.stepSinceHit = 0
                self.boundHit = "lower"
                return True
        else:
            return False

    @abstractmethod
    #Determine whether boundary should be transparent or reflective
    def updateBounds(self):
        if ((self.reverse) and (self.criteriaMet(self.boxList[self.box].lower))):
            self.boxList[self.box].lower.transparent = True
        elif ((self.reverse == False) and (self.criteriaMet(self.boxList[self.box].upper))):
            self.boxList[self.box].upper.transparent = True


    @abstractmethod
    # Check whether the criteria for box convergence has been met
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

    @abstractmethod
    def convertStoBound(self, lower, upper):
        b1 = bxdBound(1,-lower)
        b2 = bxdBound(1,-upper)
        return b1, b2


class Energy(Constraint):

    
    @abstractmethod
    def del_constraint(self, mol):
        self.del_phi = mol.get_forces()
        return self.del_phi

    @abstractmethod
    def getS(self, mol):
        return [mol.get_potential_energy(),mol.get_potential_energy(),mol.get_potential_energy()]

    @abstractmethod
    def getDefaultBox(self,lower, upper):
        if self.runType == "adaptive":
            b = bxdBox(lower,upper,"adap",True)
        else:
            b = bxdBox(lower, upper,"fixed",True)
        return b

    @abstractmethod
    def reachedTop(self):
        if self.box >= self.numberOfBoxes:
            return True
        else:
            return False

    @abstractmethod
    def stuckFix(self):
        self.boxList[self.box].lower.D += 0.1

    @abstractmethod
    def convertSToBound(self, lower, upper):
        b1 = bxdBound(-1,-lower)
        b2 = bxdBound(1,-upper)
        return b1, b2

class COM(Constraint):

    @abstractmethod
    def getS(self, mol):
        self.COM = ct.getCOMdist(mol, self.activeS)
        return (self.COM,self.COM,self.COM)
    
    
    @abstractmethod
    def del_constraint(self, mol):
        self.del_phi = ct.getCOMdel(mol, self.activeS)
        return self.del_phi

    @abstractmethod
    def getDefaultBox(self,lower, upper):
        b = bxdBox(lower,upper,"shrinking",False)
        return b

    @abstractmethod
    def stuckFix(self):
        self.boxList[self.box].upper.D -= 0.2
        self.boxList[self.box].type = "shrinking"

class genBXD(Constraint):

    @abstractmethod
    def getS(self, mol):
        S,Sdist,project,node = self.convertToS(mol,self.activeS)
        self.pathNode = node
        return S,Sdist,project

    @abstractmethod
    def del_constraint(self, mol):
        if self.boundHit == "lower":
            n = self.boxList[self.box].lower.norm
        elif self.boundHit == "upper":
            n = self.boxList[self.box].upper.norm
        self.del_phi = ct.genBXDDel(mol,self.s[0],self.sInd,n)
        return self.del_phi


    @abstractmethod
    def getDefaultBox(self,lower,upper):
        b = bxdBox(lower,upper,"adap",True)
        return b

    def convertToS(self, mol, activeS):
        # First get S
        S = ct.getDistMatrix(mol,activeS)
        if self.sInd == 0:
            self.sInd = S[1]
            self.reac = S[0]
        try:
            Snorm, project, node = ct.projectPointOnPath(S[0],self.path, self.pathType,self.boxList[self.box].lower.norm,self.boxList[self.box].lower.D,self.reac, self.pathNode )
        except:
            Snorm = 0
            project = 0
            node = 0
        return S[0],Snorm, project, node

    def definePath(self, path, type):
        self.path = path
        self.pathType = type

    @abstractmethod
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
        return b1, b2

    @abstractmethod
    def convertSToBound(self, s1, s2,):
        n2 = (s2 - s1) / np.linalg.norm(s1-s2)
        if self.reverse:
            D2 = -1*np.vdot(n2,s1)
        else:
            D2 = -1*np.vdot(n2,s2)
        b2 = bxdBound(n2,D2)
        return b2

    @abstractmethod
    def reachedTop(self):
        if self.adaptive == False and self.box == (len(self.boxList)-1):
            return True
        else:
            return False

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


    def getSExtremes(self,b):
        self.topData = []
        self.botData = []
        data = [d[2] for d in self.data]
        hist, edges = np.histogram(data, bins=b)
        for d in self.data:
            if d[2] > edges[b-1]:
                self.topData.append(d[0])
        self.top = np.mean(self.topData,axis=0)
        for d in self.data:
            if d[2] >= edges[0] and d[2]< edges[1]:
                self.botData.append(d[0])
        self.bot = np.mean(self.botData,axis=0)

    def getFullHistogram(self):
        data = [d[1] for d in self.data]
        data.append(0)
        hist, edges = np.histogram(data, bins=10, density=False)
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

    def hit(self, s, bound):
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
