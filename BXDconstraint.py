from abc import ABCMeta, abstractmethod
import ConnectTools  as ct
import numpy as np


# Class to track constraints and calculate required derivatives for BXD constraints
# Inversion procedure occurs in MDintegrator class
class Constraint:
    @abstractmethod
    def __init__(self, mol, sampling, start,  end, hitLimit, adapMax, Idx):
        self.runType = sampling
        if (sampling == "adaptive"):
            self.adaptive = True
        else:
            self.adaptive = False
        self.mol = mol
        self.adapMax = adapMax
        self.del_phi = 0
        self.reverse = False
        self.boxList = []
        self.box = 0
        self.hitLimit = hitLimit
        self.start, self.end = start, end
        self.completeRuns = 0
        self.activeS = Idx
        self.oldS = 0
        self.s = 0
        self.inversion = False
        self.stuckCount = 0
        self.path = 0
        self.pathType = "linear"
        start,end = self.convertStoBound(self.start,self.end)
        self.boxList.append(self.getDefaultBox(start, end))
        self.totalGibbs = 0

    def createFixedBoxes(self, grainSize):
        lower = self.end
        for i in range(0,self.adapMax):
            upper = lower + grainSize
            start,end = self.convertStoBound(lower,upper)
            self.boxList.append(self.getDefaultBox(start, end))
            lower = upper


    def reset(self, type, hitLimit):
        self.runType = type
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

    def gatherData(self, print, T):
        profile = []
        deltaG = 0
        self.totalGibbs = 1
        for box in self.boxList:
            box.upper.averageRates()
            box.lower.averageRates()
        for i in range(0,len(self.boxList)-1):
            if i == 0:
                self.boxList[i].Gibbs = 1
            deltaG =  self.boxList[i].upper.averageRate / self.boxList[i+1].lower.averageRate
            deltaG = -1 * np.log(deltaG) * T
            self.boxList[i+1].Gibbs = deltaG + self.boxList[i].Gibbs
            self.totalGibbs += -self.boxList[i+1].Gibbs/ T

        for i in range(0,len(self.boxList)):
            self.boxList[i].eqPopulation = (1/(np.exp(self.totalGibbs))) * np.exp(-(self.boxList[i].Gibbs / T))

        for i in range(0,len(self.boxList)):
            s,dens = self.boxList[i].getFullHistogram()
            for j in range(0,len(s)):
                profile.append((s,dens*self.boxList[i].eqPopulation))
                print( "S = " + str(s[j]) + " G = " + str(dens[j]*self.boxList[i].eqPopulation))

    @abstractmethod
    def update(self, mol):
        #update current and previous s(r) values
        self.s = self.getS(mol)

        self.inversion = False

        #First do stuff associated with storing data etc
        self.boxList[self.box].data.append(self.s)

        if self.__class__.__name__ == 'Energy' and self.reverse and len(self.boxList[self.box].data) > ((self.completeRuns+1) * 5000):
            self.reverse = False
            self.completeRuns += 1

        # Then update the current box depending upon the run type
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
            elif self.s[0] > self.oldS[0]:
                self.inversion = True
                self.del_constraint(mol,self.boxList[self.box].upper.norm)
            else:
                self.inversion = False

        if self.boxList[self.box].active and self.inversion == False:
            #Check whether boundary has been hit and whether an inversion is required
            self.inversion = self.boundaryCheck(mol)
            self.updateBounds()

        if self.inversion:
            self.s = self.oldS
            self.stuckCount += 1
        else:
            self.stuckCount = 0
            self.boxList[self.box].upper.stepSinceHit += 1
            self.boxList[self.box].lower.stepSinceHit += 1

        if self.stuckCount > 5:
            self.stuck()

        self.oldS = self.s

    @abstractmethod
    def updateBoxAdap(self):
        # if the box is adap mode there is no upper bound, check whether adap mode should end
        if len(self.boxList[self.box].data) > self.adapMax:
            # If adaptive sampling has ended then add a boundary based up sampled data
            self.boxList[self.box].type = "normal"
            if not self.reverse:
                self.boxList[self.box].getSExtremes()
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
                self.boxList[self.box].getSExtremes()
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
    def stuck(self):
        pass

    @abstractmethod
    def boundaryCheck(self,mol):
        #Check for hit against upper boundary
        if self.boxList[self.box].upper.hit(self.s, "up"):
            if self.boxList[self.box].upper.transparent:
                self.boxList[self.box].upper.transparent = False
                self.boxList[self.box].upper.hits = 0
                self.boxList[self.box].lower.hits = 0
                self.box += 1
                self.boxList[self.box].lower.stepSinceHit = 0
                self.boxList[self.box].upper.stepSinceHit = 0
                if self.reachedTop() and self.reverse == False:
                    self.reverse = True
                return False
            else:
                if self.stuckCount == 0:
                    self.boxList[self.box].upper.hits += 1
                    self.boxList[self.box].upper.rates.append(self.boxList[self.box].upper.stepSinceHit)
                    self.boxList[self.box].upper.stepSinceHit = 0
                self.del_constraint(mol,self.boxList[self.box].upper.norm)
                if self.boxList[self.box].type == "adap" and self.reverse == False:
                    self.reverse = True
                    self.boxList[self.box].type = "adap"
                    self.boxList[self.box].data = []
                    self.boxList[self.box].lower.transparent = True
                return True
        elif self.boxList[self.box].lower.hit(self.s, "down"):
            if self.boxList[self.box].lower.transparent:
                self.boxList[self.box].lower.transparent = False
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
                if self.stuckCount == 0:
                    self.boxList[self.box].lower.hits += 1
                    self.boxList[self.box].lower.rates.append(self.boxList[self.box].lower.stepSinceHit)
                    self.boxList[self.box].lower.stepSinceHit = 0
                self.del_constraint(mol,self.boxList[self.box].lower.norm)
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
    def del_constraint(self,mol, n):
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
    def del_constraint(self, forces,n):
        self.del_phi = forces.get_forces()
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
        if self.box >= self.activeS:
            return True
        else:
            return False

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
    def del_constraint(self, mol, n):
        self.del_phi = ct.getCOMdel(mol, self.activeS)
        return self.del_phi

    @abstractmethod
    def getDefaultBox(self,lower, upper):
        b = bxdBox(lower,upper,"shrinking",False)
        return b

    @abstractmethod
    def stuck(self):
        self.boxList[self.box].upper.D += 0.1

class genBXD(Constraint):

    @abstractmethod
    def __init__(self, mol, sampling, start,  end, hitLimit, adapMax, Idx, path, pathType):
        self.runType = sampling
        if (sampling == "adaptive"):
            self.adaptive = True
        else:
            self.adaptive = False
        self.mol = mol
        self.adapMax = adapMax
        self.del_phi = 0
        self.reverse = False
        self.boxList = []
        self.box = 0
        self.hitLimit = hitLimit
        self.start, self.end = start, end
        self.completeRuns = 0
        self.activeS = Idx
        self.oldS = 0
        self.s = 0
        self.inversion = False
        self.stuckCount = 0
        self.path = path
        self.pathType = pathType
        self.sInd = 0
        start,end = self.convertCartToBound(self.start,self.end)
        self.boxList.append(self.getDefaultBox(start, end))

    @abstractmethod
    def getS(self, mol):
        S,Sdist,project = self.convertToS(mol,self.activeS)
        return S,Sdist,project

    @abstractmethod
    def del_constraint(self, mol,n):
        self.del_phi = ct.genBXDDel(mol,self.s[0],self.sInd,n)


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
            Snorm, project = ct.projectPointOnPath(S[0],self.path, self.pathType,self.boxList[self.box].lower.norm,self.boxList[self.box].lower.D,self.reac )
        except:
            Snorm = 0
            project = 0
        return S[0],Snorm, project

    def definePath(self, path, type):
        self.path = path
        self.pathType = type

    @abstractmethod
    #Function to get two boundary based upon some starting and ending configurations.
    # If you allready have the lower bound this can be adapted to return just the upper
    def convertCartToBound(self, lower, upper):
        # Get vector of coleective variables for upper and lower boundary
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


    def getSExtremes(self):
        self.topData = []
        self.botData = []
        data = [d[2] for d in self.data]
        hist, edges = np.histogram(data, bins=100)
        for d in self.data:
            if d[2] > edges[99] and d[2] <= edges[100]:
                self.topData.append(d[0])
        self.top = np.mean(self.topData,axis=0)
        for d in self.data:
            if d[2] >= edges[0] and d[2]< edges[1]:
                self.botData.append(d[0])
        self.bot = np.mean(self.botData,axis=0)

    def getFullHistogram(self):
        data = [d[2] for d in self.data]
        hist, edges = np.histogram(data, bins=10, density=True)
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
