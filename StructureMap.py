from abc import ABCMeta, abstractmethod
import ConnectTools as ct
import numpy as np

# Class to track constraints and calculate required derivatives
# Inversion procedure occurs in MDintegrator class
class StructureMap:

    def __init__(self, species, xyz):
        self.species = species
        # Get reference bonds
        self.dRef = ct.refBonds(species)
        self.dratio = self.dRef
        # Create bonding matrix
        self.bondMat = ct.bondMatrix(self.species, self.dRef, xyz, self.dratio)
        self.criteriaMet = False
        self.breakingBond = (0,0)
        self.makingBond = (0,0)
        self.bondMatTemp = self.bondMat
        self.secondCriteriaMet = False
        self.secondBreakingBond = (0,0)
        self.secondMakingBond = (0,0)

    def reinitialise(self, xyz):
        self.bondMat = ct.bondMatrix(self.species, self.dRef, xyz)
        self.criteriaMet = False
        self.breakingBond = (0,0)
        self.makingBond = (0,0)
        self.secondCriteriaMet = False

    def ReactionType(self, xyz):
        oldbonds = np.count_nonzero(self.bondMat)
        self.bondMat = ct.bondMatrix(self.species,self.dref, xyz)
        newbonds = np.count_nonzero(self.bondMat)
        if oldbonds > newbonds:
            reacType = 'Dissociation'
        if oldbonds < newbonds:
            reacType = 'Association'
        if oldbonds == newbonds:
            reacType = 'Isomerisation'
        return reacType

    @abstractmethod
    def Update(self, xyz):
        pass


class NunezMartinez(StructureMap):

    @abstractmethod
    def Update(self, xyz):
        self.bondMatTemp = ct.bondMatrix(self.species, self.dRef, xyz)
        atom1 = 0
        atom2 = 0
        atom3 = 0
        for i in self.species.size:
            bond = 0.0
            nonBond = 100.0
            for j in self.species.size:
                if self.bondMat[i][j] == 1.0 and self.dratio > bond:
                    bond = self.dratio
                    atom1 = i
                    atom2 = j
                if self.bondMat[i][j] == 0.0 and self.dratio[i][j] < nonBond:
                    nonBond = self.dratio[i][j]
                    atom3 = j
            if bond > nonBond:
                if self.criteriaMet is False:
                    self.criteriaMet = True
                    self.makingBond = (atom1,atom3)
                    self.breakingBond = (atom1,atom2)
                else:
                    self.secondCriteriaMet = True
                    self.secondMakingBond = (atom1, atom3)
                    self.secondBreakingBond = (atom1, atom2)






