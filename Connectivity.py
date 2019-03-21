from abc import ABCMeta, abstractmethod
import ConnectTools as CT
import numpy as np


# Class to control connectivity maps / to determine  whether transitions have occured
class StructureMap:

    def __init__(self, mol):
        self.criteriaMet = False


    @abstractmethod
    def reinitialise(self, mol):
        pass

    @abstractmethod
    def update(self, mol):
        pass






class NunezMartinez(StructureMap):

    def __init__(self,mol):
        self.dRef = CT.refBonds(mol)
        self.C = CT.bondMatrix(self.dRef, mol)
        self.transitionIndices = np.zeros(3)
        self.criteriaMet = False

    @abstractmethod
    def update(self, mol):
        self.criteriaMet = False
        size = len(mol.get_positions())
        # Loop over all atoms
        for i in range(0,size):
            bond = 0
            nonbond = 1000
            a_1  = 0
            a_2 = 0
            for j in range(0,size):
                # Get distances between all atoms bonded to i
                if self.C[i][j] == 1:
                    newbond = max(bond,mol.get_distance(i,j)/self.dRef[i][j])
                    #Store Index corresponding to current largest bond
                    if newbond > bond:
                        a_1 = j
                    bond = newbond
                elif self.C[i][j] == 0:
                    newnonbond = min(nonbond, mol.get_distance(i,j)/self.dRef[i][j])
                    if newnonbond < nonbond:
                        a_2 = j
                    nonbond = newnonbond
            if bond > nonbond:
                self.criteriaMet = True
                self.transitionIndices = [a_1, i, a_2]
                print(self.transitionIndices)

    @abstractmethod
    def reinitialise(self, mol):
        self.dRef = CT.refBonds(mol)
        self.C = CT.bondMatrix(self.dRef, mol)
        self.transitionIdices = np.zeros(3)

    def ReactionType(self, mol):
        oldbonds = np.count_nonzero(self.C)
        self.dRef = CT.refBonds(mol)
        self.C = CT.bondMatrix(self.dRef, mol)
        newbonds = np.count_nonzero(self.C)
        if oldbonds > newbonds:
            reacType = 'Dissociation'
        if oldbonds < newbonds:
            reacType = 'Association'
        if oldbonds == newbonds:
            reacType = 'Isomerisation'
        return reacType

