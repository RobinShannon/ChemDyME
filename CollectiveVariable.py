from abc import abstractmethod
import numpy as np
import util

# Abstract base class for collective  variable types
class CollectiveVariable:
    @abstractmethod
    def __init__(self):
        pass


    # Update the distance from the current point to the lower boundary
    @abstractmethod
    def getS(self):
        pass


class PrincipalCoordinates(CollectiveVariable):

    # Collective variable is a linear combinations of interatomic distances.
    # "PC_files" : List of files defining principal coordinates. One PC per file
    # 'numberOfElements' : Point at which to truncate linear combination in each PC
    # 'highestIndexConsidered" : If PC's only refer to a subset of the atoms this variable give the highest atom index
    #                            needed
    def __init__(self, PC_files, numberOfElements, highestIndexConsidered = float("inf")):
        self.numberOfElements = numberOfElements
        self.highestIndexConsidered = highestIndexConsidered
        #Seperate arrays for indices and coefficients
        # Create list of arrays, one array for each PC file
        self.indicies = []
        self.coeffs = []
        for f in PC_files:
            i,c = self.readPrincipalComponents(f)
            self.indicies.append(i)
            self.coeffs.append(c)

    # Function to return n dimesnional vector of PC's
    # util contains Cythonized function for calculating the value of each PC at the geometry given by mol
    def getS(self,mol):
        dist = self.getDistVect(mol,self.highestIndexConsidered)
        D, Dind = util.getPC(self.indicies, self.coeffs, dist)
        return D, Dind

    # Read principal coordijnates from file
    def readPrincipalComponents(self, inp):
        #Get number of lines in file
        num_lines = sum(1 for line in open(inp))
        # size of array is min of specified lines to be read and the actual size
        size = min(num_lines, self.numberOfElements)
        coeff = np.zeros(size)
        indicies = np.zeros((size,2))
        i = 0
        with open(inp) as f:
            lines = f.read().splitlines()
        for i in range(1,size+1):
            indicies[i-1][0] = int(lines[i].split('\t')[1])
            indicies[i-1][1] = int(lines[i].split('\t')[2])
            coeff[i-1][2] = float(lines[i].split('\t')[0])
        return indicies, coeff

    # Vectorised function to quickly get array of euclidean distances between atoms
    def getDistVect(self, mol, highestIndexConsidered):
        xyz = mol.get_positions()
        if highestIndexConsidered < len(xyz):
            xyz = xyz[0:highestIndexConsidered,:]
        D = np.sqrt(np.sum(np.square(xyz[:, np.newaxis, :] - xyz), axis=2))
        return D

    # Function to return a BXD constrain at a given geometry (mol) having hit a given boundary (n)
    # TODO, this function needs tidying up and Cythoning
    def genCombBXDDel(self, mol, n):
        constraintFinal = np.zeros(mol.get_positions().shape)
        pos = mol.get_positions()
        # First loop over all atoms
        for j in range(0, len(self.indicies)):
            constraint = np.zeros(mol.get_positions().shape)
            for k in range(0, self.indicies[j].shape[0]):
                # need a check here in case the collective variable is zero since nan would be returned
                index1 = int(self.indicies[j][k][0])
                index2 = int(self.indicies[j][k][1])
                distance = np.sqrt((pos[index1][0] - pos[index2][0]) ** 2 + (pos[index1][1] - pos[index2][1]) ** 2 + (
                            pos[index1][2] - pos[index2][2]) ** 2)
                firstTerm = (1 / (2 * distance))
                constraint[index1][0] += firstTerm * 2 * (pos[index1][0] - pos[index2][0]) * self.coeffs[j][k]
                constraint[index1][1] += firstTerm * 2 * (pos[index1][1] - pos[index2][1]) * self.coeffs[j][k][2]
                constraint[index1][2] += firstTerm * 2 * (pos[index1][2] - pos[index2][2]) * self.coeffs[j][k][2]
                # alternate formula for case where i is the seccond atom
                constraint[index2][0] += firstTerm * 2 * (pos[index1][0] - pos[index2][0]) * -1 * self.coeffs[j][k][2]
                constraint[index2][1] += firstTerm * 2 * (pos[index1][1] - pos[index2][1]) * -1 * self.coeffs[j][k][2]
                constraint[index2][2] += firstTerm * 2 * (pos[index1][2] - pos[index2][2]) * -1 * self.coeffs[j][k][2]
            constraint *= n[j]
            constraintFinal += constraint
        return constraintFinal


class Distances(CollectiveVariable):

    # Collective variable is a list of interatomic distances
    # "indicies" : n by 2 array of atom indicies. Each row defines an internuclear distance
    def __init__(self, indicies ):
        self.indicies = indicies

    # Function to return a BXD constrain at a given geometry (mol) having hit a given boundary (n)
    def genDistBXDDel(self, mol, S, n):
        constraint = np.zeros(mol.get_positions().shape)
        pos = mol.get_positions()
        # First loop over all atoms
        for i in range(0, len(mol)):
            # Then check wether that atom contributes to a collective variable
            for j in range(0, len(self.indicies)):
                # need a check here in case the collective variable is zero since nan would be returned
                if isinstance(S, list) and S[j] != 0:
                    firstTerm = (1 / (2 * S[j]))
                elif not isinstance(S, list) and S != 0:
                    firstTerm = (1 / (2 * S))
                else:
                    firstTerm = 0
                if isinstance(n, list):
                    norm = n[j]
                else:
                    norm = n
                # if atom i is the first atom in bond j then add component to the derivative based upon chain rule differentiation
                if self.indicies[j][0] == i:
                    index = int(self.indicies[j][1])
                    constraint[i][0] += firstTerm * 2 * (pos[i][0] - pos[index][0]) * norm
                    constraint[i][1] += firstTerm * 2 * (pos[i][1] - pos[index][1]) * norm
                    constraint[i][2] += firstTerm * 2 * (pos[i][2] - pos[index][2]) * norm
                # alternate formula for case where i is the seccond atom
                if self.indicies[j][1] == i:
                    index = int(self.indicies[j][0])
                    constraint[i][0] += firstTerm * 2 * (pos[index][0] - pos[i][0]) * -1 * norm
                    constraint[i][1] += firstTerm * 2 * (pos[index][1] - pos[i][1]) * -1 * norm
                    constraint[i][2] += firstTerm * 2 * (pos[index][2] - pos[i][2]) * -1 * norm
        return constraint

    # Function to return n dimensional vector of interatomic distances corresponding
    def getS(self,mol):
        size = len(self.indicies)
        D = np.zeros((size))
        Dind = []
        for i in range(0, size):
            D[i] = mol.get_distance(int(self.indicies[i][0]), int(self.indiciese[i][1]))
            Dind.append([self.indicies[i][0], self.indicies[i][1]])
        return D, Dind


