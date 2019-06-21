from abc import abstractmethod
import numpy as np
from ChemDyME import util

# Abstract base class for collective  variable types


class CollectiveVariable:
    @abstractmethod
    def __init__(self):
        pass

    # Update the distance from the current point to the lower boundary
    @abstractmethod
    def get_s(self, mol):
        pass

    # Update the distance from the current point to the lower boundary
    @abstractmethod
    def get_delta(self, mol):
        pass


class PrincipalCoordinates(CollectiveVariable):

    # Collective variable is a linear combinations of interatomic distances.
    # "pc_files" : List of files defining principal coordinates. One PC per file
    # 'number_of_elements' : Point at which to truncate linear combination in each PC
    # 'highest_index_considered" : If PC's only refer to a subset of the atoms this variable give the highest atom index
    #                            needed
    def __init__(self, pc_array, number_of_elements, highest_index_considered=float("inf")):
        self.number_of_elements = number_of_elements
        self.highest_index_considered = highest_index_considered
        # Seperate arrays for indices and coefficients
        # Create list of arrays, one array for each PC file
        self.indicies = []
        self.coefficients = []
        for f in pc_array:
            c, i = self.read_principal_components(f)
            self.indicies.append(i)
            self.coefficients.append(c)
        self.number_of_pcs = len(self.coefficients)

    def __len__(self):
        return len(self.indicies)

    # Function to return n dimesnional vector of PC's
    # util contains Cythonized function for calculating the value of each PC at the geometry given by mol
    def get_s(self, mol):
        d = util.getPC(self.indicies, self.coefficients, mol.get_positions())
        return d

    # Read principal coordijnates from array
    def read_principal_components(self, arr):
        coefficients = arr[:self.number_of_elements, 0]
        indicies = arr[:self.number_of_elements,1:].astype(int)
        return coefficients, indicies

    # Vectorised function to quickly get array of euclidean distances between atoms
    def get_distance_vector(self, mol, highest_index_considered):
        xyz = mol.get_positions()
        if highest_index_considered < len(xyz):
            xyz = xyz[0:highest_index_considered, :]
        d = np.sqrt(np.sum(np.square(xyz[:, np.newaxis, :] - xyz), axis=2))
        return d

    # Function to return a BXD constrain at a given geometry (mol) having hit a given boundary (n)
    # TODO, this function needs tidying up and Cythoning
    def get_delta(self, mol, n):
        return util.get_delta(mol, n, self.indicies, self.coefficients)

    def get_delta2(self, mol, n):
        constraint_final = np.zeros(mol.get_positions().shape)
        pos = mol.get_positions()
        # First loop over all atoms
        for j in range(0, len(self.indicies)):
            constraint = np.zeros(mol.get_positions().shape)
            for k in range(0, self.indicies[j].shape[0]):
                # need a check here in case the collective variable is zero since nan would be returned
                i1 = int(self.indicies[j][k][0])
                i2 = int(self.indicies[j][k][1])
                distance = np.sqrt((pos[i1][0] - pos[i2][0]) ** 2 + (pos[i1][1] - pos[i2][1]) ** 2
                                   + (pos[i1][2] - pos[i2][2]) ** 2)
                first_term = (1 / (2 * distance))
                constraint[i1][0] += first_term * 2 * (pos[i1][0] - pos[i2][0]) * self.coefficients[j][k]
                constraint[i1][1] += first_term * 2 * (pos[i1][1] - pos[i2][1]) * self.coefficients[j][k]
                constraint[i1][2] += first_term * 2 * (pos[i1][2] - pos[i2][2]) * self.coefficients[j][k]
                # alternate formula for case where i is the seccond atom
                constraint[i2][0] += first_term * 2 * (pos[i1][0] - pos[i2][0]) * -self.coefficients[j][k]
                constraint[i2][1] += first_term * 2 * (pos[i1][1] - pos[i2][1]) * -self.coefficients[j][k]
                constraint[i2][2] += first_term * 2 * (pos[i1][2] - pos[i2][2]) * -self.coefficients[j][k]
            constraint *= n[j]
            constraint_final += constraint
        return constraint_final

class Energy(CollectiveVariable):
    # Collective variable is a list of interatomic distances
    # "indicies" : n by 2 array of atom indicies. Each row defines an internuclear distance
    def __init__(self, mol):
        self.mol = mol

    # Function to return n dimensional vector of interatomic distances corresponding
    def get_s(self, mol):
        return np.asarray(mol.get_potential_energy())

    # Function to return a BXD constrain at a given geometry (mol) having hit a given boundary (n)
    def get_delta(self, mol, bound):
        return mol.get_forces()
