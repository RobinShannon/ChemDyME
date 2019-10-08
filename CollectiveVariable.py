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

class CartesianPrincipalCoordinates(CollectiveVariable):

        # Collective variable is a linear combinations of interatomic distances.
        # "pc_files" : List of files defining principal coordinates. One PC per file
        # 'number_of_elements' : Point at which to truncate linear combination in each PC
        # 'highest_index_considered" : If PC's only refer to a subset of the atoms this variable give the highest atom index
        #                            needed
    def __init__(self, number_of_PCs, input_prefix):
        self.number_of_pcs = number_of_PCs
        self.pc_array = []
        for i in range(0,number_of_PCs):
            input = input_prefix + str(i) + '.txt'
            file = open(input,'r')
            array = np.loadtxt(file)
            self.pc_array.append(array)

        # Function to return n dimesnional vector of PC's
        # util contains Cythonized function for calculating the value of each PC at the geometry given by mol
    def get_s(self, mol):
        d = np.zeros(self.number_of_pcs)
        for i,pc in enumerate(self.pc_array):
            s = np.vdot(pc,mol.get_positions())
            d[i] = s
        return d


    def get_delta(self, mol, n):
        constraint = np.zeros(mol.get_positions().shape)
        for pc,norm in zip(self.pc_array,n):
            constraint += pc * norm
        return constraint

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

class Distances(CollectiveVariable):

    def __init__(self, mol, distances):
        self.mol = mol
        self.distances = distances

    def get_s(self, mol):
        d = np.zeros(len(self.distances))
        for i,dis in enumerate(self.distances):
            d[i] = mol.get_distance(int(dis[0]), int(dis[1]))
        return d

    def get_delta(self, mol, n):
        constraint = np.zeros(mol.get_positions().shape)
        pos = mol.get_positions()
        # First loop over all atoms
        for i in range(0, len(mol)):
            # Then check wether that atom contributes to a collective variable
            for j,d in enumerate(self.distances):
                # need a check here in case the collective variable is zero since nan would be returned
                if mol.get_distance(int(d[0]), int(d[1])) != 0:
                    firstTerm = (1 / (2 * mol.get_distance(int(d[0]), int(d[1]))))
                else:
                    firstTerm = 0
                norm = n[j]
                # if atom i is the first atom in bond j then add component to the derivative based upon chain rule differentiation
                if d[0] == i:
                    index = int(d[1])
                    constraint[i][0] += firstTerm * 2 * (pos[i][0] - pos[index][0]) * norm
                    constraint[i][1] += firstTerm * 2 * (pos[i][1] - pos[index][1]) * norm
                    constraint[i][2] += firstTerm * 2 * (pos[i][2] - pos[index][2]) * norm
                # alternate formula for case where i is the seccond atom
                if d[1] == i:
                    index = int(d[0])
                    constraint[i][0] += firstTerm * 2 * (pos[index][0] - pos[i][0]) * -1 * norm
                    constraint[i][1] += firstTerm * 2 * (pos[index][1] - pos[i][1]) * -1 * norm
                    constraint[i][2] += firstTerm * 2 * (pos[index][2] - pos[i][2]) * -1 * norm
        return constraint

class COM(CollectiveVariable):

    def __init__(self, mol, fragment_1, fragment_2):
        self.mol = mol
        self.fragment_1 = fragment_1
        self.fragment_2 = fragment_2

    def get_s(self, mol):
        self.mass_1 = 0.0
        self.mass_2 = 0.0
        com_1 = np.zeros(3)
        com_2 = np.zeros(3)
        masses = mol.get_masses()
        # First need total mass of each fragment
        for f_1 in self.fragment_1:
            i = int(f_1)
            self.mass_1 += masses[i]
            com_1[0] += masses[i] * mol.get_positions()[i, 0]
            com_1[1] += masses[i] * mol.get_positions()[i, 1]
            com_1[2] += masses[i] * mol.get_positions()[i, 2]
        for f_2 in self.fragment_2:
            i = int(f_2)
            self.mass_2 += masses[i]
            com_2[0] += masses[i] * mol.get_positions()[i, 0]
            com_2[1] += masses[i] * mol.get_positions()[i, 1]
            com_2[2] += masses[i] * mol.get_positions()[i, 2]

        com_1 /= self.mass_1
        com_2 /= self.mass_2

        self.com_1 = com_1
        self.com_2 = com_2

        #Finally get distance between the two centers of mass
        com_dist = np.sqrt(((com_1[0]-com_2[0])**2)+((com_1[1]-com_2[1])**2)+((com_1[2]-com_2[2])**2))
        return com_dist

    def get_delta(self, mol, n):
        constraint = np.zeros(mol.get_positions().shape)
        masses = mol.get_masses()
        com_dist = self.get_s(mol)
        for i in range(0, masses.size):
            for j in range(0, 3):
                constraint[i][j] = 1 / (2 * com_dist)
                constraint[i][j] *= 2 * (self.com_1[j] - self.com_2[j])
                if i in self.fragment_1:
                    constraint[i][j] *= -masses[i] / self.mass_1
                elif i in self.fragment_2:
                    constraint[i][j] *= masses[i] / self.mass_2
                else:
                    constraint[i][j] *= 0
        return constraint