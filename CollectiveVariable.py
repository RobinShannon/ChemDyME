from abc import abstractmethod
import numpy as np
from ChemDyME import util


class CollectiveVariable:
    """
    Abstract base class for collective variable types:
    A collective variable describes some reduced dimensional property of the molecular structure which BXD will act
    upon. The CollectiveVariable class controls a collective variable s of arbitrary dimension.
    All instances of this class must implement the folowing two functions;

    get_s : Returns the value of the collective variable for a particular set of cartesian coordinates

    get_delta : Returns the derivate of s (the colective variable) with respect to the cartesian coordinates

    """
    @abstractmethod
    def __init__(self):
        pass


    @abstractmethod
    def get_s(self, mol):
        pass

    @abstractmethod
    def get_delta(self, mol, n):
        pass


class PrincipalCoordinates(CollectiveVariable):
    """
    Collective variable consists of an arbitary number of principal coordiantes, each of which  is a linear
    combination of interatomic distances of the form PC = c1 * r1 + ... + cn * rn where r is an interatomic
    distances and c is some coefficient.
    :param pc_array: List of arrays containing the coefficient (dimension 0) and corresponding atom indicies
                    (dimensions 1 and 2). There is one array for each PC considered. This list of array is generated
                    by the DimensionalityReduction class
    :param number_of_elements: Point at which to truncate linear combination in each PC
    """

    def __init__(self, pc_array, number_of_elements, highest_index_considered=float("inf")):

        self.number_of_elements = number_of_elements
        self.indicies = []
        self.coefficients = []
        # for each PC in the list seperate out the 3D array into coefficients and the corresponding atom indicies
        for f in pc_array:
            c, i = self.read_principal_components(f)
            # c and i are arrays and self.indicies and self.coefficients are lists containing an array for each PC
            self.indicies.append(i)
            self.coefficients.append(c)
        # Store the number of PC's considered
        self.number_of_pcs = len(self.coefficients)

    def __len__(self):
        return len(self.indicies)


    def get_s(self, mol):
        """
        Evaluate the n dimesnional vector of PC's. Util contains a Cythonized function for calculating the value of
        each PC at the geometry given by mol
        :param mol: ASE atoms object with current cartesian coordinates
        :return: Numpy array of floats corresponding to the value of each PC at the current geometry
        """
        try:
            d = util.getPC(self.indicies, self.coefficients, mol.get_positions())
        except:
            d = util.getPC(self.indicies, self.coefficients, mol)
        return d

    def read_principal_components(self, arr):
        """
        Separates out the coefficients and atom indicies from an array
        :param arr: 3D array containing coefficients and followed by atom indicies
        :return: two arrays: "coefficients" with the coefficients for each interatomic distance in a given PC
                             "indicies" 2D array with the 2 atom indicies
        """
        coefficients = arr[:self.number_of_elements, 0]
        indicies = arr[:self.number_of_elements,1:].astype(int)
        return coefficients, indicies


    def get_delta(self, mol, n):
        """
        Evaluate the derivative of the each collective variable with respect to the atom cartesian
        coordinates (del_phi see DOI: 10.1039/c6fd00138f) at a particular BXD boundary.
        :param mol: ASE atoms object containing current cartesian coordinates
        :param n: norm to the the BXD boundary.
        :return: 3N by 3 array with the derivative del_phi
        """
        return util.get_delta(mol, n, self.indicies, self.coefficients)


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
        try:
            pos = mol.get_positions()
        except:
            pos = mol
        d = np.zeros(self.number_of_pcs)
        for i,pc in enumerate(self.pc_array):
            s = np.vdot(pc,pos)
            d[i] = s
        return d


    def get_delta(self, mol, n):
        constraint = np.zeros(mol.get_positions().shape)
        for pc,norm in zip(self.pc_array,n):
            constraint += pc * norm
        return constraint

class Energy(CollectiveVariable):
    """
    Collective variable consisting of the potential energy of the system.
    :param mol: ASE atoms object. This object must have a calculator atatched.
    """

    def __init__(self, mol):
        self.mol = mol


    def get_s(self, mol):
        """
        Function to return the potential energy of the system as a numpy array
        :param mol: ASE atoms object
        :return: The potential energy of the system
        """
        return np.asarray(mol.get_potential_energy())

    def get_delta(self, mol, bound):
        """
        Evaluate del_phi which in this case is given by the molecular forces
        :param mol: ASE atoms object
        :param bound: norm of BXD boundary hit, not needed in this case since for BXD in energy the norm is alway 1,
                      but included for consistency with the base class
        :return: The forces of the system
        """
        return mol.get_forces()

class Distances(CollectiveVariable):
    """
    Collective variable consisting of an arbitrary number of interatomic distances
    :param mol: ASE atoms object
    :param distances: 2D array of atom indicies for defining a an arbitrary number of interatomic distances within the
           system
    """

    def __init__(self, mol, distances):
        self.mol = mol
        self.distances = distances

    def get_s(self, mol):
        """
        Evaluate the interatomic distances at the given geometry
        :param mol: ASE atoms object
        :return: Numpy array of inter-atomic distances
        """
        # Create a numpy array of the correct size to hold all the interatomic distances in the variable
        d = np.zeros(len(self.distances))
        # Iterate over the distances list and evaluate the interatomic distance corresponding to each pair of atom
        # indices
        for i,dis in enumerate(self.distances):
            d[i] = mol.get_distance(int(dis[0]), int(dis[1]))
        return d

    def get_delta(self, mol, n):
        """
        Evaluate del_phi at the given geometry.
        :param mol: ASE atoms object
        :param n: norm to the BXD boundary hit. The norm has the same dimensionality as the collective variable
        :return: array corresponding to del_phi
        """
        constraint = np.zeros(mol.get_positions().shape)
        pos = mol.get_positions()
        # First loop over all atoms
        for i in range(0, len(mol)):
            # Then check whether that atom contributes to a collective variable
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
    """
    Collective variable consisting a the distance between the centers of mass of two groups of atoms within the
    system
    :param mol: Ase atoms object
    :param fragment_1: List of atom indices in first group
    :param fragment_2: List of atom indices in second group
    """

    def __init__(self, mol, fragment_1, fragment_2):
        self.mol = mol
        self.fragment_1 = fragment_1
        self.fragment_2 = fragment_2

    def get_s(self, mol):
        """
        Get the distance between the centers of mass of the two groups at the current geometry
        :param mol: ASE atoms type
        :return: COM separation
        """
        self.mass_1 = 0.0
        self.mass_2 = 0.0
        # Set up numpy arrays to hold the x,y and z coordinates of the two COM's
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

        # Finally get distance between the two centers of mass
        com_dist = np.sqrt(((com_1[0]-com_2[0])**2)+((com_1[1]-com_2[1])**2)+((com_1[2]-com_2[2])**2))
        return com_dist

    def get_delta(self, mol, n):
        """
        Evaluate the derivative of the collective variable with respect to the atom cartesian
        coordinates (del_phi see DOI: 10.1039/c6fd00138f) at a particular BXD boundary.
        :param mol: ASE atoms object containing current cartesian coordinates
        :param n: norm to the the BXD boundary.
        :return: 3N by 3 array with the derivative del_phi
        """
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