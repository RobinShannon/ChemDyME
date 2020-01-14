from pathreducer import DistancePCA
from pathreducer.filereaders import XYZReader
import ase.io as aio
import numpy as np


def read_pcs(file_root_name='PC', number=2):
    """
    Utility function to read a set of files with same prefix into an list of arrays defining each principal coordinate
    :param file_root_name: DEFAULT = PC
                           The function will look for all files with the given prefix when reading PC's
    :param number: Number of PC's
    :return: A list of PC arrays
    """
    pc_list = []
    for i in range(0, number):
        f = open(str(file_root_name) + str(i) + ".txt", 'r')
        pc = []
        for line in f.readlines():
            words = line.split()
            pc.append([float(words[0]), float(words[1]), float(words[2])])
        pc_list.append(np.asarray(pc))
    return pc_list


class DimensionalityReduction:
    def __init__(self, trajectory, number=3, ignore_h=False, subset=False, c_only = False, prune_frequency = 1, start_ind=[], end_ind=[], file_prefix="PC"):
        """
        Class which interfaces with the pathreducer code in order. This class alters the geometry based upon a number
        of flags, sends xyz files to pathreducer for dimensionality reduction and then reads the results into a list of
        arrays. Pathreducer defines each PC as a linear combination of interatomic distances
        :param trajectory: List of ASE atoms objects defining frames of a trajectory or guess path
        :param number: DEFAULT = 3
                       number of PC's desired
        :param ignore_h: DEFAULT = False
                         Boolean, determines whether or not to ignore remove hydrogens before sending coordinates to
                         pathreducer
        :param subset: DEFAULT = False
                       Boolean, determines whether or not to consider only a subset of atoms when sending coordinates to
                       pathreducer
        :param c_only: DEFAULT = False
                       Boolean, if True all non carbon atoms are removed before sending to pathreducer.
        :param prune_frequency: DEFAULT = 1
                                If prune_frequency = n then every nth atom will be removed before performing
                                dimensionality reduction
        :param start_ind: Only used if subset = True. This defines a list of starting atom indicies which together with
                          the parameter "end_ind" defines the subset of important atoms. "end_ind" should have the same
                          number of elements as "start_ind" and for each starting index in "start_ind" the corresponding
                          element of "end_ind". UNTIL THE NEXT REFACTOR only atoms > start_ind are considered
                          e.g assuming start_ind = [5, 15] and end_ind = [10,20] dimensionality reduction will be
                          performed upon atoms 6,7,8,9,10,16,17,18,19,20 NB indexing stats at zero.
        :param end_ind: See above
        :param file_prefix: Prefix for printing the PC's to file.
        """
        self.pc_list = []
        self.prune_frequency = prune_frequency
        self.trajectory = trajectory
        self.ignore_h = ignore_h
        self.subset = subset
        self.start_index = start_ind
        self.end_index = end_ind
        self.file_prefix = file_prefix
        self.number = number
        self.c_only = c_only
        # Reorganise the atom ordering according to the flags and print a modified trajectory for path reducer
        self.atoms_dictionary = self.alter_trajectory()
        self.variance = []
        self.call_path_reducer()


    # Alter the trajectory in order depending on ignoreHydrogens and subset flags
    def alter_trajectory(self):
        """
        This function uses the options defined when initialising the DimensionalityReduction object to delete all but
        the important atoms from each frame. This function also creates dictionaries to retain the orriginal atom
        indicies after atoms have been removed
        :return:
        """
        atoms_list = self.trajectory
        new_atoms_list = []
        new_indicies = []

        # First check if we are considering a subset of atoms
        # If so add all the desired atom indicies to new_indicies
        if self.subset:
            for i, atom in enumerate(atoms_list[0]):
                for j in range(0, len(self.start_index)):
                    if atom.index > self.start_index[j] and atom.index <= self.end_index[j]:
                        new_indicies.append(atom.index)
        else:
            # Otherwise all indicies are added new_indicies
            for i, atom in enumerate(atoms_list[0]):
                new_indicies.append(atom.index)

        for i, atom in enumerate(atoms_list[0]):
            # If c_only = True delete all non C atom indicies from new_indicies
            if self.c_only:
                if atom.symbol != 'C' and atom.index in new_indicies:
                    new_indicies.remove(atom.index)
            # If ignore_h = True delete all H atom indicies
            elif self.ignore_h:
                if atom.symbol == 'H' and atom.index in new_indicies:
                    new_indicies.remove(atom.index)

        # If we have requested pruning then delete every nth element from new_indicies
        if self.prune_frequency > 1:
            i = 0
            for atom in atoms_list[0]:
                if atom.index in new_indicies:
                    if i % self.prune_frequency != 0:
                        new_indicies.remove(atom.index)
                    i += 1

        # new_indicies now contains all of the atom indicies we wish to consider in the dimensionality reduction
        # For each frame of the trajectory create a new atoms object with unwanted atoms deleted
        for atoms in atoms_list:
            atoms2 = atoms.copy()
            if self.subset:
                del atoms2[:]
                for j in range(0, len(self.start_index)):
                    atoms_temp = atoms.copy()
                    del atoms_temp[[atom.index for atom in atoms if not (atom.index > self.start_index[j] and atom.index <= self.end_index[j])]]
                    atoms2 += atoms_temp
                atoms = atoms2.copy()
            if self.c_only:
                del atoms2[[atom.index for atom in atoms if atom.symbol != 'C']]
            elif self.ignore_h:
                del atoms2[[atom.index for atom in atoms if atom.symbol == 'H']]
            if self.prune_frequency > 1:
                del atoms2[[atom.index for atom in atoms2 if atom.index % self.prune_frequency != 0]]
            new_atoms_list.append(atoms2)
        # create a list of the same length as new_indicies with consecutive number e.g 0,1,2,3 .....
        # old_indicies and new_indicies can then be used to create a dictionary converting from atom index in the full
        # set of atoms to atom index in the reduced set of atoms
        old_indicies = range(len(new_indicies))
        atoms_dictionary = dict(zip(old_indicies,new_indicies))
        aio.write("temporaryTraj.xyz", new_atoms_list)
        return atoms_dictionary


    # This function calls the path reducer python package
    # https://github.com/share1992/PathReducer/
    def call_path_reducer(self):
        """
         This function calls the path reducer python package, https://github.com/share1992/PathReducer/ with the altered
         atom list and then uses the atoms_dictionary to convert the atom indicies in the pathreducer output back to the
         orriginal indicies in the full set of atoms
         :return:
        """
        # We need to set up a temporary file as the path reducer interface is still via i/o
        file_ = 'temporaryTraj.xyz'
        # Number of PCA components
        ndim = self.number
        # use the XYZReader in pathreducer to read in the file_ and then perform the dimensionality reduction
        data = XYZReader(file_)
        m = DistancePCA(ndim)
        m.fit(data.coordinates)
        self.variance = m._model.explained_variance_ratio_
        print("Path Reducer: Proportion of variance captured by each coordinate is " + str(self.variance))
        d = m.transform(data.coordinates)
        m.inverse_transform(d)
        # all the dimensionality reduction data is now stored in m and we can extract the coefficients and the indicies
        coeffs = m.get_components()
        # At this point use the atoms_dictionary to convert the atom indicies from pathredcuer to the corresponding
        # indicies in the full atom space.
        indicies = np.vectorize(self.atoms_dictionary.get)(m.get_component_origin().astype(int))
        #Loop over all ndim PCs
        for n, c in enumerate(coeffs):
            # Put the coefficients and the atom indicies into a single array
            new_array = np.zeros((len(c), 3))
            new_array[:, 0] = c
            new_array[:, 1:] = indicies.transpose().astype(float)
            a = np.abs(new_array)
            # Sort the new array from largest to smallest coefficient
            new_array = new_array[a[:, 0].argsort()]
            self.pc_list.append(new_array[::-1])

    def print_pcs(self, file_root_name):
        """
        Prints the stored PC list to files. One file for each PC
        :param file_root_name:
        :return:
        """
        for i in range(0,self.number):
            f = open(str(file_root_name)+str(i)+".txt", 'w')
            for row in self.pc_list[i]:
                f.write(str(row[0]) + '\t' + str(row[1]) + '\t' + str(row[2]) + '\n')







