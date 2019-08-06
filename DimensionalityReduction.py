from pathreducer import DistancePCA
from pathreducer.filereaders import XYZReader
import ase.io as aio
import numpy as np


def read_pcs(file_root_name='PC', number=2):
    pc_list = []
    for i in range(0, number):
        f = open(str(file_root_name) + str(i) + ".txt", 'r')
        pc = []
        for line in f.readlines():
            words = line.split()
            pc.append([float(words[0]), float(words[1]), float(words[2])])
        pc_list.append(np.asarray(pc))
    return pc_list

# Class which holds options for getting principal coordinates from path reducer
# "trajectory" : Trajectory file to be used
# "number" : Number of PC's wanted
# "ignore_h" : Bool, if true ignore hydrogens
# "subset" : Bool, if True look for start_ind and end_ind variables and only do dimensionality reduction on atoms
#            in this range
# "filePrefix" : Prefix for printing output files
class DimensionalityReduction:
    def __init__(self, trajectory, number=3, ignore_h=False, subset=False, c_only = False, prune_frequency = 1, start_ind=[], end_ind=[], file_prefix="PC"):
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
        atoms_list = self.trajectory
        new_atoms_list = []
        new_indicies = []

        if self.subset:
            for i, atom in enumerate(atoms_list[0]):
                for j in range(0, len(self.start_index)):
                    if atom.index > self.start_index[j] and atom.index <= self.end_index[j]:
                        new_indicies.append(atom.index)
        else:
            for i, atom in enumerate(atoms_list[0]):
                new_indicies.append(atom.index)

        for i, atom in enumerate(atoms_list[0]):
            if self.c_only:
                if atom.symbol != 'C' and atom.index in new_indicies:
                    new_indicies.remove(atom.index)
            elif self.ignore_h:
                if atom.symbol == 'H' and atom.index in new_indicies:
                    new_indicies.remove(atom.index)

        if self.prune_frequency > 1:
            i = 0
            for atom in atoms_list[0]:
                if atom.index in new_indicies:
                    if i % self.prune_frequency != 0:
                        new_indicies.remove(atom.index)
                    i += 1


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
        old_indicies = range(len(new_indicies))
        atoms_dictionary = dict(zip(old_indicies,new_indicies))
        aio.write("temporaryTraj.xyz", new_atoms_list)
        return atoms_dictionary


    # This function calls the path reducer python package
    # https://github.com/share1992/PathReducer/
    def call_path_reducer(self):
        file_ = 'temporaryTraj.xyz'
        # Number of PCA components
        ndim = self.number
        data = XYZReader(file_)
        m = DistancePCA(ndim)
        m.fit(data.coordinates)
        self.variance = m._model.explained_variance_ratio_
        print("Path Reducer: Proportion of variance captured by each coordinate is " + str(self.variance))
        d = m.transform(data.coordinates)
        m.inverse_transform(d)
        coeffs = m.get_components()
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
        for i in range(0,self.number):
            f = open(str(file_root_name)+str(i)+".txt", 'w')
            for row in self.pc_list[i]:
                f.write(str(row[0]) + '\t' + str(row[1]) + '\t' + str(row[2]) + '\n')







