from pathreducer import DistancePCA
from pathreducer.filereaders import XYZReader
import ase.io as aio
import numpy as np

# Class which holds options for getting principal coordinates from path reducer
# "trajectory" : Trajectory file to be used
# "number" : Number of PC's wanted
# "ignore_h" : Bool, if true ignore hydrogens
# "subset" : Bool, if True look for start_ind and end_ind variables and only do dimensionality reduction on atoms
#            in this range
# "filePrefix" : Prefix for printing output files
class DimensionalityReduction:
    def __init__(self, trajectory, number=3, ignore_h=False, subset=False, start_ind=0, end_ind=0, file_prefix="PC"):
        self.pc_list = []
        self.trajectory = trajectory
        self.ignore_h = ignore_h
        self.subset = subset
        self.start_index = start_ind
        self.end_index = end_ind
        self.file_prefix = file_prefix
        self.number = number
        # Reorganise the atom ordering according to the flags and print a modified trajectory for path reducer
        self.atoms_dictionary = self.alter_trajectory()
        self.call_path_reducer()
        self.variance = []

    # Alter the trajectory in order depending on ignoreHydrogens and subset flags
    def alter_trajectory(self):
        atoms_list = self.trajectory
        new_atoms_list = []
        new_indicies = []
        # Start creating a new list of that only contains the atom indicies of the remaining atoms after all alterations
        if self.ignore_h and self.subset:
            for i, atom in enumerate(atoms_list[0]):
                if atom.symbol != 'H' and (atom.index < self.start_index and atom.index >= self.end_index):
                    new_indicies.append(atom.index)
        elif self.ignore_h :
            for i, atom in enumerate(atoms_list[0]):
                if atom.symbol != 'H':
                    new_indicies.append(atom.index)
        elif self.subset:
            for i, atom in enumerate(atoms_list[0]):
                if atom.index < self.start_index and atom.index >= self.end_index:
                    new_indicies.append(atom.index)
        else:
            new_indicies = range(len(atoms_list[0]))
        # For each frame of the trajectory create a new atoms object with unwanted atoms deleted
        for atoms in atoms_list:
            atoms2 = atoms.copy()
            if self.subset:
                del atoms2[[atom.index for atom in atoms if (atom.index >= self.start_index and atom.index < self.end_index)]]
            if self.ignore_h:
                del atoms2[[atom.index for atom in atoms if atom.symbol == 'H']]
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





