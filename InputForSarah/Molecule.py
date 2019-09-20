
class Molecule:
    @abstractmethod
    def __init__(self):
        pass

    # Update the distance from the current point to the lower boundary
    @abstractmethod
    def characterise(self, mol):
        pass

    @abstractmethod
    def optimise(self, level):

    def get_frequencies(self, level):

    def get_energy(self, level):



class Stable(Molecule):

    def __init__(self, molecule, level_list):
        self.species = []
        self.level_list = level_list
        self.two_fragments = False
        self.name = []
        self.frequencies = []
        self.energy = []
        self.characterise(molecule)

    def characterise(self, mol):
        pass

class TS(Molecule):

    def __init__(self, molecule, level_list):
        self.species = []
        self.level_list = level_list
        self.frequencies = []
        self.imaginary_frequency = 0
        self.energy = []
        self.characterise(molecule)


