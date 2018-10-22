class Characterise:

    def __init__(self, mol, TS, string):
        self.mol = mol
        self.TS = TS
        self.string = string
        self.prod


    @abstractmethod
    def Run(self):
        if TS:
           self.OptTS()
        else:
            self.Opt()


