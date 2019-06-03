import yaml
import DimensionalityReduction as DR
import Tools as tl
import sys
from ase.io import write, read

class Globals:

    def __init__(self, path = 'inp.yaml'):
        # Read YAML input file
        with open(path, 'r') as stream:
            data = yaml.safe_load(stream)

        # Get root
        root = data
        # First get the runType
        self.RunType = data['RunType']
        # Then create an ASE atoms type for the statrting molecule
        start = data['Start']

        if 'xyz' in start.keys():
            self.startMol = read(start['xyz'])
        elif 'smile' in start.keys():
            self.startMol = tl.getMolFromSmile(start['smile'])
        else:
            sys.exit("Unreognised start type")

        if self.RunType == "GenBXD":
            #Look for path reducer commands
            if 'DimensionalityReduction' in root.keys():
                dimLevel = root['DimensionalityReduction']
                dimRed = DR.DimensionalityReduction(dimLevel[trajectory])











