from subprocess import Popen, PIPE
import os

# Class to run master equation calculation
class MasterEq:

    def __init__(self):

        self.newSpeciesFound = False
        self.time = 0.0
        self.ene = 0
        self.prodName = 'none'
        self.visitedList = []
        self.eneList = []
        self.equilCount = 0
        self.MESCommand = os.environ['CHEMDYME_ME_PATH']
        #self.MESCommand = '/Users/RobinS/Documents/mesmerStoch/src/mesmer'

    def runTillReac(self, args2):
        p = Popen([self.MESCommand,args2], stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        out = stderr.decode("utf-8")
        lines = str(out).split('\n')
        words = lines[len(lines)-5].split(' ')
        self.ene = float(words[1])
        self.eneList.append(self.ene)
        words = lines[len(lines)-4].split(' ')
        self.time = float(words[1])
        words = lines[len(lines)-3].split(' ')
        self.prodName = words[1]
        self.visitedList.append(self.prodName)

    def repeated(self):
        length = len(self.visitedList)
        if length > 2:
            if self.visitedList[(length-1)] == self.visitedList[(length - 3)]:
                return True
        else:
            return False

