import ChemDyME.Trajectory
from ase.optimize import BFGS
import ChemDyME.Tools as tl
import os
from ase.io import write, read

def run(gl):
    #Read reactant definition
    if gl.StartType == 'file':
        Reac = read(gl.Start)
    elif gl.StartType == 'Smile':
        Reac = tl.getMolFromSmile(gl.Start)


    # Set up calculator
    if gl.trajMethod == "openMM":
        Reac = tl.setCalc(Reac,"DOS/", gl.trajMethod, gl)
    else:
        Reac = tl.setCalc(Reac,"DOS/", gl.trajMethod, gl.trajLevel)

    # Do we want to minimise the reactant
    if gl.GenBXDrelax:
        min = BFGS(Reac)
        try:
            min.run(fmax=0.1, steps=150)
        except:
            min.run(fmax=0.1, steps=50)
    t = ChemDyME.Trajectory.Trajectory(Reac,gl,os.getcwd(),0,False)
    t.runBXDEconvergence(gl.maxHits,gl.maxAdapSteps,gl.eneAdaptive, gl.decorrelationSteps, gl.histogramLevel, gl.runsThrough, gl.numberOfBoxes,gl.grainSize)