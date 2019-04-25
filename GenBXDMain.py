import Trajectory
import ConnectTools as ct
import numpy as np
import Tools as tl
from ase.optimize import BFGS
from ase.neb import NEB, NEBtools
import os
from ase.optimize import FIRE
from ase.neb import NEBtools
from ase.io import write, read


def run(gl):
    #Read reactant definition
    if gl.StartType == 'file':
        Reac = read(gl.Start)
    elif gl.StartType == 'Smile':
        Reac = tl.getMolFromSmile(gl.Start)
    #Read product definition
    if gl.EndType == 'file':
        Prod= read(gl.End)
    elif gl.EndType == 'Smile':
        Prod = tl.getMolFromSmile(gl.End)

    #Set calculatiors
    #Reac = tl.setCalc(Reac,"DOS/", gl.trajMethod, gl.atomTypes)
    if gl.trajMethod == "openMM":
        Reac = tl.setCalc(Reac,"GenBXD/", gl.trajMethod, gl)
        #Prod = tl.setCalc(Prod,"GenBXD/", gl.trajMethod, gl)
    else:
        Reac = tl.setCalc(Reac,"GenBXD/", gl.trajMethod, gl.trajLevel)
        Prod = tl.setCalc(Prod,"GenBXD/", gl.trajMethod, gl.trajLevel)
    # Partially minimise both reactant and product
    if gl.GenBXDrelax:
        min = BFGS(Reac)
        try:
            min.run(fmax=0.1, steps=20)
        except:
            min.run(fmax=0.1, steps=20)
        min2 = BFGS(Prod)
        try:
            min2.run(fmax=0.1, steps=20)
        except:
            min2.run(fmax=0.1, steps=20)

    # Get important interatomic distances
    if gl.CollectiveVarType == "changedBonds":
        cbs = ct.getChangedBonds2(Reac, Prod)
    elif gl.CollectiveVarType == "all":
        cbs = ct.getChangedBonds2(Reac, Prod)
    elif gl.CollectiveVarType == "specified":
        cbs = gl.CollectiveVar
    elif gl.CollectiveVarType == "file":
        cbs = gl.principalCoordinates

    #Get path to project along
    distPath = []
    totalPathLength = 0
    if gl.PathType == 'curve' or gl.PathType == 'gates':
        if gl.PathFile == 'none':
            Path = getPath(Reac,Prod,gl)
        else:
            Path = read(gl.PathFile,index='::1')

        distPath.append((ct.getDistMatrix(Path[0],cbs)[0],0))
        for i in range(1,len(Path)):
            l = np.linalg.norm(ct.getDistMatrix(Path[i],cbs)[0] - ct.getDistMatrix(Path[i-1], cbs)[0])
            totalPathLength += l
            distPath.append((ct.getDistMatrix(Path[i],cbs)[0],totalPathLength))
    elif gl.PathType == 'linear':
        distPath = ct.getDistMatrix(Prod,cbs)[0] - ct.getDistMatrix(Reac, cbs)[0]


    # initialise then run trajectory
    t = Trajectory.Trajectory(Reac,gl,os.getcwd(),0,False)
    t.runGenBXD(Reac,Prod,gl.maxHits,gl.maxAdapSteps,gl.PathType,distPath, cbs, gl.decorrelationSteps, gl.histogramBins,totalPathLength, gl.fixToPath, gl.pathDistCutOff,gl.epsilon)

def getPath(Reac,Prod,gl):
    xyzfile3 = open(("IRC3.xyz"), "w")
    Path = []
    Path.append(Reac.copy())
    for i in range(0,30):
        image = Reac.copy()
        image = tl.setCalc(image,"DOS/", gl.lowerMethod, gl.lowerLevel)
        Path.append(image)
    image = Prod.copy()
    image = tl.setCalc(image,"DOS/", gl.lowerMethod, gl.lowerLevel)
    Path.append(image)

    neb1 = NEB(Path,k=1.0,remove_rotation_and_translation = True)
    try:
        neb1.interpolate('idpp',optimizer="MDMin",k=1.0)
    except:
        neb1.interpolate()

    optimizer = FIRE(neb1)
    optimizer.run(fmax=0.07, steps = 500)
    neb2 = NEB(Path,k=1.0, climb=True, remove_rotation_and_translation = True )
    optimizer = FIRE(neb2)
    optimizer.run(fmax=0.07, steps = 1500)
    for i in range(0,len(Path)):
        tl.printTraj(xyzfile3, Path[i])
    xyzfile3.close()



    return Path