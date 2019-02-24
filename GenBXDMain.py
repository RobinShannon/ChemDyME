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
from OpenMMCalc import OpenMMCalculator


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
        Reac = tl.setCalc(Reac,"DOS/", gl.trajMethod, gl)
        Prod = tl.setCalc(Prod,"DOS/", gl.trajMethod, gl)
    else:
        Reac = tl.setCalc(Reac,"DOS/", gl.trajMethod, gl.trajLevel)
        Prod = tl.setCalc(Prod,"DOS/", gl.trajMethod, gl.trajLevel)
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
    t.runGenBXD(Reac,Prod,gl.maxHits,gl.maxAdapSteps,gl.PathType,distPath, cbs, gl.decorrelationSteps, gl.histogramLevel,totalPathLength)

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

    nebtools = NEBtools(Path)

    # Get the calculated barrier and the energy change of the reaction.
    Ef, dE = nebtools.get_barrier()

    # Get the barrier without any interpolation between highest images.
    Ef, dE = nebtools.get_barrier(fit=False)

    # Get the actual maximum force at this point in the simulation.
    max_force = nebtools.get_fmax()

    # Create a figure like that coming from ASE-GUI.
    fig = nebtools.plot_band()
    fig.savefig('diffusion-barrier.png')

    # Create a figure with custom parameters.
    fig = plt.figure(figsize=(5.5, 4.0))
    ax = fig.add_axes((0.15, 0.15, 0.8, 0.75))
    nebtools.plot_band(ax)
    fig.savefig('diffusion-barrier.png')
    return Path