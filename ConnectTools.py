import numpy as np
from time import process_time

# Method to get the center of mass seperation for two fragments.
# Frag1 has the start and end index of the smaller fragment in xyz
def getCOMdist(mol, frag):
    mass1 = 0.0
    mass2 = 0.0
    masses = mol.get_masses()
    COM1 = np.zeros(3)
    COM2 = np.zeros(3)

    # First need total mass of each fragment
    for i in range(0,masses.size):
        if i >= frag[0] and i <= frag[1]:
            mass1 += masses[i]
        else:
            mass2 += masses[i]

    # Then determine center of mass co-ordinates
    for i in range(0,masses.size):
        if i >= frag[0] and i <= frag[1]:
            COM1[0] += masses[i] * mol.get_positions()[i,0]
            COM1[1] += masses[i] * mol.get_positions()[i,1]
            COM1[2] += masses[i] * mol.get_positions()[i,2]
        else:
            COM2[0] += masses[i] * mol.get_positions()[i,0]
            COM2[1] += masses[i] * mol.get_positions()[i,1]
            COM2[2] += masses[i] * mol.get_positions()[i,2]

    COM1 /= mass1
    COM2 /= mass2

    # Finally calculate the distance between COM1 and COM2
    COMdist = np.sqrt( ((COM1[0] - COM2[0]) ** 2) + ((COM1[1] - COM2[1]) ** 2) + ((COM1[2] - COM2[2]) ** 2))
    return COMdist

def getCOMonly(mol):
    mass = 0.0
    COM = np.zeros(3)
    masses = mol.get_masses()
    # First need total mass of each fragment
    for i in range(0,masses.size):
        mass += masses[i]

    # Then determine center of mass co-ordinates
    for i in range(0,masses.size):
        COM[0] += masses[i] * mol.get_positions()[i,0]
        COM[1] += masses[i] * mol.get_positions()[i,1]
        COM[2] += masses[i] * mol.get_positions()[i,2]

    COM /= mass

    return COM

# Method to return derivative of COM seperation via chain rule
# Needs double CHECKING
def getCOMdel(Mol, frag):
    mass1 = 0.0
    mass2 = 0.0
    masses = Mol.get_masses()
    COM1 = np.zeros(3)
    COM2 = np.zeros(3)
    #First need total mass of each fragment
    for i in range(0,masses.size):
        if i >= frag[0] and i <= frag[1]:
            mass1 += masses[i]
        else :
            mass2 += masses[i]
    #Then determine center of mass co-ordinates
    for i in range(0,masses.size):
        if i >= frag[0] and i <= frag[1]:
            COM1[0] += masses[i] * Mol.get_positions()[i,0]
            COM1[1] += masses[i] * Mol.get_positions()[i,1]
            COM1[2] += masses[i] * Mol.get_positions()[i,2]
        else:
            COM2[0] += masses[i] * Mol.get_positions()[i,0]
            COM2[1] += masses[i] * Mol.get_positions()[i,1]
            COM2[2] += masses[i] * Mol.get_positions()[i,2]

    COM1 /= mass1
    COM2 /= mass2

    # Finally calculate the distance between COM1 and COM2
    COMdist = np.sqrt( ((COM1[0] - COM2[0]) ** 2) + ((COM1[1] - COM2[1]) ** 2) + ((COM1[2] - COM2[2]) ** 2))

    # Now need the derivative component wise
    constraint = np.zeros(Mol.get_positions().shape)
    for i in range(0,masses.size):
        for j in range(0,3):
            constraint[i][j] = 1 / ( 2 * COMdist)
            constraint[i][j] *= 2 * (COM1[j] - COM2[j])
            if i >= frag[0] and i <= frag[1]:
                constraint[i][j] *= -masses[i] / mass1
            else:
                constraint[i][j] *= masses[i] / mass2
    return constraint

# Set up a reference matrix for ideal bond length between any two atoms in the system
# Maps species types onto a grid of stored ideal bond distances stored in the global variables module
def refBonds(mol):
    dict = {'CC' : 1.6, 'CH' : 1.2, 'HC' : 1.2, 'CO' : 1.6, 'OC' : 1.6, 'OH' : 1.2, 'HO' : 1.2, 'OO' : 1.6, 'HH' : 1.0, 'CF' : 1.4, 'FC' : 1.4, 'OF' : 1.4, 'FO' : 1.4, 'HF' : 1.1, 'FH' : 1.1, 'FF' : 1.4 }
    size =len(mol.get_positions())
    symbols = mol.get_chemical_symbols()
    dRef = np.zeros((size,size))
    for i in range(0 ,size) :
        for j in range(0, size) :
            sp = symbols[i] + symbols[j]
            dRef[i][j] = dict[sp]
    return dRef

def bondMatrix(dRef,mol):
    size =len(mol.get_positions())
    C = np.zeros((size,size))
    dratio = np.zeros((size,size))
    for i in range(0,size):
        for j in range(0,size):
            C[i][j] = mol.get_distance(i,j)
            if i != j:
                dratio[i][j] = C[i][j] / dRef[i][j]
            if i == j:
                C[i][j] = 2
            elif C[i][j] < dRef[i][j]:
                C[i][j] = 1.0
            else:
                C[i][j] = 0.0
    return C

def getChangedBonds(mol1, mol2):
    r = refBonds(mol1)
    C1 = bondMatrix(r, mol1)
    C2 = bondMatrix(r, mol2)
    indicies = []
    size =len(mol1.get_positions())
    for i in range(1,size):
        for j in range(0,i):
            if C1[i][j] != C2[i][j]:
                indicies.append(i)
                indicies.append(j)
    ind2 = []
    [ind2.append(item) for item in indicies if item not in ind2]
    return ind2

def getChangedBonds2(mol1, mol2):
    r = refBonds(mol1)
    C1 = bondMatrix(r, mol1)
    C2 = bondMatrix(r, mol2)
    indicies = []
    size =len(mol1.get_positions())
    for i in range(1,size):
        for j in range(0,i):
            if C1[i][j] != C2[i][j]:
                indicies.append([i,j])
    ind2 = []
    [ind2.append(item) for item in indicies if item not in ind2]
    return ind2


def get_bi_xyz(mol1, mol2):
    COM1 = getCOMonly(mol1)

    #Translate COM1 to the origin
    xyz1 = mol1.get_positions()
    for i in range(0,xyz1.shape[0]):
        xyz1[i][0] -= COM1[0]
        xyz1[i][1] -= COM1[1]
        xyz1[i][2] -= COM1[2]

    # Get random point vector at 8 angstrom separation from COM1
    # Get three normally distrubted numbers
    x_y_z = np.random.normal(0,1,3)
    # normalise and multiply by sphere radius
    sum = np.sqrt(x_y_z[0]**2 + x_y_z[1]**2 + x_y_z[2]**2)
    x_y_z *= 1/sum * 8

    # Get displacement from COM2 to x_y_z
    COM2 = getCOMonly(mol2)
    displace = COM2 - x_y_z
    # Modify xyz2 coords accordingly
    xyz2 = mol2.get_positions()
    for i in range(0,xyz2.shape[0]):
        xyz2[i][0] -= displace[0]
        xyz2[i][1] -= displace[1]
        xyz2[i][2] -= displace[2]

    # Append xyz2 onto xyz1 and return
    xyz1 = np.append(xyz1,xyz2,axis=0)

    return xyz1

# Vectorised function to quickly get array of euclidean distances between atoms
def getDistVect(mol):
    xyz = mol.get_positions()
    size =len(mol.get_positions())
    D = np.zeros((size,size))
    D = np.sqrt(np.sum(np.square(xyz[:,np.newaxis,:] - xyz), axis=2))
    return D

def getSPRINT(xyz):
    pass




def projectPointOnPath2(S,path,type,n,D,reac, pathNode):
    baseline = S - reac
    Sdist = np.vdot(S,n) + D
    distFromPath = 0
    if type == 'curve':
        mini = 10000
        minPoint = 0
        distArray =[]
        #Only consider nodes either side of current node
        #Use current node to define start and end point
        if pathNode == 0:
            start = 0
            end = 3
        elif pathNode == 1:
            start = 0
            end =  4
        else:
            start = pathNode - 1
            end = max(pathNode + 2, len(path))
        for i in range(start,end):
            dist = np.linalg.norm(S - path[i][0])
            distArray.append(dist)
            if dist < minim:
                minPoint = i
                minim = dist
                distArray.append(dist)
        if minPoint == (len(path)-1) or (minPoint !=0 and distArray[minPoint + 1] > distArray[minPoint - 1]):
            pathSeg = path[minPoint][0] - path[minPoint-1][0]
            project = np.vdot((S - path[minPoint-1][0]),pathSeg) / np.linalg.norm(pathSeg)
            project += path[minPoint-1][1]
            # Also get vector projection
            vProject = (np.vdot((S - path[minPoint-1][0]),pathSeg)/np.vdot(pathSeg,pathSeg)) * pathSeg
            # Length of this vector projection gives distance from line
            distFromPath = np.linalg.norm(vProject)
        else:
            pathSeg = path[minPoint+1][0] - path[minPoint][0]
            project = np.vdot((S - path[minPoint][0]),pathSeg) / np.linalg.norm(pathSeg)
            project += path[minPoint][1]
            # Also get vector projection
            vProject = (np.vdot((S - path[minPoint-1][0]),pathSeg)/np.vdot(pathSeg,pathSeg)) * pathSeg
            # Length of this vector projection gives distance from line
            distFromPath = np.linalg.norm(vProject)
    if type == 'linear':
        project = np.vdot(baseline,path) / np.linalg.norm(path)
        # Also get vector projection
        vProject = (np.vdot(baseline,path) / np.vdot(path, path)) * path
        # Length of this vector projection gives distance from line
        distFromPath = np.linalg.norm(vProject)
    if type == 'distance':
        project = np.linalg.norm(baseline)
    if type =='simple distance':
        project = Sdist
    return Sdist,project,minPoint,distFromPath

def projectPointOnPath(S,path,type,n,D,reac, pathNode, reverse):
    numberOfSegments = 4
    baseline = S - reac
    Sdist = np.vdot(S,n) + D
    minPoint = 0
    distFromPath = 0
    if type == 'gates':
        gBase = S - path[pathNode][0]
        #check if gate has been hit
        RMSD = np.linalg.norm(S-path[pathNode+1][0])
        if RMSD < 0.1:
            print ("hit")
            pathNode += 1
        project = np.vdot(gBase,path[pathNode+1][0]) / np.linalg.norm(path[pathNode+1][0])
        project += path[minPoint][1]
    if type == 'curve':
        distFromPath = 100000
        minPoint = 0
        #Only consider nodes either side of current node
        #Use current node to define start and end point
        if not reverse:
            start = max(pathNode, 0)
            end = min(pathNode + numberOfSegments, len(path)-2)
        else:
            start = min(pathNode, pathNode - numberOfSegments)
            end = max(pathNode, len(path) - 2)
        for i in range(start,end):
            pathSeg = path[i+1][0] - path[i][0]
            proj = np.vdot((S - path[i][0]), pathSeg) / np.linalg.norm(pathSeg)
            pathSegLength = np.linalg.norm(pathSeg)
            # get distance of projected point along vec
            plength = path[i][0] + ((proj * pathSeg)/ pathSegLength)
            # Length of this vector projection gives distance from line
            if proj < 0:
                dist = np.linalg.norm(S - path[i][0])
            elif proj > pathSegLength:
                dist = np.linalg.norm(S - path[i+1][0])
            else:
                dist = np.linalg.norm(S - plength)
            if dist <= distFromPath:
                minPoint = i
                distFromPath = dist
                project = proj
        project += path[minPoint][1]
    if type == 'linear':
        project = np.vdot(baseline,path) / np.linalg.norm(path)
        minPoint = False
    if type == 'distance':
        project = np.linalg.norm(baseline)
    if type =='simple distance':
        project = Sdist
    return Sdist,project,minPoint,distFromPath

def genBXDDel(mol,S,Sind,n):
    try:
        l = Sind[0].shape[1]
    except:
        l =0
    if l == 3:
        constraint = genLinCombBXDDel(mol,S,Sind,n)
    else:
        constraint = genDistBXDDel(mol,S,Sind,n)
    return constraint


# Get del_phi for bxd for arbitrary number of interatomic distances
# S is the vector of collective variables and Sind gives the index of the bonding atoms in the full catesian coords
# mol stores the full moleculular structure and n gives the coordinates of the boundary that has been hit
def genDistBXDDel(mol,S,Sind,n):
    print("constrain")
    constraint = np.zeros(mol.get_positions().shape)
    pos = mol.get_positions()
    #First loop over all atoms
    for i in range(0,len(mol)):
        #Then check wether that atom contributes to a collective variable
        for j in range(0,len(Sind)):
            # need a check here in case the collective variable is zero since nan would be returned
            if isinstance(S,list) and S[j] != 0:
                firstTerm = (1/(2*S[j]))
            elif not isinstance(S,list) and S != 0:
                firstTerm = (1 / (2 * S))
            else:
                firstTerm=0
            if isinstance(n,list):
                norm = n[j]
            else:
                norm = n
            # if atom i is the first atom in bond j then add component to the derivative based upon chain rule differentiation
            if Sind[j][0] == i:
                index=int(Sind[j][1])
                constraint[i][0] += firstTerm*2*(pos[i][0]-pos[index][0])*norm
                constraint[i][1] += firstTerm*2*(pos[i][1]-pos[index][1])*norm
                constraint[i][2] += firstTerm*2*(pos[i][2]-pos[index][2])*norm
            # alternate formula for case where i is the seccond atom
            if Sind[j][1] == i:
                index = int(Sind[j][0])
                constraint[i][0] += firstTerm*2*(pos[index][0]-pos[i][0])*-1*norm
                constraint[i][1] += firstTerm*2*(pos[index][1]-pos[i][1])*-1*norm
                constraint[i][2] += firstTerm*2*(pos[index][2]-pos[i][2])*-1*norm
    return constraint

# Get del_phi for bxd for linear combination of interatomic distances
# S is the vector of collective variables
# mol stores the full moleculular structure and n gives the coordinates of the boundary that has been hit
def genLinCombBXDDel(mol,S,PC,n):
    t = process_time()
    constraintFinal = np.zeros(mol.get_positions().shape)
    pos = mol.get_positions()
    #First loop over all atoms
    for j in range(0,len(PC)):
        constraint = np.zeros(mol.get_positions().shape)
        for k in range(0,PC[j].shape[0]):
            # need a check here in case the collective variable is zero since nan would be returned
            index1 = int(PC[j][k][0])
            index2 = int(PC[j][k][1])
            distance = np.sqrt((pos[index1][0]-pos[index2][0])**2 + (pos[index1][1]-pos[index2][1])**2 + (pos[index1][2]-pos[index2][2])**2)
            firstTerm = (1/(2*distance))
            constraint[index1][0] += firstTerm*2*(pos[index1][0]-pos[index2][0])*PC[j][k][2]
            constraint[index1][1] += firstTerm*2*(pos[index1][1]-pos[index2][1])*PC[j][k][2]
            constraint[index1][2] += firstTerm*2*(pos[index1][2]-pos[index2][2])*PC[j][k][2]
            # alternate formula for case where i is the seccond atom
            constraint[index2][0] += firstTerm*2*(pos[index1][0]-pos[index2][0])*-1*PC[j][k][2]
            constraint[index2][1] += firstTerm*2*(pos[index1][1]-pos[index2][1])*-1*PC[j][k][2]
            constraint[index2][2] += firstTerm*2*(pos[index1][2]-pos[index2][2])*-1*PC[j][k][2]
        constraint *= n[j]
        constraintFinal += constraint
    elapsed_time = process_time() - t
    #print("time to get constraint = " + str(elapsed_time))
    return constraintFinal
