import numpy as np

def getPC(PCs, dist):
    s2 = len(PCs)
    D = np.zeros((s2))
    for i in range(0,s2):
        for j in range(0,PCs[i].shape[0]):
            D[i] += dist[PCs[i][j][0]][PCs[i][j][1]]*PCs[i][j][2]
    Dind = PCs
    return (D,Dind)