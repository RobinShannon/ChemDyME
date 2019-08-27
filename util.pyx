import numpy as np

def getPC(indicies, coefficients, dist):
    s2 = len(indicies)
    D = np.zeros((s2))
    for i in range(0,s2):
        for j in range(0,indicies[i].shape[0]):
            D[i] += dist[indicies[i][j][0]][indicies[i][j][1]]*coefficients[i][j][2]
    Dind = PCs
    return (D,Dind)