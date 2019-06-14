import numpy as np

def getPC(indicies, coefficients, pos):
    s2 = len(indicies)
    D = np.zeros((s2))
    for i in range(0,s2):
        for j in range(0,indicies[i].shape[0]):
            ind1 = indicies[i][j][0]
            ind2 = indicies[i][j][1]
            dist = np.sqrt((pos[ind2][0] - pos[ind1][0])**2 + (pos[ind2][1] - pos[ind1][1])**2 + (pos[ind2][2] - pos[ind1][2])**2)
            D[i] += dist*coefficients[i][j]
    return (D)

def getPC2(indicies, coefficients, dist):
    s2 = len(indicies)
    D = np.zeros((s2))
    for i in range(0,s2):
        for j in range(0,indicies[i].shape[0]):
            D[i] += dist[indicies[i][j][0]][indicies[i][j][1]]*coefficients[i][j]
    return (D)

def get_delta(mol, n, indicies, coefficients):
    constraint_final = np.zeros(mol.get_positions().shape)
    pos = mol.get_positions()
    # First loop over all atoms
    for j in range(0, len(indicies)):
        constraint = np.zeros(mol.get_positions().shape)
        for k in range(0, indicies[j].shape[0]):
            # need a check here in case the collective variable is zero since nan would be returned
            i1 = int(indicies[j][k][0])
            i2 = int(indicies[j][k][1])
            distance = np.sqrt((pos[i1][0] - pos[i2][0]) ** 2 + (pos[i1][1] - pos[i2][1]) ** 2
                               + (pos[i1][2] - pos[i2][2]) ** 2)
            first_term = (1 / (2 * distance))
            constraint[i1][0] += first_term * 2 * (pos[i1][0] - pos[i2][0]) * coefficients[j][k]
            constraint[i1][1] += first_term * 2 * (pos[i1][1] - pos[i2][1]) * coefficients[j][k]
            constraint[i1][2] += first_term * 2 * (pos[i1][2] - pos[i2][2]) * coefficients[j][k]
            # alternate formula for case where i is the seccond atom
            constraint[i2][0] += first_term * 2 * (pos[i1][0] - pos[i2][0]) * -coefficients[j][k]
            constraint[i2][1] += first_term * 2 * (pos[i1][1] - pos[i2][1]) * -coefficients[j][k]
            constraint[i2][2] += first_term * 2 * (pos[i1][2] - pos[i2][2]) * -coefficients[j][k]
        constraint *= n[j]
        constraint_final += constraint
    return constraint_final