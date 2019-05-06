from abc import ABCMeta, abstractmethod
import ConnectTools  as ct
import numpy as np


# Controlling the path used to guide BXD
# Start is an ASE Atoms object holding the reactant geometry
# collectiveVar is an instance of the collectiveVar class which holds the particular distances considered
class Path:
    def __init__(self, start, collectiveVar):
        self.start = start
        self.collectiveVar = collectiveVar

    def projectPointOnPath(S, path, type, n, D, reac, pathNode):
        numberOfSegments = 5
        baseline = S - reac
        Sdist = np.vdot(S, n) + D
        minPoint = 0
        distFromPath = 0
        if type == 'gates':
            gBase = S - path[pathNode][0]
            # check if gate has been hit
            RMSD = np.linalg.norm(S - path[pathNode + 1][0])
            if RMSD < 0.1:
                print("hit")
                pathNode += 1
            project = np.vdot(gBase, path[pathNode + 1][0]) / np.linalg.norm(path[pathNode + 1][0])
            project += path[minPoint][1]
        if type == 'curve':
            minim = 100000
            minPoint = 0
            distArray = []
            # Only consider nodes either side of current node
            # Use current node to define start and end point
            start = max(pathNode - (numberOfSegments - 1), 0)
            end = min(pathNode + numberOfSegments, len(path))
            for i in range(start, end):
                dist = np.linalg.norm(S - path[i][0])
                distArray.append(dist)
                if dist < minim:
                    minPoint = i
                    minim = dist
            if minPoint == 0:
                node = 1
            else:
                pathSeg = path[minPoint][0] - path[minPoint - 1][0]
                pathSegLength = np.linalg.norm(pathSeg)
                linProject = np.vdot((S - path[minPoint - 1][0]), pathSeg) / np.linalg.norm(pathSeg)
                if linProject < pathSegLength or minPoint == len(path) - 1:
                    minPoint -= 1
                node = minPoint + 1
            pathSeg = path[node][0] - path[node - 1][0]
            project = np.vdot((S - path[node - 1][0]), pathSeg) / np.linalg.norm(pathSeg)
            # Also get vector projection
            # Get vector for and length of linear segment
            pathSegLength = np.linalg.norm(pathSeg)
            # finally get distance of projected point along vec
            plength = path[node - 1][0] + project * pathSeg / pathSegLength
            # Length of this vector projection gives distance from line
            distFromPath = np.linalg.norm(S - plength)
            project += path[node - 1][1]
        if type == 'linear':
            project = np.vdot(baseline, path) / np.linalg.norm(path)
            minPoint = False
        if type == 'distance':
            project = np.linalg.norm(baseline)
        if type == 'simple distance':
            project = Sdist
        return Sdist, project, minPoint, distFromPath

class Curve(Path):

    def __init__(self, start,  traj, coordinates, maxNodesSkiped = 1):
                self.path = self.getPathRepresentation(traj)
                self.maxNodesSkipped = maxNodesSkiped
                self.pathNode = 0
                # Get coefficients
                super(Curve, self).__init__(start, coordinates)

    def projectPointOnPath(self, S, n, D, mol):
        Sdist = np.vdot(S, n) + D
        minPoint = 0
        if type == 'curve':
            minim = 100000
            minPoint = 0
            distArray = []
            # Only consider nodes either side of current node
            # Use current node to define start and end point
            start = max(self.pathNode - (self.maxNodesSkipped), 0)
            end = min(self.pathNode + (self.maxNodesSkipped-1), len(self.path))
            for i in range(start, end):
                dist,projection = self.distanceToSegment(self.path[i+1][0] - self.path[i][0])
                distArray.append(dist)
                if dist < minim:
                    minPoint = i
                    minim = dist
                    pathSeg = self.path[i+1][0] - self.path[i][0]
                    project = projection
            # Get length of closest linear segment on curve
            pathSegLength = np.linalg.norm(pathSeg)
            # Get distance of projected point along the linear segment
            plength = self.path[minPoint - 1][0] + project * pathSeg / pathSegLength
            # Length of this vector projection gives distance from line
            self.distFromPath = np.linalg.norm(S - plength)
            project += self.path[minPoint - 1][1]
        return Sdist, project, minPoint, distFromPath

class Line(Path):

    def __init__(self, start, end, coordinates, totalDistance=100):
        self.end = end
        # Get coefficients
        super(Curve, self).__init__(start, coordinates, totalDistance)



