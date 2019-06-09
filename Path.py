import numpy as np
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Class which holds the details of the path in the case of a curve projection.
# "trajectory" : List of ASE atoms objects defining the path
# "collective_variable" : collectiveVarObject
# "stride" : resolution with which to read path. The code will read the path every n elements


class Path:
    def __init__(self, trajectory, collective_variable, stride=1, max_distance_from_path=float("inf")):
        # Apply the defined stride to the path
        self.trajectory = trajectory[0::stride]
        # S stores each path node in the collective Variable representation
        self.s = []
        # totalDistance stores the cumulative path length upto a given path node
        self.total_distance = []
        length = 0
        for i, mol in enumerate(trajectory):
            self.s.append(collective_variable.get_s(mol))
            if i == 0:
                length += 0
            else:
                length += np.linalg.norm(self.s[i] - self.s[i-1])
            self.total_distance.append(length)
        # Format the max distance from path so it is an array of the correct length
        self.max_distance = self.format_max_distance(max_distance_from_path)
        # Print path to file
        path_file = open('reducedPath.txt', 'w')
        for p in self.s:
            path_file.write('s = ' + str(p) + '\n')
        path_file.close()

    def format_max_distance(self, distance):
        # The required length of maxDistance is equal to len(self.S) - 1 since S counts nodes rather than segments
        length = len(self.s) - 1
        # Determine whether distance has been input as a list
        if isinstance(distance, list):
            # If so pad or prune the list till its the right length
            if len(distance) < length:
                warnings.warn("Max distance list has less elements than there are path nodes. The path has been padded")
                while len(distance) < length:
                    distance.append(distance[-1])
            elif len(distance) > length:
                warnings.warn("Max distance list has more elements than there are nodes. The path has been pruned")
                while len(distance) > length:
                    del distance[-1]
            return distance
        else:
            # If array was given as a single value then generate a list of the right length
            dist_array = [distance] * length
            return dist_array

    def make_array(self):
        return np.array(self.s)

    def print_path2D(self, colour_map="jet"):
        plt.ion()
        data = self.make_array()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colour = plt.cm.jet(np.linspace(0,1,len(data)))
        for i in range(0, len(data) - 1):
            ax.plot(np.array([data[i][0], data[i + 1][0]]), np.array([data[i][1], data[i + 1][1]]),color=colour[i])
            fig.show()

    def print_path3D(self, colour_map="jet"):
        plt.ion()
        data = self.make_array()
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        colour = plt.cm.jet(np.linspace(0,1,len(data)))
        for i in range(0, len(data) - 1):
            ax.plot(np.array([data[i][0], data[i + 1][0]]), np.array([data[i][1],data[i + 1][1]]),zs=np.array([data[i][2],data[i + 1][2]]),color=colour[i])
            fig.show()