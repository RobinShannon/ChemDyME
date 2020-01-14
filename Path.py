import numpy as np
import warnings
import matplotlib.pyplot as plt

class Path:
    """
    Stores a list of geometries or frames as a linearly interpolated path in a given collective variable space.
    :param trajectory: List of ASE atoms objects representing the path
    :param collective_variable: a CollectiveVariable object describing how to transform the cartestesian coordinates
                                into the desired collective variable
    :param stride: Resolution with which to read path. The code will read the path every n elements
    :param max_distance_from_path: The maximum distance BXD is allowed to stray from the path. This is enforced in a
                                   ProgressMetric object which places a BXD boundary parallel to the current path
                                   segment. This can either be a single value ( in which case the max distance will
                                   be constant along the path) or it can be a list of numbers specifying a different
                                   max distance for each path segment.
    :param path_file: File name for printing the path to file
    """

    def __init__(self, trajectory, collective_variable, stride=1, max_distance_from_path=float("inf"), path_file='reducedPath.txt'):
        # Apply the defined stride to the path
        self.trajectory = trajectory[0::stride]
        # s stores each path node in the collective Variable representation
        self.s = []
        # totalDistance stores the cumulative path length upto a given path node
        self.total_distance = []
        length = 0
        for i, mol in enumerate(self.trajectory):
            self.s.append(collective_variable.get_s(mol))
            if i == 0:
                length += 0
            else:
                length += np.linalg.norm(self.s[i] - self.s[i-1])
            self.total_distance.append(length)
        # Format the max distance from path so it is an array of the correct length
        self.max_distance = self.format_max_distance(max_distance_from_path)
        # Print path to file
        file = open(path_file, 'w')
        for p in self.s:
            string = 's = ' + ' '.join(map(str, p))
            file.write(string + '\n')
        file.close()

    def format_max_distance(self, distance):
        """
        Determines whether or not the max_distance_from_path parameter was given as a single value or a list. The
        max_distance_from_path is then padded or pruned so that it is a list of the correct length.
        :param distance:
        :return: list with the max_distance_from_path for each path segment
        """
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
        """
        Uses matplotlib to plot the path in two dimensions
        """
        plt.ion()
        data = self.make_array()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colour = plt.cm.jet(np.linspace(0,1,len(data)))
        for i in range(0, len(data) - 1):
            ax.plot(np.array([data[i][0], data[i + 1][0]]), np.array([data[i][1], data[i + 1][1]]),color=colour[i])
            fig.show()

    def print_path3D(self, colour_map="jet"):
        """
        Uses matplotlib to plot the path in three dimensions
        """
        plt.ion()
        data = self.make_array()
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        colour = plt.cm.jet(np.linspace(0,1,len(data)))
        for i in range(0, len(data) - 1):
            ax.plot(np.array([data[i][0], data[i + 1][0]]), np.array([data[i][1],data[i + 1][1]]),zs=np.array([data[i][2],data[i + 1][2]]),color=colour[i])
            fig.show()