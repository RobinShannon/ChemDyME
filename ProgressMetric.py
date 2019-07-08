from abc import abstractmethod
import numpy as np
import sys
import collections

class ProgressMetric:
    """
    Abstract Base Class controlling the metric used to define BXD progress for a given point in collective variable space.
    The base class measures the BXD progress as the distance from the lower boundary to the a given point: The line and
    curve derived classes project a given point onto a line or curve respectivley describing reaction from reactants to
    products.
    :start_mol: :   An ASE Atoms object holding the reactant geometry
    :collectiveVar: An instance of the collectiveVariable class which holds the particular distances considered
    :distanceType:  "distance"  BXD progress is defined by the distance from the reactant geometry to the current point
                    "simple" Progress defined by the distance of the current point from the current lower boundary
    """
    def __init__(self, start_mol, collective_variable, distance_type="simple"):
        self.collective_variable = collective_variable
        # Convert start geometry into the appropriate collective variable
        self.start_s = collective_variable.get_s(start_mol)
        self.distance_type = distance_type
        # Distance from current point to lower boundary
        self.dist_from_bound = 0
        self.project = collections.namedtuple('distance', 'projection, distance_from_bound')

    # Update the distance from the current point to the lower boundary
    def update_dist_from_bound(self, s, n, d):
        self.dist_from_bound = np.vdot(s, n) + d

    def get_dist_from_bound(self, s, bound):
        return np.vdot(s, bound.n) + bound.d

    # Get the current distance of BXD trajectory along defined path
    @abstractmethod
    def project_point_on_path(self, s):
        if self.distance_type == 'distance':
            line = s - self.start_s
            p = np.linalg.norm(line)
        elif self.distance_type == 'simple':
            p = self.dist_from_bound
        else:
            sys.exit("Simple projection type " + str(self.distance_type) +
                     " is not recognised. This should be either \"distance\" or \"simple\"\n")
        return self.project(p)

    # This method returns a bool signifying whether the current max distance from the path has been exceeded
    # In the simple case there is no path and this returns False
    @abstractmethod
    def reflect_back_to_path(self):
        return False

    def get_s(self,mol):
        return self.collective_variable.get_s(mol)

    def get_delta(self,mol,bound):
        return self.collective_variable.get_delta(mol, bound)


class Curve(ProgressMetric):
    """
    Subclass of "Projection" which deals with the instance where one wishes to project BXD onto a linearly
    interpolated path
    :collective_variable: Instance of the collective variable class holding the details of the current collective
                          variables
    :path:                Instance of a path object defining the guess trajectory
    :max_nodes_skipped:   The projection algorithm will consider the n linear segments adjoining the current
                          segment when looking for the closest segment. This allows some segments to be skipped if
                          there are kinks in the path.
    :one_direction:       Boolean flag which if True specifies that only nodes further along the reaction direction will
                          be considered when looking for the closest node. This can prevent the BXD trajectory getting
                          lost on more rugged paths.
    """

    def __init__(self, path, collective_variable, max_nodes_skiped=1, one_direction=False, end_type='distance'):
        self.collective_variable = collective_variable
        self.start_s = path.s[0]
        self.path = path
        self.max_nodes_skipped = max_nodes_skiped
        self.one_direction = one_direction
        # The max distance from the path can be input as either an array or a single value.
        # This function makes an array of maxDistanceFromPath of the correct length
        self.max_distance_from_path = self.path.max_distance
        # current closest linear segment
        self.path_segment = 0
        # get the current distance from the path
        self.distance_from_path = 0
        self.old_distance_from_path = 0
        # Keeps track of whether the bxd trajectory is going in a forward or reverse direction
        self.bxd_reverse = False
        self.end_type = end_type
        self.end_point = path.total_distance[-1]

    '''
    Get distance from S to closest point on a linear segment.
    :segment_start: Coordinates of the start of the segment
    :segment_end:   Coordinates of the end of the segment
    '''
    def distance_to_segment(self, s, segment_end, segment_start):
        # Get vector from segStart to segEnd
        segment = segment_end - segment_start
        # Scalar projection of S onto segment
        scalar_projection = np.vdot((s - segment_start), segment) / np.linalg.norm(segment)
        # Length of segment
        path_segment_length = np.linalg.norm(segment)
        # Vector projection of S onto segment
        vector_projection = segment_start + (scalar_projection * (segment / path_segment_length))
        # Length of this vector projection gives distance from line
        # If the vector projection is past the start or end point of the segment then use distance to the distance to
        # segStart or segEnd respectively
        if scalar_projection < 0:
            dist = np.linalg.norm(s - segment_start)
        elif scalar_projection > path_segment_length:
            dist = np.linalg.norm(s - segment_end)
        else:
            dist = np.linalg.norm(s - vector_projection)
        return dist, scalar_projection

    def vector_to_segment(self, s, segment_end, segment_start):
        # Get vector from segStart to segEnd
        segment = segment_end - segment_start
        # Scalar projection of S onto segment
        scalar_projection = np.vdot((s - segment_start), segment) / np.linalg.norm(segment)
        # Length of segment
        path_segment_length = np.linalg.norm(segment)
        # Vector projection of S onto segment
        vector_projection = segment_start + (scalar_projection * (segment / path_segment_length))
        norm = (s - vector_projection)/np.linalg.norm(s - vector_projection)
        return norm

    def project_point_on_path(self, s):
        # Set up tracking variables to log the closest segment and the distance to it
        minim = float("inf")
        closest_segment = 0
        dist = 0
        p = 0
        # Use self.max_nodes_skipped to track set up the start and end points for looping over path segments.
        start = max(self.path_segment - self.max_nodes_skipped, 0)
        end = min(self.path_segment + (self.max_nodes_skipped+1), len(self.path.s) - 1)
        # If the one_direction flag is True then only consider nodes in one direction
        if self.one_direction:
            # If BXD is going forward then the start node is the current node
            if not self.bxd_reverse:
                start = max(self.path_segment, 0)
            # If BXD is going backwards the end node is the current node + 1
            else:
                end = min(self.path_segment + 1, len(self.path.s) - 1)
        # Now loop over all segments considered and get the distance from S to that segment and the projected distance
        # of S along that segment
        for i in range(start, end):
            dist, projection = self.distance_to_segment(s, self.path.s[i+1], self.path.s[i])
            if dist < minim:
                closest_segment = i
                minim = dist
                p = projection
        # Update the current distance from path and path segment
        self.old_distance_from_path = self.distance_from_path
        self.distance_from_path = minim
        self.path_segment = closest_segment
        # To get the total distance along the path add the total distance along all segments seg < minPoint
        p += self.path.total_distance[closest_segment]
        return p

    def reflect_back_to_path(self):
        if self.distance_from_path > self.max_distance_from_path[self.path_segment]:
            if self.distance_from_path > self.old_distance_from_path:
                return True
            else:
                return False
        else:
            return False

    def path_bound_distance_at_point(self):
        self.max_distance_from_path[self.path_segment]


    # Set the current BXD direction
    def set_bxd_reverse(self, reverse):
        self.bxd_reverse = reverse

    def get_start_s(self):
        s1 = self.path.s[0]
        s2 = self.path.s[1]
        return(s1, s2)

    def get_path_delta(self, mol):
        s = self.collective_variable.get_s(mol)
        seg_start = self.path.s[self.path_segment]
        seg_end = self.path.s[self.path_segment+1]
        norm = self.vector_to_segment(s,seg_end,seg_start)
        return self.collective_variable.get_delta(mol, norm)

    def get_norm_to_path(self):
        seg_start = self.path.s[self.path_segment]
        seg_end = self.path.s[self.path_segment+1]
        n = (seg_end - seg_start) / np.linalg.norm(seg_end - seg_start)
        return n


class Line(ProgressMetric):

    # Subclass of "Projection" where the path is a line connecting start and end geometries
    # "end" : Target geometry. This can either be an ASE atoms object or an array holding the target values of S
    # " max_distance_from_path": Maximum distance from path BXD is allowed to stray.
    # "end_type" : Specifies the format of the target geometry. Can be either "geom : ASE object" or
    #              "collective : array of collective variable values"

    def __init__(self, start_mol, collective_variable, end, max_distance_from_path=float("inf"), end_type="geom"):
        super(Line, self).__init__(start_mol, collective_variable)
        self.max_distance_from_path = max_distance_from_path
        # The current distance from the path
        self.distance_from_path = 0
        if end_type == "collective":
            self.end = end
        elif end_type == "geom":
            self.end = collective_variable.get_s(end)
        else:
            sys.exit("Projection endType " + str(end_type) +
                     " is not recognised. This should be either \"geom\" or \"collective\"\n")
        # In this case the path is simply the vector between start and end coordinates
        self.path = self.start_s - self.end

    def project_point_on_path(self, s):
        # get line from the start coordinate to current S
        baseline = s - self.start_s
        # project this line onto the path line
        project = np.vdot(baseline, self.path) / np.linalg.norm(self.path)
        # Also get vector projection
        vector_projection = (np.vdot(baseline, self.path) / np.vdot(self.path, self.path)) * self.path
        # Length of this vector projection gives distance from line
        self.distance_from_path = np.linalg.norm(vector_projection)
        return project

    def reflect_back_to_path(self):
        if self.distance_from_path > self.max_distance_from_path:
            return True
        else:
            return False












