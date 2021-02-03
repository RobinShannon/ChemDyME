from abc import abstractmethod
import numpy as np
import sys
import collections

class ProgressMetric:
    """
    Abstract Base Class controlling the metric used to define BXD progress for a given point in collective variable
    space.The base class measures the BXD progress as the distance from the lower boundary to the a given point: The
    line and curve derived classes project a given point onto a line or curve respectively describing reaction from
    reactants to products.
    All derived classes must implement two methods:

    project_point_on_path: Function to transform any point in collective variable space into into progress toward the
                           defined BXD end point

    reflect_back_to_path: Function returns true if the BXD run has stayed to far from the path and False if otherwise.
                          In the base class this allways evaluates as False

    :param start_mol: An ASE Atoms object holding the reactant geometry
    :param collective_variable: An instance of the collectiveVariable class which holds the particular distances considered
    :param end_point: Either a list defining the collective variable at the target geometry or an ASE atoms object
                      defining the target geometry
    :param end_type: DEFAULT("distance")
                     "distance" An adaptive run is considered to have reached the end if the when the progress
                                metric reaches a particular value equal to that at the specified end_point
    :param distance_type: "distance"  BXD progress is defined by the distance from the reactant geometry to the
                          current point
                          "simple" Progress defined by the distance of the current point from the current lower
                          boundary
    :param number_of_boxes: Perform adaptive BXD for defined number of boxes only (deprecated, this variable is
                            rarely used and has been superceded by the BXDConstraint class
    """
    def __init__(self, start_mol, collective_variable, end_point, end_type='distance',  distance_type="simple", number_of_boxes = 50):
        self.collective_variable = collective_variable
        # Convert start geometry into the appropriate collective variable
        self.start_s = collective_variable.get_s(start_mol)
        self.distance_type = distance_type
        # Distance from current point to lower boundary
        self.dist_from_bound = 0
        # If the end point is a list then assume this corresponds to the collective variable at the end point
        # Otherwise it is an ASE atoms type and the collective variable object is used to convert
        if isinstance(end_point,list):
            self.target_mol = np.asarray(end_point)
        else:
            self.target_mol = collective_variable.get_s(end_point)

        if end_type == 'distance':
            self.end_type = end_type
            self.end_point = self.target_mol
        elif end_type == 'boxes':
            self.end_type = end_type
            self.end_point = number_of_boxes
        self.path_segment = 0
        self.distance_from_path = 0
        self.old_distance_from_path = 0

    def outside_path(self):
        pass

    def get_start_s(self):
        return self.start_s, self.target_mol

    # Update the distance from the current point to the lower boundary
    def update_dist_from_bound(self, s, n, d):
        self.dist_from_bound = np.vdot(s, n) + d

    def get_dist_from_bound(self, s, bound):
        return np.vdot(s, bound.n) + bound.d

    # Get the current distance of BXD trajectory along defined path
    @abstractmethod
    def project_point_on_path(self, s, min_segment = 0, max_segment = np.inf):
        line = s - self.start_s
        p = np.linalg.norm(line)
        return p

    # This method returns a bool signifying whether the current max distance from the path has been exceeded
    # In the simple case there is no path and this returns False
    @abstractmethod
    def reflect_back_to_path(self):
        return False

    def get_s(self,mol):
        return self.collective_variable.get_s(mol)

    def get_delta(self,mol,bound):
        return self.collective_variable.get_delta(mol, bound)

    def set_bxd_reverse(self, reverse):
        pass


class Curve(ProgressMetric):
    """
    Subclass of ProgressMetric for the instance where one wishes to project BXD onto a linearly interpolated guess
    path. The algorithm for determining progress is briefly as follows:
    1. Store closest path segment self.path_segment (starting at segment 0)
    2. Convert MD frame to collective variable (s)
    3. Considering  all path segments between self.path_segment - max_nodes_skiped and self.path_segment + max_nodes_
       skiped get the distance of the shortest line between the current value of s and each path segment.
    4. Get the cumulative distance along the path up to the new closest segment (d) This is stored in the path object.
    5. Get the scalar projection of the point s onto the line defining the closest path segment and add this to d
    6. Return d as the projected distance for the given MD frame
    7. Update self.path_segment with the new closest path segment
    :param path: A path object defining the nodes of the guess path in collective variable space
    :param collective_variable: CollectiveVarible object defining the collective variable space
    :param max_nodes_skiped: DEFAULT = 1
                             The ProgressMetric stores the closest path segment at a given point in the BXD
                             trajectory. The max_nodes_skipped parameter defines the number how muany adjacent path
                             segments should be considered when determining the closest path segment for the next
                             point in the BXD trajectory
    :param one_direction: WARNING  use "True" with caution and never for a converging run.
                          DEFAULT = False
                          If True is specified then only path segments futher along the current direction of travel
                          will be considered when determining the closest path segment
    :param end_point: DEFAULT = 0 (in this case the end point will be the full length of the path)
                     Point at which the adaptive BXD is considered finished. Either defined a projected distance or
                     a number of boxes depending upon the value of the end_type parameter.
    :param end_type: DEFAULT = "distances"
                     "distances" the end_point defines a distance along the path at which the BXD run is considered
                                 complete
                     "boxes" end_point defined number of boxes for the adaptive run. If the adaptive BXD run goes
                             in both directions then more boxes may be added in the reverse direction. See
                             BXDConstraint class
    """

    def __init__(self, path, collective_variable, max_nodes_skiped=1, one_direction=False, end_point=0,  end_type='distance'):
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
        if end_point == 0:
            self.end_point = path.total_distance[-1]
        else:
            self.end_point = end_point


    def distance_to_segment(self, s, segment_end, segment_start):
        """
        Get the distance of the shortest line between s and a given path segment, also return the scalar projection of s
        onto the line segment_end - segment start
        :param s: Current colective variable
        :param segment_end: Starting node of path segment
        :param segment_start: End node of path segment
        :return: Distance, scalar projection, scalar projection as fraction of total distance.
        """
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
            #scalar_projection = 0
        elif scalar_projection > path_segment_length:
            #scalar_projection = path_segment_length
            dist = np.linalg.norm(s - segment_end)
        else:
            dist = np.linalg.norm(s - vector_projection)
        return dist, scalar_projection, scalar_projection/path_segment_length

    def vector_to_segment(self, s, segment_end, segment_start):
        """
        Get a unit vector from s to a given path segment. Used to define the norm for BXD reflection back towards the
        path
        :param s: Current colective variable
        :param segment_end: Starting node of path segment
        :param segment_start: End node of path segment
        :return:
        """
        # Get vector from segStart to segEnd
        segment = segment_end - segment_start
        # Scalar projection of s onto segment
        scalar_projection = np.vdot((s - segment_start), segment) / np.linalg.norm(segment)
        # Length of segment
        path_segment_length = np.linalg.norm(segment)
        # Vector projection of S onto segment
        vector_projection = segment_start + (scalar_projection * (segment / path_segment_length))
        if scalar_projection < 0:
            norm = (s - segment_start) / np.linalg.norm(s - vector_projection)
        elif scalar_projection > 0:
            norm = (s - segment_end) / np.linalg.norm(s - vector_projection)
        else:
            norm = (s - vector_projection)/np.linalg.norm(s - vector_projection)
        norm = (s - vector_projection)/np.linalg.norm(s - vector_projection)
        return norm

    def project_point_on_path(self, s, min_segment = 0, max_segment = np.inf):
        """
        Get projected distance along the path for a given geometry with collective variable s
        :param s: collective variable value for current MD frame
        :return: distance along path.
        """
        # Set up tracking variables to log the closest segment and the distance to it
        minim = float("inf")
        closest_segment = 0
        if max_segment > len(self.path.s) - 1:
            max_segment = len(self.path.s) - 1
        dist = 0
        p = 0
        # Use self.max_nodes_skipped to set up the start and end points for looping over path segments.
        start = self.path_segment - self.max_nodes_skipped
        start = max(start,0)
        end = min(self.path_segment + (self.max_nodes_skipped+1), max_segment)
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
        percentage = 0
        for i in range(start, end):
            dist, projection, percent = self.distance_to_segment(s, self.path.s[i+1], self.path.s[i])
            if dist < minim:
                percentage = percent
                closest_segment = i
                minim = dist
                p = projection
        # Update the current distance from path and path segment
        self.old_distance_from_path = self.distance_from_path
        self.distance_from_path = minim
        self.percentage_along_segment = percentage
        self.path_segment = closest_segment
        # To get the total distance along the path add the total distance along all segments seg < minPoint
        p += self.path.total_distance[closest_segment]
        return p

    def get_node(self, s):
        """
        Get closes path segment to a given value of s, ignoring  self.max_nodes_skipped and self.path_segment
        :param s: Collective variable value
        :return: Closest node
        """
        # Set up tracking variables to log the closest segment and the distance to it
        minim = float("inf")
        closest_segment = 0
        dist = 0
        p = 0
        # Use self.max_nodes_skipped to track set up the start and end points for looping over path segments.
        start = 0
        end =len(self.path.s) - 1
        # Now loop over all segments considered and get the distance from S to that segment and the projected distance
        # of S along that segment
        for i in range(start, end):
            dist, projection, percent = self.distance_to_segment(s, self.path.s[i+1], self.path.s[i])
            if dist < minim:
                closest_segment = i
                minim = dist
        return closest_segment


    def reflect_back_to_path(self):
        """
        Determine whether the current point is outside the defined max_distance_from_path.
        If we are outside the path, but moving closer to the path then return False instead of True
        :return: Boolean
        """
        #Check whether the current distance to the path is outside of the maximum allowable
        if self.distance_from_path > self.path_bound_distance_at_point():
            # If so compare to the previous MD frame to see whether we are moving closer or further away from path
            if self.distance_from_path > self.old_distance_from_path:
                # If moving further awy then return signal the need for reflection
                return True
            else:
                # Otherwise leave reflection for this frame to see the trajectory moves back inside the bound of its own
                # accord
                return False
        else:
            return False

    def outside_path(self):
        """
        Possible duplicate function to be reviewed upon refactor
        :return:
        """
        if self.distance_from_path > self.path_bound_distance_at_point():
            return True
        else:
            return False

    def path_bound_distance_at_point(self):
        """
        Get max distance from path at the current point. This is only neccessay when different max_distances_from_path
        have been defined for different segments.
        :return:
        """
        try:
            gradient = self.max_distance_from_path[self.path_segment][self.path_segment + 1] - self.max_distance_from_path[self.path_segment]
            return self.max_distance_from_path[self.path_segment] + gradient * self.percentage_along_segment
        except:
            return self.max_distance_from_path[self.path_segment]

    def set_bxd_reverse(self, reverse):
        """
        Tells the progress metric object that the BXD trajctory has started reversing back to the starting point
        :param reverse: Boolean
        """
        self.bxd_reverse = reverse

    def get_start_s(self):
        s1 = self.path.s[0]
        s2 = self.path.s[1]
        return(s1, s2)

    def get_path_delta(self, mol):
        """
        Gets the delta_phi for a boundary hit on the path_boundary and sends this to the BXDConstraint object.
        Usually this would be dealt with in the BXDConstraint object but using BXD to reflect back towards the path is
        an exception
        :param mol: ASE atoms type
        :return: Array with del_phi
        """
        s = self.collective_variable.get_s(mol)
        seg_start = self.path.s[self.path_segment]
        seg_end = self.path.s[self.path_segment+1]
        norm = self.vector_to_segment(s,seg_end,seg_start)
        #check norm
        seg = seg_end - seg_start
        n = np.dot(seg,norm)
        return self.collective_variable.get_delta(mol, norm)



    def get_norm_to_path(self, s):
        """
        Get vector norm to current path segment. BXDConstraint will use this to perform the BXD inversion
        :param s: Collective variable value
        :return: vector norm
        """
        seg = self.get_node(s)
        seg_start = self.path.s[seg]
        seg_end = self.path.s[seg+1]
        n = (seg_end - seg_start) / np.linalg.norm(seg_end - seg_start)
        return n


class Line(ProgressMetric):
    """
    Subclass of "Projection" where the path is a line connecting start and end geometries.
    :param start_mol: ASE atoms object defining starting geometry
    :param collective_variable: Collective variable object
    :param end_point: Target point in BXD trajectory, either an ASE atoms object or a list corresponding to the
                      collective variable at the end point
    :param end_type: DEFAULT = "distances"
                     "distances" the end_point defines a distance along the path at which the BXD run is considered
                                 complete
                     "boxes" end_point defined number of boxes for the adaptive run. If the adaptive BXD run goes
                             in both directions then more boxes may be added in the reverse direction. See
                             BXDConstraint class
    :param max_distance_from_path: DEFAULT = "inf"
    :param number_of_boxes: max number of boxes to place in the adaptive run. will be removed upon refactor
    """
    def __init__(self, start_mol, collective_variable, end_point, end_type='distance',  max_distance_from_path=float("inf"), number_of_boxes = 50):

        start_s = collective_variable.get_s(start_mol)
        if isinstance(end_point, list):
            self.path = np.asarray(end_point) - start_s
        else:
            self.path = collective_variable.get_s(end_point) - start_s
        super(Line, self).__init__(start_mol, collective_variable, end_point,  end_type=end_type, number_of_boxes=number_of_boxes)
        self.max_distance_from_path = max_distance_from_path

    def project_point_on_path(self, s, min_segment = None, max_segment = None):
        """
        Get projected distance along the path for a given geometry with collective variable s
        :param s: collective variable value for current MD frame
        :return: distance along path.
        """
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
        """
        Determine whether the current point is outside the defined max_distance_from_path.
        If we are outside the path, but moving closer to the path then return False instead of True
        :return: Boolean
        """
        if self.distance_from_path > self.max_distance_from_path:
            return True
        else:
            return False

    def outside_path(self):
        if self.distance_from_path > self.max_distance_from_path:
            return True
        else:
            return False

        # Set the current BXD direction
    def set_bxd_reverse(self, reverse):
        """
        Tells the progress metric object that the BXD trajctory has started reversing back to the starting point
        :param reverse: Boolean
        """
        print('reversing')












