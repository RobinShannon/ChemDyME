from abc import abstractmethod
import numpy as np
from ase.io import read, write
import os
from copy import deepcopy


class BXD:
    """
    BXD base class, there are currently three derived versions of BXD:
    1. Adaptive BXD, controlling the placement of BXD boundaries.
    2. Shrinking BXD, WIP, reintoduce a BXD class to gradually decrease a progress_metric until a particular value
       is reached. This class can be used to bring to seperate molecular fragments together and hold them in close
       proximity for example if you are running automated mechanism generation starting from a bimolecular reactant
    3. Converging BXD, uses a set of BXD boundaries generated from an adaptive BXD object and controls a converging
       run where each boundary is hit a specified number of times. This derived class also implements the free
       energy analysis methods.
    All members of this class must implement a number of abstract methods:
    update: This method takes the current geometry, and stores collective variable and progress metric data. This method
            does most of the bookeeping and calls other methods to place new boundaries and determine boundary hits
    stuck_fix: method to alter the geometry if a BXD run gets stuck at a particular boundary
    boundary_check: Checks whether a BXD boundary has been hit at the current geometry
    criteria_met: Checks whether the current bounds have been hit sufficiently to become transparent and allow BXD into
                  the next box
    reached_end: Checks the progress metric to see whether the BXD run needs to be reversed or whether it is complete

    """
    def __init__(self, progress_metric, stuck_limit=20):
        self.progress_metric = progress_metric
        self.steps_since_any_boundary_hit = 0
        self.stuck_limit = stuck_limit
        self.box_list = []
        self.box = 0
        self.stuck_count = 0
        self.inversion = False
        self.reverse = False
        self.stuck = False
        self.bound_hit = "none"
        self.path_bound_hit = False
        self.s = 0
        self.old_s = 0
        self.completed_runs = 0
        self.delta = 0
        self.new_box = True
        self.furthest_progress = 0
        self.geometry_of_furthest_point = 0

    def __len__(self):
        return len(self.box_list)

    def __getitem__(self, item):
        return self.box_list[item]

    def __reversed__(self):
        return self[::-1]

    @abstractmethod
    def update(self, mol):
        pass

    @abstractmethod
    def stuck_fix(self):
        pass

    @abstractmethod
    def boundary_check(self):
        pass

    # Check whether the criteria for box convergence has been met
    @abstractmethod
    def criteria_met(self, boundary):
        pass

    @abstractmethod
    def reached_end(self):
        pass

    def initialise_files(self):
        pass


    # Determine whether boundary should be transparent or reflective
    def update_bounds(self):
        """
        Utilises the abstract class' criteria_met method to manage whether or not a boundary should be transparent
        """
        if self.reverse:
            self.box_list[self.box].upper.transparent = False
        else:
            self.box_list[self.box].lower.transparent = False

        if self.reverse and (self.criteria_met(self.box_list[self.box].lower)):
            self.box_list[self.box].lower.transparent = True
        elif self.reverse is False and (self.criteria_met(self.box_list[self.box].upper)):
            self.box_list[self.box].upper.transparent = True

    def del_constraint(self, mol):
        """
        interfaces with the progress metric to return the delta_phi for the constraint at whichever boundary is hit
        :param mol: ASE atoms object with the current geometry
        :return:
        """
        if self.bound_hit == 'upper':
            norm = self.box_list[self.box].upper.n
        if self.bound_hit == 'lower':
            norm = self.box_list[self.box].lower.n
        self.delta = self.progress_metric.get_delta(mol, norm)
        return self.delta

    def path_del_constraint(self, mol):
        """
        Interfaces with the progress metric to return the del_phi for the path constraint parallel to the current path
        segment
        :param mol: ASE atoms object with the current geometry
        :return:
        """
        return self.progress_metric.get_path_delta(mol)

    def get_s(self, mol):
        """
        Get the current value of the collective variable
        :param mol: ASE atoms object with the current geometry
        :return:
        """
        return self.progress_metric.get_s(mol)

    def print_bounds(self, file="bounds.txt"):
        """
        Prints the BXD boundaries to file.
        :param file:
        :return:
        """
        f = open(file,'w')
        f.write("BXD boundary list \n\n")
        string = ("Boundary\t" + str(0) + "\tD\t=\t" + str(self.box_list[0].lower.d) + "\tn\t=\t" + str(self.box_list[0].lower.n) + "\n" )
        string = string.replace('\n', '')
        f.write(string + "\n")
        for i in range(0, len(self.box_list)):
            string = "Boundary\t" + str(i+1) + "\tD\t=\t" + str(self.box_list[i].upper.d) + "\tn\t=\t" + str(self.box_list[i].upper.n) + "\tS\t=\t" + str(self.box_list[i].upper.s_point)
            string = string.replace('\n','')
            f.write(string + "\n")
        f.close()


class Adaptive(BXD):
    """
    Derived BXD class controlling and Adaptive BXD run. This class controls the adaptive placing of boundaries and
    determines whether or not a boundary has been hit and whether an inversion is required.
    :param progress_metric: ProgressMetric object which manages the CollectiveVariable representation, transforms
                            the current MD geometry into "progess" between the BXD start and end points and
                            optionally contains a path object if BXD is following a guess path
    :param stuck_limit: Integer, DEFAULT = 5.
                        Controls the number of subsequent hits at a boundary before BXD is considered stuck and the
                        stuck fix method is called to remedy the situation
    :param fix_to_path: Boolean, DEFAULT = False.
                        If True all boundaries will be alligned such that they are perpendicular to the path. Should
                        only be used for a curve progress_metric
    :param adaptive_steps: Integer, DEFAULT = 1000
                           Number of MD steps sampled before placing a new boundary in the direction of BXD progress
    :param epsilon: Float. DEFAULT = 0.9
                           Used in histograming to determine the proportion of the adaptive sampling points which
                           should be outside the new adaptive boundary. The cumulative probability of points outside
                           the new boundary should be ( 1.0 - epsilon)
    :param reassign_rate: DEFAULT = 2.
                          If an adaptive bound has not been hit after adaptive_steps * reassign_rate then the
                          boundary will be moved based on new sampling
    :param one_direction: Boolean, DEFAULT = False,
                          If True, then the adaptive BXD run will be considerd complete once it reached the
                          progress_metric end_point and will not attempt to place extra boundaries in the reverse
                          direction
    :param decorellation_limit: Integer, DEFAULT = 0
                                A boundary hit / passage is only counted if it occurs decorrelation_limit steps
                                after the previous hit.
    """

    def __init__(self, progress_metric, stuck_limit=2,  fix_to_path=True, adaptive_steps=1000, epsilon=0.9,
                 reassign_rate=2, one_direction = False, decorrelation_limit = 0, adaptive_reverse = False):
        # call the base class init function to set up general parameters
        super(Adaptive, self).__init__(progress_metric, stuck_limit)
        self.adaptive_reverse = adaptive_reverse
        self.fix_to_path = fix_to_path
        self.one_direction = one_direction
        self.adaptive_steps = adaptive_steps
        self.histogram_boxes = int(np.sqrt(adaptive_steps))
        self.epsilon = epsilon
        self.reassign_rate = reassign_rate
        self.completed_runs = 0
        # set up the first box based upon the start and end points of the progress metric, other will be added as BXD
        # progresses
        s1, s2 = self.progress_metric.get_start_s()
        b1, b2 = self.get_starting_bounds(s1, s2)
        box = self.get_default_box(b1, b2)
        # List of box objects this list along with the self.box parameter keeps track of which box we are in and the
        # associated data for that particular box
        self.box_list.append(box)
        self.skip_box = False
        self.steps_since_any_boundary_hit = 0
        self.decorrelation_limit = decorrelation_limit


    def update(self, mol, decorrelated):
        """
        General bookeeping method. Takes an ASE atoms object, stores data from progress_metric at the current geometry,
        and calls the update_adaptive_bounds and boundary_check methods to add new boundaries and to keep track of which
        box we are in and whether and inversion is neccessary
        :param mol: ASE atoms object
        :return:
        """

        # If this is the first step in a new box then store its geometry in the box object, then set new_box to False
        if self.new_box:
            self.box_list[self.box].geometry = mol.copy()
        self.new_box = False

        # get the current value of the collective variable and the progress data
        self.s = self.get_s(mol)

        projected_data = self.progress_metric.project_point_on_path(self.s)
        distance_from_bound = self.progress_metric.get_dist_from_bound(self.s, self.box_list[self.box].lower)

        if not self.reverse:
            if self.progress_metric.path_segment < self.box_list[self.box].min_segment:
                self.box_list[self.box].min_segment = self.progress_metric.path_segment
            if self.progress_metric.path_segment > self.box_list[self.box].max_segment:
                self.box_list[self.box].max_segment = self.progress_metric.path_segment

        self.inversion = False
        self.bound_hit = "none"


        # Check whether we are in an adaptive sampling regime.
        # If so update_adaptive_bounds checks current number of samples and controls new boundary placement
        if self.box_list[self.box].type == "adap":
            self.update_adaptive_bounds()

        self.reached_end(projected_data)

        # If we have sampled for a while and not hot the upper bound then reassign the boundary.
        # How often the boundary is reassigned depends upon the reasign_rate parameter
        if self.box_list[self.box].type == "normal" and len(self.box_list[self.box].data) > \
                self.reassign_rate * self.adaptive_steps:
            self.reassign_boundary()

        # Check whether a velocity inversion is needed, either at the boundary or back towards the path
        self.inversion = self.boundary_check() or self.path_bound_hit
        self.steps_since_any_boundary_hit += 1

        # If we are stuck at a boundary call the apropriate stuck_fix method
        if self.stuck_count > self.stuck_limit:
            self.stuck_fix()
            self.inversion = False
            self.stuck_count = 0

        # update counters depending upon whether a boundary has been hit
        if self.inversion:
            self.stuck_count += 1
            self.steps_since_any_boundary_hit = 0
        else:
            self.steps_since_any_boundary_hit += 1
            self.stuck_count = 0
            self.old_s = self.s
            self.box_list[self.box].upper.step_since_hit += 1
            self.box_list[self.box].lower.step_since_hit += 1
            # Provided we are close enough to the path, store the data of the current point
            if not self.progress_metric.reflect_back_to_path():
                if decorrelated:
                    self.box_list[self.box].data.append((self.s, projected_data, distance_from_bound))
                # If this is point is the largest progress metric so far then store its geometry.
                # At the end of the run this will store the geometry of the furthest point along the BXD path
                if projected_data > self.furthest_progress:
                    self.furthest_progress = projected_data
                    self.geometry_of_furthest_point = mol.copy()




    def update_adaptive_bounds(self):
        """
        If a box is in an adaptive sampling regime, this method checks the number of data points and determines whether
        or not to add a new adaptive bound. If BXD is in the reverse direction the new a new box is created between the
        current box and the previous one the self.box_list.
        :return:
        """
        # If adaptive sampling has ended then add a boundary based on sampled data
        if len(self.box_list[self.box].data) > self.adaptive_steps:
            # Fist indicate the box no longer needs adaptive sampling
            self.box_list[self.box].type = "normal"
            # If not reversing then update the upper boundary and add a new adaptive box on the end of the list
            if not self.reverse:
                # Histogram the box data to get the averaged top and bottom values of s based on the assigned epsilon
                self.box_list[self.box].get_s_extremes(self.histogram_boxes, self.epsilon)
                bottom = self.box_list[self.box].bot
                top = self.box_list[self.box].top
                # use the bottom and top s to generate a new upper bound
                b1 = self.convert_s_to_bound(bottom, top)
                # copy this bound as it will form the lower bound of the next box
                b2 = deepcopy(b1)
                # copy the current upper bound which will be used for the new box, this upper bound is a dummy bound
                # which can never be hit
                b3 = deepcopy(self.box_list[self.box].upper)
                b3.invisible = True
                b3.s_point = self.box_list[self.box].upper.s_point
                # assign b1 to the current upper bound and create a new box which is added to the end of the list
                self.box_list[self.box].upper = b1
                self.box_list[self.box].upper.transparent = True
                new_box = self.get_default_box(b2, b3)
                new_box.min_segment = self.box_list[self.box].max_segment
                self.box_list.append(new_box)
            elif self.reverse:
                print("making new adaptive bound in reverse direction")
                # same histogramming procedure as above but this time it is the lower bound which is updated
                self.box_list[self.box].get_s_extremes(self.histogram_boxes, self.epsilon)
                bottom = self.box_list[self.box].bot
                top = self.box_list[self.box].top
                b1 = self.convert_s_to_bound(bottom, top)
                b2 = deepcopy(b1)
                b3 = deepcopy(self.box_list[self.box].lower)
                self.box_list[self.box].lower = b1
                new_box = self.get_default_box(b3, b2)
                new_box.max_segment = self.box_list[self.box].max_segment
                new_box.min_segment = self.box_list[self.box].min_segment
                # at this point we partition the  current box into two by inserting a new box at the correct point in the box_list
                self.box_list.insert(self.box, new_box)
                self.box_list[self.box].active = True
                self.box_list[self.box].upper.transparent = False
                self.box += 1
                self.box_list[self.box].lower.transparent = True
            if self.adaptive_reverse:
                self.reverse = True
                self.box_list[self.box].lower.pause = True
                self.box_list[self.box].upper.pause = True
                self.box_list[self.box].data = []

    def get_default_box(self, lower, upper):
        """
        method returns a default adaptive box object from the given upper and lower bounds
        :param lower: BXDbound object represnting the lower bound
        :param upper: BXDbound object represnting the upper bound
        :return:
        """
        b = BXDBox(lower, upper, "adap", True)
        return b

    def reached_end(self, projected):
        """
        Method to check whether bxd has either reached the end point and needs reversing. This method also checks
        whether the bxd run should be considered finished
        :param projected:
        :return:
        """
        if not self.reverse:
            if self.progress_metric.end_type == 'distance':
                if projected >= self.progress_metric.end_point:
                    self.box_list[self.box].get_s_extremes(self.histogram_boxes, self.epsilon)
                    bottom = self.box_list[self.box].bot
                    top = self.s
                    # use the bottom and top s to generate a new upper bound
                    b1 = self.convert_s_to_bound(bottom, top)
                    self.box_list[self.box].upper = b1
                    self.box_list[self.box].upper.transparent = False
                    if self.box_list[self.box].type != 'adap':
                        del self.box_list[-1]
                    if self.one_direction:
                        self.completed_runs += 1
                    else:
                        self.reverse = True
                        self.box_list[self.box].type = 'adap'
                        self.box_list[self.box].data = []
                        self.progress_metric.set_bxd_reverse(self.reverse)
            elif self.progress_metric.end_type == 'boxes':
                if self.box >= self.progress_metric.end_point and self.box_list[self.box].type != 'adap':
                    if self.one_direction:
                        self.completed_runs += 1
                        del self.box_list[-1]
                    else:
                        self.reverse = True
                        self.box_list[self.box].type = 'adap'
                        self.box_list[self.box].data = []
                        del self.box_list[-1]
                        self.progress_metric.set_bxd_reverse(self.reverse)
        else:
            if self.box == 0:
                self.completed_runs += 1
                self.reverse = False
                self.progress_metric.set_bxd_reverse(self.reverse)

    def reassign_boundary(self):
        """
        repeates the procedure of producing an adaptive bound and replaces the exsisting upper or lower bound
         depending whether self.reverse is true or false respectively
        :return:
        """
        fix = self.fix_to_path
        print("re-assigning boundary")
        self.fixToPath = False
        self.box_list[self.box].get_s_extremes(self.histogram_boxes, self.epsilon)
        bottom = self.box_list[self.box].bot
        top = self.box_list[self.box].top
        b = self.convert_s_to_bound(bottom, top)
        b2 = self.convert_s_to_bound(bottom, top)
        b.transparent = True
        if self.reverse:
            self.box_list[self.box].lower = b
            self.box_list[self.box - 1].upper = b2
        else:
            self.box_list[self.box].upper = b
            self.box_list[self.box+1].lower = b2
        self.fix_to_path = fix
        self.box_list[self.box].data = []

    def stuck_fix(self):
        """
        If the fixed (non-adaptive) bound has been hit repeatedly then this method allows BXD to go back (or forward in
        the case of self.reverse = True) to the previous visited box to try and fix the stuck issue

        :return:
        """
        if self.bound_hit == 'lower' and self.reverse is False:
            self.box -= 1
            self.box_list[self.box].upper.transparent = True
        elif self.bound_hit == 'upper' and self.reverse is True:
            self.box += 1
            self.box_list[self.box].lower.transparent = True

    def criteria_met(self, boundary):
        pass

    def get_starting_bounds(self, low_s, high_s):
        n1 = (high_s - low_s) / np.linalg.norm(high_s - low_s)
        n2 = n1
        d1 = -1 * np.vdot(n2, low_s)
        d2 = -1 * np.vdot(n2, high_s)
        b1 = BXDBound(n1, d1)
        b2 = BXDBound(n2, d2)
        b2.invisible = True
        b1.s_point = low_s
        b2.s_point = high_s
        return b1, b2

    def convert_s_to_bound(self, low_s, high_s):
        if not self.fix_to_path:
            b = self.convert_s_to_bound_general(low_s, high_s)
        else:
            if self.reverse:
                b = self.convert_s_to_bound_on_path(low_s)
            else:
                b = self.convert_s_to_bound_on_path(high_s)
        return b

    def convert_s_to_bound_general(self, s1, s2):
        if self.reverse:
            n2 = (s2 - s1) / np.linalg.norm(s1 - s2)
            d2 = -1 * np.vdot(n2, s1)
        else:
            n2 = (s2 - s1) / np.linalg.norm(s1 - s2)
            d2 = -1 * np.vdot(n2, s2)
        b2 = BXDBound(n2, d2)
        b2.s_point = s2
        return b2

    def convert_s_to_bound_on_path(self, s):
        n = self.progress_metric.get_norm_to_path(s)
        d = -1 * np.vdot(n, s)
        b = BXDBound(n, d)
        b.s_point=s
        return b

    def output(self):
        out = " box = " + str(self.box) + ' path segment = ' + str(self.progress_metric.path_segment) +\
              ' % progress = ' + str(self.progress_metric.project_point_on_path(self.s) / self.progress_metric.end_point)\
              + " bound hit = " + str(self.bound_hit) + " distance from path  = " + str(self.progress_metric.distance_from_path) \
              + ' Sampled points = ' + str(len(self.box_list[self.box].data))
        return out

    def get_converging_bxd(self, hits = 10, decorrelation_limit = 10, boxes_to_converge = []):
        con = Converging(self.progress_metric, self.stuck_limit, bound_hits=hits, decorrelation_limit = decorrelation_limit, boxes_to_converge = boxes_to_converge)
        con.box_list = []
        for b in self.box_list:
            n1 = b.lower.n
            n2 = b.upper.n
            d1 = b.lower.d
            d2 = b.upper.d
            box =BXDBox(BXDBound(n1,d1),BXDBound(n2,d2), 'fixed', True, decorrelation_time = b.decorrelation_time)
            box.geometry = b.geometry
            con.box_list.append(box)
        con.generate_output_files()

        return con

    def boundary_check(self):
        self.path_bound_hit = self.progress_metric.reflect_back_to_path()
        self.bound_hit = 'none'

        #Check for hit against upper boundary
        if self.box_list[self.box].upper.hit(self.s, 'up'):
            if self.reverse and self.box_list[self.box].upper.pause:
                self.bound_hit = 'upper'
                self.box_list[self.box].upper.hits += 1
                return True
            if not self.reverse and self.steps_since_any_boundary_hit > self.decorrelation_limit:
                self.box_list[self.box].upper.pause = False
                self.box_list[self.box].upper.transparent = False
                self.box_list[self.box].lower.transparent = True
                self.box_list[self.box].data = []
                self.box += 1
                self.box_list[self.box].min_segment = self.progress_metric.path_segment
                self.box_list[self.box].data = []
                self.new_box = True
                return False
            else:
                self.bound_hit = 'upper'
                self.box_list[self.box].upper.hits += 1
                return True
        elif self.box_list[self.box].lower.hit(self.s, 'down'):
            if self.box_list[self.box].lower.pause and self.reverse:
                self.reverse = False
                self.box_list[self.box].lower.pause = False
            if self.reverse and not self.box_list[self.box].type == 'adap' and self.steps_since_any_boundary_hit > self.decorrelation_limit:
                self.box_list[self.box].data = []
                self.box -= 1
                self.box_list[self.box].data = []
                self.box_list[self.box].min_segment = self.progress_metric.path_segment
                self.new_box = True
                self.box_list[self.box].max_segment = self.progress_metric.path_segment
                if self.box == 0:
                    self.reverse = False
                    self.completed_runs += 1
                self.box_list[self.box].type = 'adap'
                return False
            else:
                self.bound_hit = 'lower'
                self.box_list[self.box].lower.hits += 1
                if self.reverse:
                    self.box_list[self.box].type = 'normal'
            return True
        else:
            return False

    def final_printing(self, temp_dir, mol):
        box_geoms = open(temp_dir + '/box_geoms.xyz', 'w')
        furthest_geom = open(temp_dir + '/furthest_geometry.xyz', 'w')
        end_geom = open(temp_dir + '/final_geometry.xyz', 'w')
        for box in self.box_list:
            write(box_geoms, box.geometry, format='xyz', append=True)
        write(furthest_geom, self.geometry_of_furthest_point, format='xyz')
        write(end_geom, mol, format='xyz')


class Shrinking(BXD):

    def __init__(self, collective_variable, projection, lower_s, upper_s, stuck_limit=20, bound_file="bounds.txt"):
        super(Shrinking, self).__init__(collective_variable, projection, stuck_limit, bound_file,  fix_to_path=False)
        self.lower_bound, self.upper_bound = self.convert_s_to_bound(lower_s, upper_s)
        self.box = 0
        self.box_list
        self.old_s = lower_s
        self.s = 0


    def update(self, mol):
        # update current and previous s(r) values
        self.s = self.get_s(mol)
        self.inversion = False
        self.bound_hit = "none"

        # Check whether BXD direction should be changed and update accordingly
        self.reached_end()

        # First check whether we are inside the box
        if not self.box_list[self.box].upper.hit(self.s, "up") or not self.box_list[self.box].upper.hit(self.s, "down"):
            # if so box type is now fixed
            self.box_list[self.box].type = "fixed"
            self.box_list[self.box].active = "True"
        # If we are outside the top boundary and moving further away then invert velocities
        elif self.box_list[self.box].upper.hit(self.s, "up") and self.s[0] > self.old_s[0]:
            self.inversion = True

        elif self.box_list[self.box].upper.hit(self.s, "down") and self.s[0] < self.old_s[0]:
            self.inversion = False

            if self.inversion:
                self.stuck_count += 1
                self.steps_since_any_boundary_hit = 0
            else:
                self.steps_since_any_boundary_hit += 1
                self.stuck_count = 0
                self.stuck = False
                self.old_s = self.s
                self.box_list[self.box].upper.step_since_hit += 1
                self.box_list[self.box].lower.step_since_hit += 1
                if not self.progress_metric.reflect_back_to_path():
                    self.box_list[self.box].data.append(self.s)

            if self.stuck_count > self.stuck_limit:
                self.stuck = True
                self.stuck_count = 0



    def get_default_box(self, lower, upper):
        b = BXDBox(lower, upper, "adap", True)
        return b

    def reached_end(self):
        return False

    def boundary_check(self):
        pass

    def stuck_fix(self):
        pass

    def criteria_met(self, boundary):
        return False

    def convert_s_to_bound(self, s1, s2):
        n1 = (s2 - s1) / np.linalg.norm(s1 - s2)
        n2 = (s2 - s1) / np.linalg.norm(s1 - s2)
        D1 = -1 * np.vdot(n1, s1)
        D2 = -1 * np.vdot(n2, s2)
        b1 = BXDBound(n1, D1)
        b2 = BXDBound(n2, D2)
        return b1,b2


class Converging(BXD):
    """
    Derived BXD class controlling a Converging BXD run. This class keeps track of the number of hits on each
    boundary and stores data such as mean first passage times. It also determines when sufficient boundary hits have
    occured to move to the next box. One all the data has been collected, this class also contains methods for
    generating a free energy profile
    :param progress_metric: ProgressMetric object which manages the CollectiveVariable representation, transforms
                            the current MD geometry into "progess" between the BXD start and end points and
                            optionally contains a path object if BXD is following a guess path
    :param stuck_limit: Integer, DEFAULT = 5.
                        Controls the number of subsequent hits at a boundary before BXD is considered stuck and the
                        stuck fix method is called to remedy the situation
    :param box_skip_limit: Integer DEFAULT = 500000
                           Max number of steps without a boundary hit before attempting to move to a the next box
    :param bound_file: String DEFAULT = "bounds.txt"
                       Filename containing a list of BXD boundaries from an adaptive run.
    :param geom_file: String DEFAULT = 'box_geoms.xyz'
                      Filename containing represntative geometries for each box. If this is not present, it will not
                      be possible to skip between boxes
    :param bound_hits: Integer DEFAULT = 100
                       Number of boundary hits before moving to the next box
    :param read_from_file: Boolean DEFAULT = False,
                           If True this tries to read BXD boundaries from the bound_file
    :param convert_fixed_boxes: Boolean DEFAULT = False
                                Niche case where you dont have a bounds file to read from and want to create boxes
                                of a fixed size. NOT TESTED
    :param box_width: If convert_fixed_boxes = True then this defines the width of the box.
    :param number_of_boxes: If convert_fixed_boxes = True then this defines the number of boxes
    :param decorrelation_limit: Integer, DEFAULT = 0
                                A boundary hit / passage is only counted if it occurs decorrelation_limit steps
                                after the previous hit.
    :param boxes_to_converge: List.
                              Specifies a subset of the total boxes which to converge.
                              e.g if boxes_to_converge = [3,6] then only boxes 3 to 6 inclusive will be converged
    :param print_directory: String, DEFAULT="Converging_Data"
                            Directory name for printing converging data. If this directory already exsist the new
                            data will be appended to the exsisting
    :param converge_ends: Boolean DEFAULT = False
                          If True then the start and end boxes will be fully converged. This means that the start
                          box will aim for "bound_hits" at the lower boundary and the top box will aim for
                          "bound_hits" at the top boundary
    :param box_data_print_freqency: Integer, DEFAULT = 10
                                    Frequency at which collective variable and progress metric data for a box is
                                    printed to file. Larger values help prevent files becoming too large for long
                                    BXD runs.
    """
    def __init__(self, progress_metric, stuck_limit=5, box_skip_limit = 500000, bound_file="bounds.txt", geom_file='box_geoms.xyz', bound_hits=100,
                 read_from_file=False, convert_fixed_boxes = False, box_width=0, number_of_boxes=0, decorrelation_limit=0,  boxes_to_converge = [],
                 print_directory='Converging_Data', converge_ends = False, box_data_print_freqency = 10, box_geom_print_frequency=10000  ):

        super(Converging, self).__init__(progress_metric, stuck_limit)
        self.bound_file = bound_file
        self.box_data_print_freqency = box_data_print_freqency
        self.box_geom_print_freqency = box_geom_print_frequency
        self.geom_file = geom_file
        self.box_skip_limit = box_skip_limit
        self.dir = str(print_directory)
        self.read_from_file = read_from_file
        self.converge_ends = converge_ends
        self.convert_fixed_boxes = convert_fixed_boxes
        self.box_width = box_width
        self.number_of_boxes = number_of_boxes
        self.boxes_to_converge = boxes_to_converge
        if self.read_from_file:
            self.box_list = self.read_exsisting_boundaries(self.bound_file, decorrelation_limit)
            self.generate_output_files()
            try:
                self.get_box_geometries(self.geom_file)
            except:
                print('couldnt read box geometries, turning off box skipping')
                self.box_skip_limit = np.inf
        elif self.convert_fixed_boxes:
            self.generate_output_files()
            self.box_list=self.create_fixed_boxes(self.box_width, self.number_of_boxes, progress_metric.start_s, decorrelation_limit)
            self.generate_output_files()
        self.old_s = 0
        self.number_of_hits = bound_hits
        self.decorrelation_limit = decorrelation_limit
        try:
            self.start_box = self.boxes_to_converge[0]
            self.box = self.start_box
            self.end_box = self.boxes_to_converge[1]
        except:
            self.start_box = 0
            self.end_box = len(self.box_list)-1

    def reset(self, output_directory):
        """
        Function reseting a converging BXD object to its orriginal state but with a different output directory
        :param output_directory:
        :return:
        """
        self.__init__(self.progress_metric, self.stuck_limit, self.box_skip_limit, self.bound_file, self.geom_file, self.number_of_hits, self.read_from_file, self.convert_fixed_boxes, self.box_width, self.number_of_boxes, self.decorrelation_limit,  self.boxes_to_converge, output_directory, self.converge_ends)

    def initialise_files(self):
        """
        Open all the output files for the current box
        :return:
        """
        self.upper_rates_file = open(self.box_list[self.box].upper_rates_path, 'a')
        self.upper_milestoning_rates_file = open(self.box_list[self.box].upper_milestoning_rates_path, 'a')
        self.lower_rates_file = open(self.box_list[self.box].lower_rates_path, 'a')
        self.lower_milestoning_rates_file = open(self.box_list[self.box].lower_milestoning_rates_path, 'a')
        self.data_file = open(self.box_list[self.box].data_path, 'a')
        self.hit_file = open(self.box_list[self.box].hit_path, 'a')

    def get_box_geometries(self, file):
        """
        Read a box geometries file generated from an adaptive run and associate each frame with a box.
        :param file:
        :return:
        """
        for i,box in enumerate(self.box_list):
            box.geometry = read(file, index=i)


    def generate_output_files(self):
        """
        Assigns the correct file paths for the output from different boxes. The root output directory is given by
        self.dir and a subdirectory is created for each box.
        :return:
        """
        for i,box in enumerate(self.box_list):
            box.temp_dir = self.dir + ("/box_" + str(i))
            if not os.path.isdir(box.temp_dir):
                os.makedirs(box.temp_dir)
            box.upper_rates_path = box.temp_dir + '/upper_rates.txt'
            box.lower_rates_path = box.temp_dir + '/lower_rates.txt'
            box.upper_milestoning_rates_path = box.temp_dir + '/upper_milestoning.txt'
            box.lower_milestoning_rates_path = box.temp_dir + '/lower_milestoning.txt'
            box.data_path = box.temp_dir + '/box_data.txt'
            box.hit_path = box.temp_dir + '/hits.txt'
            box.geom_path = box.temp_dir + '/geom.xyz'

    def update(self, mol, decorrelated):
        """
        Does the general BXD bookkeeping and management. Firsts gets the progress_metric data from the mol object and
        then calls functions to check whether a BXD inversion is required and whether we need to move to the next box.
        :param mol: ASE atoms object
        :return:
        """

        self.skip_box = False

        # Check the upper and lower bounds to see whether either has had sufficient hits to be made transparent
        self.update_bounds()

        # update current and previous s(r) values
        self.s = self.get_s(mol)

        #if self.box == 0:
        #    min_seg = 0
        #else:
        #    min_seg = self.box_list[self.box-1].max_segment

        #if self.box == len(self.box_list) -1:
        #    max_seg = np.inf
        #else:
        #    max_seg = self.box_list[self.box+1].min_segment

        projected_data = self.progress_metric.project_point_on_path(self.s)

        if self.progress_metric.path_segment < self.box_list[self.box].min_segment:
            self.box_list[self.box].min_segment = self.progress_metric.path_segment
        if self.progress_metric.path_segment > self.box_list[self.box].max_segment:
            self.box_list[self.box].max_segment = self.progress_metric.path_segment

        distance_from_bound = self.progress_metric.get_dist_from_bound(self.s, self.box_list[self.box].lower)
        distance_from_upper = self.progress_metric.get_dist_from_bound(self.s, self.box_list[self.box].upper)

        # make sure to reset the inversion and bound_hit flags to False / none
        self.inversion = False
        self.bound_hit = "none"

        # Check whether BXD direction should be changed and update accordingly
        self.reached_end()


        # Check whether a boundary has been hit and if so update the hit boundary
        self.path_bound_hit = self.progress_metric.reflect_back_to_path()
        if self.path_bound_hit:
            self.hit_file.write("HIT\tType\t=\tPATH\tStep\t=\t" + str(self.box_list[self.box].points_in_box)+ "\n")
        self.inversion = self.boundary_check(decorrelated) or self.path_bound_hit
        if self.inversion and not self.path_bound_hit:
            if self.bound_hit == "upper" :
                self.hit_file.write("HIT\tType\t=\tUPPER\tStep\t=\t" + str(self.box_list[self.box].points_in_box) + "\n")
            else:
                self.hit_file.write("HIT\tType\t=\tLOWER\tStep\t=\t" + str(self.box_list[self.box].points_in_box) + "\n")
        # If there is a BXD inversion increment the stuck counter and set the steps_since_any_boundary_hit counter to 0
        if self.inversion:
            self.stuck_count += 1
            self.steps_since_any_boundary_hit = 0
        # Otherwise increment the appropriate counters
        else:
            self.steps_since_any_boundary_hit += 1
            self.stuck_count = 0
            self.stuck = False
            self.old_s = self.s
            if decorrelated:
                # Consult box_data_print_freqency to determine whether or not print the data to a file
                if self.box_list[self.box].points_in_box != 0 and self.box_list[self.box].points_in_box % self.box_data_print_freqency == 0:
                    line = str(self.s) + '\t' + str(projected_data) + '\t' + str(distance_from_bound) + '\t' + str(mol.get_potential_energy()) + '\t' + str(distance_from_upper)
                    line = line.strip('\n')
                    self.data_file.write( str(line) +'\n')
                if self.box_list[self.box].points_in_box != 0 and self.box_list[self.box].points_in_box % self.box_geom_print_freqency == 0:
                    write(str(self.box_list[self.box].geom_path),mol,append=True)
                self.box_list[self.box].points_in_box += 1
        # Check whether we are stuck in a loop of inversions. If stuck, make the boundary we are stuck at transparent to move to then next box
        if self.stuck_count > self.stuck_limit:
            self.stuck_count = 0
            if self.bound_hit == 'upper':
                self.upper_rates_file.close()
                self.upper_milestoning_rates_file.close()
                self.lower_rates_file.close()
                self.lower_milestoning_rates_file.close()
                self.data_file.close()
                self.hit_file.close()
                self.box_list[self.box].upper_non_milestoning_count = 0
                self.box_list[self.box].lower_non_milestoning_count = 0
                self.box += 1
                self.upper_rates_file = open(self.box_list[self.box].upper_rates_path, 'a')
                self.upper_milestoning_rates_file = open(self.box_list[self.box].upper_milestoning_rates_path, 'a')
                self.lower_rates_file = open(self.box_list[self.box].lower_rates_path, 'a')
                self.lower_milestoning_rates_file = open(self.box_list[self.box].lower_milestoning_rates_path, 'a')
                self.data_file = open(self.box_list[self.box].data_path, 'a')
                self.hit_file = open(self.box_list[self.box].hit_path, 'a')
            elif self.bound_hit == 'lower':
                self.upper_rates_file.close()
                self.upper_milestoning_rates_file.close()
                self.lower_rates_file.close()
                self.lower_milestoning_rates_file.close()
                self.data_file.close()
                self.hit_file.close()
                self.box_list[self.box].upper_non_milestoning_count = 0
                self.box_list[self.box].lower_non_milestoning_count = 0
                self.box -= 1
                self.upper_rates_file = open(self.box_list[self.box].upper_rates_path, 'a')
                self.upper_milestoning_rates_file = open(self.box_list[self.box].upper_milestoning_rates_path, 'a')
                self.lower_rates_file = open(self.box_list[self.box].lower_rates_path, 'a')
                self.lower_milestoning_rates_file = open(self.box_list[self.box].lower_milestoning_rates_path, 'a')
                self.data_file = open(self.box_list[self.box].data_path, 'a')
                self.hit_file = open(self.box_list[self.box].hit_path, 'a')
        # Check whether we have reach the box_skip_limit and alter the box accordingly.
        if self.reverse and self.box_list[self.box].lower.step_since_hit > self.box_skip_limit:
            self.skip_box = True
            self.upper_rates_file.close()
            self.upper_milestoning_rates_file.close()
            self.lower_rates_file.close()
            self.lower_milestoning_rates_file.close()
            self.data_file.close()
            self.hit_file.close()
            self.box_list[self.box].upper_non_milestoning_count = 0
            self.box_list[self.box].lower_non_milestoning_count = 0
            self.box -= 1
            self.upper_rates_file = open(self.box_list[self.box].upper_rates_path, 'a')
            self.upper_milestoning_rates_file = open(self.box_list[self.box].upper_milestoning_rates_path, 'a')
            self.lower_rates_file = open(self.box_list[self.box].lower_rates_path, 'a')
            self.lower_milestoning_rates_file = open(self.box_list[self.box].lower_milestoning_rates_path, 'a')
            self.data_file = open(self.box_list[self.box].data_path, 'a')
            self.hit_file = open(self.box_list[self.box].hit_path, 'a')
        if not self.reverse and self.box_list[self.box].upper.step_since_hit > self.box_skip_limit:
            self.skip_box = True
            self.upper_rates_file.close()
            self.upper_milestoning_rates_file.close()
            self.lower_rates_file.close()
            self.lower_milestoning_rates_file.close()
            self.data_file.close()
            self.box_list[self.box].upper_non_milestoning_count = 0
            self.box_list[self.box].lower_non_milestoning_count = 0
            self.box += 1
            self.upper_rates_file = open(self.box_list[self.box].upper_rates_path, 'a')
            self.upper_milestoning_rates_file = open(self.box_list[self.box].upper_milestoning_rates_path, 'a')
            self.lower_rates_file = open(self.box_list[self.box].lower_rates_path, 'a')
            self.lower_milestoning_rates_file = open(self.box_list[self.box].lower_milestoning_rates_path, 'a')
            self.data_file = open(self.box_list[self.box].data_path, 'a')
            self.hit_file = open(self.box_list[self.box].hit_path, 'a')


    def create_fixed_boxes(self, width, number_of_boxes, start_s, decorrelation_limit):
        box_list = []
        s = deepcopy(start_s)
        lower_bound = BXDBound(1.0,-1.0*deepcopy(s))
        for i in range(0,number_of_boxes):
            s += width
            upper_bound = BXDBound(1.0,-1.0*deepcopy(s))
            box = BXDBox(lower_bound, upper_bound, "fixed", True, decorrelation_time=decorrelation_limit)
            box_list.append(box)
            lower_bound = deepcopy(upper_bound)
        return box_list

    def read_exsisting_boundaries(self,file, decorrelation_limit):
        """
        Read BXD boundaries from a file
        :param file: string giving the filename of the bounds file
        :return:
        """
        box_list = []
        lines = open(file,"r").readlines()
        for i in range(2, len(lines)-1):
            words = lines[i].split("\t")
            d_lower = (float(words[4]))
            n_l = (words[7]).strip("[]\n")
            norm_lower = (n_l.split())
            for l in range(0,len(norm_lower)):
                norm_lower[l] = float(norm_lower[l])
            lower_bound = BXDBound(norm_lower,d_lower)
            words = lines[i+1].split("\t")
            d_upper = (float(words[4]))
            n_u = (words[7]).strip("[]\n")
            norm_upper = (n_u.split())
            for l2 in range(0,len(norm_upper)):
                norm_upper[l2] = float(norm_upper[l2])
            upper_bound = BXDBound(norm_upper,d_upper)
            box = BXDBox(lower_bound, upper_bound, "fixed", True, decorrelation_time=decorrelation_limit)
            box_list.append(box)
        return box_list

    def reached_end(self):
        """
        Checks whether the converging run either needs to be reversed or whether it is complete
        :return:
        """
        # First if we are not currently reversing check whether or not we have reached the end box and reversing should
        # be turned on.
        if self.box == self.end_box and self.reverse is False:
            # If converge_ends then make sure the final bound meets the bound_hits criteria before reversing
            if not self.converge_ends or self.criteria_met(self.box_list[self.box].upper):
                self.reverse = True
                self.progress_metric.set_bxd_reverse(self.reverse)
                print('reversing')
        # If we are reversing then check whether we are back in box 0 and the run is complete
        elif self.box == self.start_box and self.reverse is True:
            self.reverse = False
            self.progress_metric.set_bxd_reverse(self.reverse)
            self.completed_runs += 1
            for bx in self.box_list:
                bx.upper.hits=0
                bx.lower.hits=0

    def stuck_fix(self):
        pass

    def criteria_met(self, boundary):
        """
        Check whether a boundary has exceeded the specified number of hits
        :param boundary: BXDbound object
        :return:
        """
        return boundary.hits >= self.number_of_hits

    def get_free_energy(self, T, boxes=1, milestoning = False, directory = 'Converging_Data', decorrelation_limit = 1, data_frequency=1):
        """
        Reads the data in the output directory to calculate the free_energy profile
        :param T: Temperature MD was run at in K
        :param boxes: Integer DEFAULT = 1
                      NB needs renaming. Controls the resolution of the free energy profile. Each bxd box with be
                      histogrammed into "boxes" subboxes
        :param milestoning: Boolean DEFAULT = False
                            If True the milestoning rates files will be used, otherwise normal rates files will be used
        :param directory: String DEFAULT = 'Converging_Data"
                          Name of output directory to be read
        :param decorrelation_limit: Integer DEFAULT = 1
                                    Only rates in excess of decorrelation_limit will be read
        :return:
        """
        # Multiply T by the gas constant in kJ/mol
        T *= (8.314 / 1000)
        for i,box in enumerate(self.box_list):
            temp_dir = directory + ("/box_" + str(i) + '/')
            try:
                box.upper.average_rates(milestoning, 'upper', temp_dir, decorrelation_limit)
            except:
                box.lower.average_rate = 0
            try:
                box.lower.average_rates(milestoning, 'lower', temp_dir, decorrelation_limit)
            except:
                box.lower.average_rate = 0
            try:
                box.read_box_data(temp_dir)
            except:
                pass

        for i in range(0, len(self.box_list) - 1):
            if i == 0:
                self.box_list[i].gibbs = 0
                self.box_list[i].gibbs_err = 0
            try:
                k_eq = self.box_list[i].upper.average_rate / self.box_list[i + 1].lower.average_rate
                K_eq_err = k_eq * np.sqrt((self.box_list[i].upper.rate_error/self.box_list[i].upper.average_rate)**2 + (self.box_list[i+1].lower.rate_error/self.box_list[i+1].lower.average_rate)**2)
                try:
                    delta_g = -1.0 * np.log(k_eq)
                except:
                    delta_g = 0
                delta_g_err = (K_eq_err) / k_eq
                self.box_list[i + 1].gibbs = delta_g + self.box_list[i].gibbs
                self.box_list[i + 1].gibbs_err = delta_g_err + self.box_list[i].gibbs_err
            except:
                self.box_list[i+1].gibbs = 0
                self.box_list[i+1].gibbs_err = 0
        if boxes == 1:
            profile = []
            for i in range(0, len(self.box_list)):
                try:
                    enedata = [float(d[3]) for d in self.box_list[i].data]
                    ave_ene = min(np.asarray(enedata))
                except:
                    ave_ene = "nan"
                profile.append((str(i), self.box_list[i].gibbs, self.box_list[i].gibbs_err, ave_ene))
            return profile
        else:
            try:
                profile = []
                total_probability =0
                for i in range(0, len(self.box_list)):
                    self.box_list[i].eq_population = np.exp(-1.0 * (self.box_list[i].gibbs))
                    self.box_list[i].eq_population_err = self.box_list[i].eq_population * (1) * self.box_list[i].gibbs_err
                    total_probability += self.box_list[i].eq_population


                for i in range(0, len(self.box_list)):
                    self.box_list[i].eq_population /= total_probability
                    self.box_list[i].eq_population_err /= total_probability
                last_s = 0
                for i in range(0, len(self.box_list)):
                    s, dens,ene= self.box_list[i].get_full_histogram(boxes,data_frequency)
                    for sj in s:
                        sj -= s[0]
                    for j in range(0, len(dens)):
                        d_err = np.sqrt(float(dens[j])) / (float(len(self.box_list[i].data))/data_frequency)
                        d = float(dens[j]) / (float(len(self.box_list[i].data))/data_frequency)
                        p = d * self.box_list[i].eq_population
                        p_err = p * np.sqrt((d_err / d) ** 2 + (self.box_list[i].eq_population_err / self.box_list[i].eq_population) ** 2)
                        p_log = -1.0 * np.log(p)
                        p_log_err = (p_err) / p
                        s_path = s[j] + last_s
                        profile.append((s_path, p_log, p_log_err, ene[j]))
                    last_s += s[-1]
                return profile
            except:
                print('couldnt find histogram data for high resolution profile')

    def get_alternate_free_energy(self, T, boxes=15, milestoning = False, directory = 'Converging_Data', decorrelation_limit = 1):
        """
        Reads the data in the output directory to calculate the free_energy profile. This is an experimental variation
        on the standard function.
        :param T: Temperature MD was run at in K
        :param boxes: Integer DEFAULT = 1
                      NB needs renaming. Controls the resolution of the free energy profile. Each bxd box with be
                      histogrammed into "boxes" subboxes
        :param milestoning: Boolean DEFAULT = False
                            If True the milestoning rates files will be used, otherwise normal rates files will be used
        :param directory: String DEFAULT = 'Converging_Data"
                          Name of output directory to be read
        :param decorrelation_limit: Integer DEFAULT = 1
                                    Only rates in excess of decorrelation_limit will be read
        :return:
        """
        # Multiply T by the gas constant in kJ/mol
        T *= (8.314 / 1000)
        for i,box in enumerate(self.box_list):
            temp_dir = directory + ("/box_" + str(i) + '/')
            try:
                box.upper.average_rates(milestoning, 'upper', temp_dir, decorrelation_limit)
            except:
                box.lower.average_rate = 0
            try:
                box.lower.average_rates(milestoning, 'lower', temp_dir, decorrelation_limit)
            except:
                box.lower.average_rate = 0
            try:
                box.read_box_data(temp_dir)
            except:
                pass

        for i in range(0, len(self.box_list) - 1):
            if i == 0:
                self.box_list[i].gibbs = 0
                self.box_list[i].gibbs_err = 0
            try:
                k_eq = self.box_list[i].upper.average_rate / self.box_list[i + 1].lower.average_rate
                K_eq_err = k_eq * np.sqrt((self.box_list[i].upper.rate_error/self.box_list[i].upper.average_rate)**2 + (self.box_list[i+1].lower.rate_error/self.box_list[i+1].lower.average_rate)**2)
                try:
                    delta_g = -1.0 * np.log(k_eq) * T
                except:
                    delta_g = 0
                delta_g_err = (T * K_eq_err) / k_eq
                self.box_list[i + 1].gibbs = delta_g + self.box_list[i].gibbs
                self.box_list[i + 1].gibbs_err = delta_g_err
            except:
                self.box_list[i+1].gibbs = 0
                self.box_list[i+1].gibbs_err = 0

            total_probability = 0
            for i in range(0, len(self.box_list)):
                self.box_list[i].eq_population = np.exp(-1.0 * (self.box_list[i].gibbs / T))
                self.box_list[i].eq_population_err = self.box_list[i].eq_population * (1 / T) * self.box_list[i].gibbs_err
                total_probability += self.box_list[i].eq_population

            all_data = []
            for i in range(0, len(self.box_list)):
                self.box_list[i].eq_population /= total_probability
                self.box_list[i].eq_population_err /= total_probability
                all_data += self.box_list[i].get_modified_box_data()

            profile = self.histogram_full_profile(all_data, T, boxes)
            return profile

    def histogram_full_profile(self, data, T, boxes):
        data1 = data[0::10000]
        data2 = [float(d[0]) for d in data1]
        top = max(data2)
        edges = []
        energies = []
        for i in range(0, boxes + 1):
            edges.append(i * (top / boxes))
        hist = np.zeros(boxes)

        for j in range(0, boxes):
            temp_ene = []
            for d in data:
                if float(d[0]) > edges[j] and float(d[0]) <= edges[j + 1]:
                    hist[j] += d[1]
                    temp_ene.append(float(d[2]))
            temp_ene = np.asarray(temp_ene)
            energies.append(np.mean(temp_ene))

        cumulative_probability = 0
        for h in hist:
            cumulative_probability += h

        hist_array = []
        for h in hist:
            his = h / cumulative_probability
            his = -1.0 * np.log(his) * T
            hist_array.append(his)

        profile = []
        for e,h,ene in zip(edges,hist_array,energies):
            profile.append([e,h,ene])

        return profile


    def collate_free_energy_data(self, prefix = 'Converging_Data', outfile = 'Combined_converging'):
        """
        Collates data from a number of different output directories with the same prefix into a new directory
        :param prefix: String DEFAULT = 'Converging_Data'
                       Prefix of input directories to be read
        :param outfile: String DEFAULT = 'Combined_converging'
                        Filename for output directory
        :return:
        """
        dir_root_list = []
        number_of_boxes = []
        for subdir, dirs, files in os.walk(os.getcwd()):
            for dir in dirs:
                if prefix in dir:
                    dir_root_list.append(dir)
                    number_of_boxes.append(len(next(os.walk(dir))[1]))
        # Check number of boxes is consistant among directories
        if number_of_boxes.count(number_of_boxes[0]) == len(number_of_boxes):
            boxes = number_of_boxes[0]
        else:
            boxes = min(number_of_boxes)
            print('not all directories have the same number of boxes. Check these all correspond to the same system')

        os.mkdir(outfile)
        for i in range(0, boxes):
            os.mkdir(outfile+'/box_'+str(i))
            u_rates = [(root +'/box_'+str(i)+'/upper_rates.txt') for root in dir_root_list]
            with open(outfile+'/box_'+str(i)+'/upper_rates.txt', 'w') as outfile0:
                for u in u_rates:
                    try:
                        with open(u) as infile:
                            for line in infile:
                                outfile0.write(line)
                    except:
                        pass
            u_m = [(root +'/box_'+str(i)+'/upper_milestoning.txt') for root in dir_root_list]
            with open(outfile +'/box_'+str(i)+'/upper_milestoning.txt', 'w') as outfile1:
                for um in u_m:
                    try:
                        with open(um) as infile:
                            for line in infile:
                                outfile1.write(line)
                    except:
                        pass
            l_rates = [(root +'/box_'+str(i)+'/lower_rates.txt') for root in dir_root_list]
            with open(outfile+'/box_'+str(i)+'/lower_rates.txt', 'w') as outfile2:
                for l in l_rates:
                    try:
                        with open(l) as infile:
                            for line in infile:
                                outfile2.write(line)
                    except:
                        pass
            l_m = [(root +'/box_'+str(i)+'/lower_milestoning.txt') for root in dir_root_list]
            with open(outfile+'/box_'+str(i)+'/lower_milestoning.txt', 'w') as outfile3:
                for lm in l_m:
                    try:
                        with open(lm) as infile:
                            for line in infile:
                                outfile3.write(line)
                    except:
                        pass
            data = [(root +'/box_'+str(i)+'/box_data.txt') for root in dir_root_list]
            with open(outfile+'/box_'+str(i)+'/box_data.txt', 'w') as outfile4:
                for d in data:
                    try:
                        with open(d) as infile:
                            for line in infile:
                                if line.rstrip():
                                    outfile4.write(line)
                    except:
                        pass




    def boundary_check(self, decorrelated):
        """
        Check upper and lower boundaries for hits and return True if an inversion is required. Also determines the mean
        first passage times for hits against each bound.
        :return: Boolean indicating whether or not a BXD inversion should be performed
        """
        self.bound_hit = 'none'
        self.box_list[self.box].milestoning_count += 1
        self.box_list[self.box].upper_non_milestoning_count += 1
        self.box_list[self.box].lower_non_milestoning_count += 1

        #Check for hit against upper boundary
        if self.box_list[self.box].upper.hit(self.s, 'up'):
            self.bound_hit = 'upper'
            if self.progress_metric.outside_path():
                return True
            elif self.box_list[self.box].upper.transparent and not self.path_bound_hit:
                self.box_list[self.box].upper.transparent = False
                self.box_list[self.box].milestoning_count = 0
                self.box_list[self.box].upper_non_milestoning_count = 0
                self.box_list[self.box].lower_non_milestoning_count = 0
                self.hit_file.write("PASS\tUPPER\tStep\t=\t" + str(self.box_list[self.box].points_in_box) + "HITS LOWER UPPER" + "\t" + str(self.box_list[self.box].upper.hits ) + "\t" + str(self.box_list[self.box].lower.hits) +  "\n")
                self.upper_rates_file.close()
                self.upper_milestoning_rates_file.close()
                self.lower_rates_file.close()
                self.lower_milestoning_rates_file.close()
                self.data_file.close()
                self.hit_file.close()
                self.box += 1
                self.upper_rates_file = open(self.box_list[self.box].upper_rates_path, 'a')
                self.upper_milestoning_rates_file = open(self.box_list[self.box].upper_milestoning_rates_path, 'a')
                self.lower_rates_file = open(self.box_list[self.box].lower_rates_path, 'a')
                self.lower_milestoning_rates_file = open(self.box_list[self.box].lower_milestoning_rates_path, 'a')
                self.data_file = open(self.box_list[self.box].data_path, 'a')
                self.hit_file = open(self.box_list[self.box].hit_path, 'a')
                self.box_list[self.box].milestoning_count = 0
                self.box_list[self.box].upper_non_milestoning_count = 0
                self.box_list[self.box].lower_non_milestoning_count = 0
                self.box_list[self.box].last_hit = 'lower'

                return False
            else:
                self.bound_hit = 'upper'
                if decorrelated:
                    self.box_list[self.box].upper.hits += 1
                    self.upper_rates_file.write(str(self.box_list[self.box].upper_non_milestoning_count) + '\t' + '\n')
                    if self.box_list[self.box].last_hit == 'lower':
                        self.upper_milestoning_rates_file.write(str(self.box_list[self.box].milestoning_count) + '\n')
                        self.box_list[self.box].milestoning_count = 0
                self.box_list[self.box].decorrelation_count = 0
                self.box_list[self.box].last_hit = 'upper'
                self.box_list[self.box].upper_non_milestoning_count = 0
                if self.box_list[self.box].last_hit == 'lower':
                    self.box_list[self.box].milestoning_count = 0
                return True
        elif self.box_list[self.box].lower.hit(self.s, 'down'):
            self.bound_hit = 'lower'
            if self.progress_metric.outside_path():
                return True
            if self.box_list[self.box].lower.transparent and not self.path_bound_hit:
                self.box_list[self.box].lower.transparent = False
                self.box_list[self.box].milestoning_count = 0
                self.box_list[self.box].upper_non_milestoning_count = 0
                self.box_list[self.box].lower_non_milestoning_count = 0
                self.box_list[self.box].decorrelation_count = 0
                self.hit_file.write("PASS\tLOWER\tStep\t=\t" + str(self.box_list[self.box].points_in_box) + "HITS LOWER UPPER" + "\t" + str(self.box_list[self.box].upper.hits) + "\t" + str(self.box_list[self.box].lower.hits) + "\n")
                self.upper_rates_file.close()
                self.upper_milestoning_rates_file.close()
                self.lower_rates_file.close()
                self.lower_milestoning_rates_file.close()
                self.data_file.close()
                self.hit_file.close()
                self.box -= 1
                self.upper_rates_file = open(self.box_list[self.box].upper_rates_path, 'a')
                self.upper_milestoning_rates_file = open(self.box_list[self.box].upper_milestoning_rates_path, 'a')
                self.lower_rates_file = open(self.box_list[self.box].lower_rates_path, 'a')
                self.lower_milestoning_rates_file = open(self.box_list[self.box].lower_milestoning_rates_path, 'a')
                self.data_file = open(self.box_list[self.box].data_path, 'a')
                self.hit_file = open(self.box_list[self.box].hit_path, 'a')
                self.box_list[self.box].milestoning_count = 0
                self.box_list[self.box].lower_non_milestoning_count = 0
                self.box_list[self.box].upper_non_milestoning_count = 0
                self.box_list[self.box].last_hit = 'upper'
                return False
            else:
                self.bound_hit = 'lower'
                if decorrelated:
                    self.box_list[self.box].lower.hits += 1
                    self.lower_rates_file.write(str(self.box_list[self.box].lower_non_milestoning_count) + '\n')
                    if self.box_list[self.box].last_hit == 'upper':
                        self.lower_milestoning_rates_file.write(str(self.box_list[self.box].milestoning_count) + '\n')
                        self.box_list[self.box].milestoning_count = 0
                self.box_list[self.box].last_hit = 'lower'
                self.box_list[self.box].lower_non_milestoning_count = 0
                if self.box_list[self.box].last_hit == 'upper':
                    self.box_list[self.box].milestoning_count = 0
                return True
        else:
            return False

    def final_printing(self, temp_dir, mol):
        limits = open(temp_dir + '/box_limits', 'w')
        for box in self.box_list:
            limits.write('lowest box point = ' + str(min(box.data)) + ' highest box point = ' + str(max(box.data) + '\n'))
        if self.full_output_printing:
            for i, box2 in enumerate(self.box_list):
                temp_dir = self.dir + ("/box_" + str(i))
                if not os.path.isdir(temp_dir):
                    os.makedirs(temp_dir)
                file = open(temp_dir + '/final_geometry.xyz', 'w')
                file.write(str(box2.data))


    def output(self):
        out = (" box = " + str(self.box) + ' Lower Bound Hits = ' + str(self.box_list[self.box].lower.hits)
               + ' Upper Bound Hits = ' + str(self.box_list[self.box].upper.hits) + ' path segment = '
               + str(self.progress_metric.path_segment) + ' % progress = ' +
               str(self.progress_metric.project_point_on_path(self.s) / self.progress_metric.end_point) +
               " distance from path  = " + str(self.progress_metric.distance_from_path) +
               ' Sampled points = ' + str(len(self.box_list[self.box].data)))
        return out

class BXDBox:

    def __init__(self, lower, upper, type, act, decorrelation_time = 0):
        self.upper = upper
        self.lower = lower
        self.type = type
        self.active = act
        # store all s values
        self.data = []
        self.top_data = []
        self.top = 0
        self.bot_data = []
        self.bot = 0
        self.eq_population = 0
        self.eq_population_err = 0
        self.gibbs = 0
        self.gibbs_err = 0
        self.last_hit = 'lower'
        self.milestoning_count = 0
        self.upper_non_milestoning_count = 0
        self.lower_non_milestoning_count = 0
        self.decorrelation_count = 0
        self.points_in_box = 0
        self.decorrelation_time = decorrelation_time
        self.min_segment = np.inf
        self.max_segment = 0

    def reset(self, type, active):
        self.type = type
        self.active = active
        self.hits = 0
        self.stuck_count = 0
        self.transparent = False
        self.step_since_hit = 0
        self.rates = []
        self.average_rate = 0
        self.rate_error = 0
        self.invisible = False
        self.s_point = 0
        self.upper.reset()
        self.lower.reset()
        self.last_hit = 'lower'
        self.milestoning_count = 0
        self.decorrelation_count = 0
        self.decorrelation_time = 0


    def get_s_extremes(self, b, eps):
        self.top_data = []
        self.bot_data = []
        data = [d[1] for d in self.data]
        hist, edges = np.histogram(data, bins=b)
        cumulative_probability = 0
        cumulative_probability2 = 0
        limit = 0
        limit2 = 0
        for h in range(0, len(hist)):
            cumulative_probability += hist[h] / len(data)
            if cumulative_probability > eps:
                limit = h
                break
        for i,h in enumerate(hist):
            cumulative_probability2 += h / len(data)
            if cumulative_probability2 > (1 - eps):
                limit2 = i
                break
        if limit == 0:
            limit = len(hist) - 2
        for d in self.data:
            if d[1] > edges[limit] and d[1] <= edges[limit + 1]:
                self.top_data.append(d[0])
        self.top = np.mean(self.top_data, axis=0)
        for d in self.data:
            if d[1] >= edges[limit2] and d[1] < edges[limit2 + 1]:
                self.bot_data.append(d[0])
        self.bot = np.mean(self.bot_data, axis=0)

    def get_modified_box_data(self):
        modified_data = []
        for d in self.data:
            normalisation_factor = (float(self.eq_population) / float(len(self.data)))
            ar = [d[1], normalisation_factor, d[3]]
            modified_data.append(ar)
        return modified_data

    def get_full_histogram(self, boxes=10, data_frequency=1):
        data1 = self.data[0::data_frequency]
        d = np.asarray([np.fromstring(d[0].replace('[', '').replace(']', ''), dtype=float, sep=' ') for d in data1])
        proj = np.asarray([float(d[2]) for d in data1])
        edge = (max(proj) - min(proj)) / boxes
        edges = np.arange(min(proj), max(proj),edge).tolist()
        energy = np.asarray([float(d[3]) for d in data1])
        sub_bound_list = self.get_sub_bounds(boxes)
        hist = [0] * boxes
        energies = []
        for j in range(0, boxes):
            temp_ene = []
            for ene,da in zip(energy,d):
                try:
                    if not (sub_bound_list[j+1].hit(da,"up")) and not (sub_bound_list[j].hit(da,"down")):
                        hist[j] += 1
                        temp_ene.append(float(ene))
                except:
                    pass
            try:
                temp_ene = np.asarray(temp_ene)
                energies.append(np.mean(temp_ene))
            except:
                energies.append(0)
        return edges, hist, energies

    def get_sub_bounds(self, boxes):
        # Get difference between upper and lower boundaries
        n_diff = np.subtract(self.upper.n,self.lower.n)
        d_diff = self.upper.d - self.lower.d

        # now divide this difference by the number of boxes
        n_increment = np.true_divide(n_diff, boxes)
        d_increment = d_diff / boxes

        # create a set of "boxes" new bounds divide the space between the upper and lower bounds,
        # these bounds all intersect at the same point in space

        bounds = []
        for i in range(0,boxes):
            new_n = self.lower.n + i * n_increment
            new_d = self.lower.d + i * d_increment
            b = BXDBound(new_n,new_d)
            bounds.append(b)
        bounds.append(deepcopy(self.upper))
        return bounds

    def convert_s_to_bound(self, lower, upper):
        pass

    def read_box_data(self, path):
        path += '/box_data.txt'
        file = open(path, 'r')
        for line in file.readlines():
            line = line.rstrip('\n')
            line = line.split('\t')
            if float(line[2]) >= 0:
                self.data.append(line)




class BXDBound:

    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.hits = 0
        self.stuck_count = 0
        self.transparent = False
        self.step_since_hit = 0
        self.rates = []
        self.average_rate = 0
        self.rate_error = 0
        self.invisible = False
        self.s_point = 0
        self.pause = False

    def reset(self):
        self.hits = 0
        self.stuck_count = 0
        self.transparent = False
        self.step_since_hit = 0
        self.rates = []
        self.average_rate = 0
        self.rate_error = 0
        self.invisible = False
        self.s_point = 0

    def get_data(self):
        return self.d, self.n, self.s_point

    def hit(self, s, bound):
        if self.invisible:
            return False
        coord = np.vdot(s, self.n) + self.d
        if bound == "up" and coord > 0:
            return True
        elif bound == "down" and coord < 0:
            return True
        else:
            return False

    def average_rates(self, milestoning, bound, path, decorrelation_limit, prune=False):
        if milestoning:
            if bound == 'upper':
                path += '/upper_milestoning.txt'
            else:
                path += '/lower_milestoning.txt'
        else:
            if bound == 'upper':
                path += '/upper_rates.txt'
            else:
                path += '/lower_rates.txt'
        file = open(path, 'r')
        rates = np.loadtxt(file)
        if prune:
            length=int(len(rates)/10)
            rates = np.sort(rates)
            rates = rates[:-length]
            rates=rates[length:]
        maxi = np.max(rates)
        if maxi > 2.0 * decorrelation_limit:
            rates = rates[rates > decorrelation_limit]
        else:
            rates = rates[rates > 2]
        self.rates = 1.0 / rates
        self.average_rate = np.mean(self.rates)
        self.rate_error = np.std(self.rates) / np.sqrt(len(self.rates))
