from abc import ABCMeta, abstractmethod
import numpy as np
from ase.io import read, write
import os
from copy import deepcopy


class BXD(metaclass=ABCMeta):
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

    def getDefaultBox(self,lower, upper):
        b = BXDBox(lower, upper,"fixed",True)
        return b

    # Determine whether boundary should be transparent or reflective
    def update_bounds(self):
        if self.reverse:
            self.box_list[self.box].upper.transparent = False
        else:
            self.box_list[self.box].lower.transparent = False

        if self.reverse and (self.criteria_met(self.box_list[self.box].lower)):
            self.box_list[self.box].lower.transparent = True
        elif self.reverse is False and (self.criteria_met(self.box_list[self.box].upper)):
            self.box_list[self.box].upper.transparent = True

    def del_constraint(self, mol):
        if self.bound_hit == 'upper':
            norm = self.box_list[self.box].upper.n
        if self.bound_hit == 'lower':
            norm = self.box_list[self.box].lower.n
        self.delta = self.progress_metric.get_delta(mol, norm)
        return self.delta

    def path_del_constraint(self, mol):
        return self.progress_metric.get_path_delta(mol)

    def get_s(self, mol):
        return self.progress_metric.get_s(mol)

    def print_bounds(self, file="bounds.txt"):
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

    def __init__(self, progress_metric, stuck_limit=5,  fix_to_path=False,
                 adaptive_steps=1000, epsilon=0.9, reassign_rate=2, incorporate_distance_from_bound = False, one_direction = False, decorellation_limit = 0):

        super(Adaptive, self).__init__(progress_metric, stuck_limit)
        self.fix_to_path = fix_to_path
        self.one_direction = one_direction
        self.adaptive_steps = adaptive_steps
        self.histogram_boxes = int(np.sqrt(adaptive_steps))
        self.epsilon = epsilon
        self.reassign_rate = reassign_rate
        self.completed_runs = 0
        s1, s2 = self.progress_metric.get_start_s()
        b1, b2 = self.get_starting_bounds(s1, s2)
        box = self.get_default_box(b1, b2)
        self.box_list.append(box)
        self.skip_box = False
        self.incorporate_distance_from_bound = False
        self.steps_since_any_boundary_hit = 0
        self.decorellation_limit = decorellation_limit


    def update(self, mol):

        if self.new_box:
            self.box_list[self.box].geometry = mol.copy()
        self.new_box = False

        # update current and previous s(r) values
        self.s = self.get_s(mol)
        projected_data = self.progress_metric.project_point_on_path(self.s)
        if self.incorporate_distance_from_bound:
            if not self.reverse:
                projected_data += self.progress_metric.get_dist_from_bound(self.s,self.box_list[self.box].lower)
            else:
                projected_data += self.progress_metric.get_dist_from_bound(self.s, self.box_list[self.box].upper)


        self.inversion = False
        self.bound_hit = "none"


        # Check whether we are in an adaptive sampling regime.
        # If so update_adaptive_bounds checks current number of samples and controls new boundary placement
        if self.box_list[self.box].type == "adap":
            self.update_adaptive_bounds()

        self.reached_end(projected_data)

        # If we have sampled for a while and not hot the upper bound then reassign the boundary.
        # Only how often the boundary is reassigned depends upon the reasign_rate parameter
        if self.box_list[self.box].type == "normal" and len(self.box_list[self.box].data) > \
                self.reassign_rate * self.adaptive_steps:
            self.reassign_boundary()

        # Check whether a boundary has been hit and if so update whether the hit boundary
        if self.stuck:
            self.stuck_fix()
            self.inversion = False
            self.stuck = False
        self.inversion = self.boundary_check() or self.path_bound_hit
        self.steps_since_any_boundary_hit += 1
        if self.inversion:
            self.steps_since_any_boundary_hit = 0

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
                self.box_list[self.box].data.append((self.s, projected_data))
                if projected_data > self.furthest_progress:
                    self.furthest_progress = projected_data
                    self.geometry_of_furthest_point = mol.copy()

        if self.stuck_count > self.stuck_limit:
            self.stuck = True


    def update_adaptive_bounds(self):
        if len(self.box_list[self.box].data) > self.adaptive_steps:
            # If adaptive sampling has ended then add a boundary based up sampled data
            self.box_list[self.box].type = "normal"
            if not self.reverse:
                self.box_list[self.box].get_s_extremes(self.histogram_boxes, self.epsilon)
                bottom = self.box_list[self.box].bot
                top = self.box_list[self.box].top
                b1 = self.convert_s_to_bound(bottom, top)
                b2 = deepcopy(b1)
                self.box_list[self.box].angle_between_bounds = np.arccos(np.dot(b1.n,self.box_list[self.box].lower.n)/(np.linalg.norm(b1.n)*np.linalg.norm(self.box_list[self.box].lower.n)))
                b3 = deepcopy(self.box_list[self.box].upper)
                b3.invisible = True
                b3.s_point = self.box_list[self.box].upper.s_point
                self.box_list[self.box].upper = b1
                self.box_list[self.box].upper.transparent = True
                new_box = self.get_default_box(b2, b3)
                self.box_list.append(new_box)
            elif self.reverse:
                print("making new adaptive bound in reverse direction")
                # at this point we partition the box into two and insert a new box at the correct point in the boxList
                self.box_list[self.box].get_s_extremes(self.histogram_boxes, self.epsilon)
                bottom = self.box_list[self.box].bot
                top = self.box_list[self.box].top
                b1 = self.convert_s_to_bound(bottom, top)
                b2 = deepcopy(b1)
                b3 = deepcopy(self.box_list[self.box].lower)
                self.box_list[self.box].lower = b1
                new_box = self.get_default_box(b3, b2)
                self.box_list.insert(self.box, new_box)
                self.box_list[self.box].active = True
                self.box_list[self.box].upper.transparent = False
                self.box += 1
                self.box_list[self.box].lower.transparent = True

    def get_default_box(self, lower, upper):
        b = BXDBox(lower, upper, "adap", True)
        return b

    def reached_end(self, projected):
        if not self.reverse:
            if self.progress_metric.end_type == 'distance':
                if projected >= self.progress_metric.end_point and self.box_list[self.box].type != 'adap':
                    if self.one_direction:
                        self.completed_runs += 1
                        del self.box_list[-1]
                    else:
                        self.reverse = True
                        self.box_list[self.box].type = 'adap'
                        self.box_list[self.box].data = []
                        del self.box_list[-1]
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

    def getDefaultBox(self, lower, upper):
        b = BXDBox(lower, upper, 'adap', False)
        return b

    def output(self):
        out = " box = " + str(self.box) + ' path segment = ' + str(self.progress_metric.path_segment) +\
              ' % progress = ' + str(self.progress_metric.project_point_on_path(self.s) / self.progress_metric.end_point)\
              + " bound hit = " + str(self.bound_hit) + " distance from path  = " + str(self.progress_metric.distance_from_path)
        return out

    def get_converging_bxd(self, hits = 10, decorrelation_limit = 10, boxes_to_converge = []):
        con = Converging(self.progress_metric, self.stuck_limit, bound_hits=hits, decorrelation_limit = decorrelation_limit, boxes_to_converge = boxes_to_converge)
        con.box_list = []
        for b in self.box_list:
            n1 = b.lower.n
            n2 = b.upper.n
            d1 = b.lower.d
            d2 = b.upper.d
            box =BXDBox(BXDBound(n1,d1),BXDBound(n2,d2), 'fixed', True)
            box.geometry = b.geometry
            con.box_list.append(box)
        con.generate_output_files()

        return con

    def boundary_check(self):
        self.path_bound_hit = self.progress_metric.reflect_back_to_path()
        self.bound_hit = 'none'

        #Check for hit against upper boundary
        if self.box_list[self.box].upper.hit(self.s, 'up'):
            if not self.reverse and self.steps_since_any_boundary_hit > self.decorellation_limit:
                self.box_list[self.box].upper.transparent = False
                self.box_list[self.box].lower.transparent = True
                self.box_list[self.box].data = []
                self.box += 1
                self.box_list[self.box].data = []
                self.new_box = True
                return False
            else:
                self.bound_hit = 'upper'
                self.box_list[self.box].upper.hits += 1
                return True
        elif self.box_list[self.box].lower.hit(self.s, 'down'):
            if self.reverse and not self.box_list[self.box].type == 'adap' and self.steps_since_any_boundary_hit > self.decorellation_limit:
                self.box_list[self.box].data = []
                self.box -= 1
                self.box_list[self.box].data = []
                self.new_box = True
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

    def __init__(self, progress_metric, stuck_limit=5, box_skip_limit = 500000, bound_file="bounds.txt", geom_file='box_geoms.xyz', bound_hits=100,
                 read_from_file=False, convert_fixed_boxes = False, box_width=0, number_of_boxes=0, decorrelation_limit=10,  boxes_to_converge = [],
                 print_directory='Converging_Data', converge_ends = False, box_data_print_freqency = 10):

        super(Converging, self).__init__(progress_metric, stuck_limit)
        self.bound_file = bound_file
        self.box_data_print_freqency = box_data_print_freqency
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
            self.box_list = self.read_exsisting_boundaries(self.bound_file)
            self.generate_output_files()
            try:
                self.get_box_geometries(self.geom_file)
            except:
                print('couldnt read box geometries, turning off box skipping')
                self.box_skip_limit = np.inf
        elif self.convert_fixed_boxes:
            self.generate_output_files()
            self.create_fixed_boxes(self.box_width, self.number_of_boxes, progress_metric.start)
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
        self.__init__(self.progress_metric, self.stuck_limit, self.box_skip_limit, self.bound_file, self.geom_file, self.number_of_hits, self.read_from_file, self.convert_fixed_boxes, self.box_width, self.number_of_boxes, self.decorrelation_limit,  self.boxes_to_converge, output_directory, self.converge_ends)

    def initialise_files(self):
        self.upper_rates_file = open(self.box_list[self.box].upper_rates_path, 'a')
        self.upper_milestoning_rates_file = open(self.box_list[self.box].upper_milestoning_rates_path, 'a')
        self.lower_rates_file = open(self.box_list[self.box].lower_rates_path, 'a')
        self.lower_milestoning_rates_file = open(self.box_list[self.box].lower_milestoning_rates_path, 'a')
        self.data_file = open(self.box_list[self.box].data_path, 'a')

    def get_box_geometries(self, file):
        for i,box in enumerate(self.box_list):
            box.geometry = read(file, index=i)


    def generate_output_files(self):
        for i,box in enumerate(self.box_list):
            temp_dir = self.dir + ("/box_" + str(i))
            if not os.path.isdir(temp_dir):
                os.makedirs(temp_dir)
            box.upper_rates_path = temp_dir + '/upper_rates.txt'
            box.lower_rates_path = temp_dir + '/lower_rates.txt'
            box.upper_milestoning_rates_path = temp_dir + '/upper_milestoning.txt'
            box.lower_milestoning_rates_path = temp_dir + '/lower_milestoning.txt'
            box.data_path = temp_dir + '/box_data.txt'

    def update(self, mol):

        self.skip_box = False

        self.update_bounds()
        # update current and previous s(r) values
        self.s = self.get_s(mol)
        projected_data = self.progress_metric.project_point_on_path(self.s)
        distance_from_bound = self.progress_metric.get_dist_from_bound(self.s, self.box_list[self.box].lower)
        self.inversion = False
        self.bound_hit = "none"

        # Check whether BXD direction should be changed and update accordingly
        self.reached_end()

        # Check whether a boundary has been hit and if so update whether the hit boundary
        self.inversion = self.boundary_check() or self.path_bound_hit



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
                if len(self.box_list[self.box].data) % self.box_data_print_freqency == 0:
                    self.data_file.write(str(self.s) + '\t' + str(projected_data) + '\t' + str(distance_from_bound) + '\t' + str(mol.get_potential_energy()) + '\n')

        if self.stuck_count > self.stuck_limit:
            self.stuck_count = 0
            if self.bound_hit == 'upper':
                self.upper_rates_file.close()
                self.upper_milestoning_rates_file.close()
                self.lower_rates_file.close()
                self.lower_milestoning_rates_file.close()
                self.data_file.close()
                self.box += 1
                self.upper_rates_file = open(self.box_list[self.box].upper_rates_path, 'a')
                self.upper_milestoning_rates_file = open(self.box_list[self.box].upper_milestoning_rates_path, 'a')
                self.lower_rates_file = open(self.box_list[self.box].lower_rates_path, 'a')
                self.lower_milestoning_rates_file = open(self.box_list[self.box].lower_milestoning_rates_path, 'a')
                self.data_file = open(self.box_list[self.box].data_path, 'a')
            elif self.bound_hit == 'lower':
                self.upper_rates_file.close()
                self.upper_milestoning_rates_file.close()
                self.lower_rates_file.close()
                self.lower_milestoning_rates_file.close()
                self.data_file.close()
                self.box -= 1
                self.upper_rates_file = open(self.box_list[self.box].upper_rates_path, 'a')
                self.upper_milestoning_rates_file = open(self.box_list[self.box].upper_milestoning_rates_path, 'a')
                self.lower_rates_file = open(self.box_list[self.box].lower_rates_path, 'a')
                self.lower_milestoning_rates_file = open(self.box_list[self.box].lower_milestoning_rates_path, 'a')
                self.data_file = open(self.box_list[self.box].data_path, 'a')

        if self.reverse and self.box_list[self.box].lower.step_since_hit > self.box_skip_limit:
            self.skip_box = True
            self.upper_rates_file.close()
            self.upper_milestoning_rates_file.close()
            self.lower_rates_file.close()
            self.lower_milestoning_rates_file.close()
            self.data_file.close()
            self.box_list[self.box].lower.step_since_hit
            self.box -= 1
            self.upper_rates_file = open(self.box_list[self.box].upper_rates_path, 'a')
            self.upper_milestoning_rates_file = open(self.box_list[self.box].upper_milestoning_rates_path, 'a')
            self.lower_rates_file = open(self.box_list[self.box].lower_rates_path, 'a')
            self.lower_milestoning_rates_file = open(self.box_list[self.box].lower_milestoning_rates_path, 'a')
            self.data_file = open(self.box_list[self.box].data_path, 'a')
        if not self.reverse and self.box_list[self.box].upper.step_since_hit > self.box_skip_limit:
            self.skip_box = True
            self.upper_rates_file.close()
            self.upper_milestoning_rates_file.close()
            self.lower_rates_file.close()
            self.lower_milestoning_rates_file.close()
            self.data_file.close()
            self.box_list[self.box].upper.step_since_hit = 0
            self.box += 1
            self.upper_rates_file = open(self.box_list[self.box].upper_rates_path, 'a')
            self.upper_milestoning_rates_file = open(self.box_list[self.box].upper_milestoning_rates_path, 'a')
            self.lower_rates_file = open(self.box_list[self.box].lower_rates_path, 'a')
            self.lower_milestoning_rates_file = open(self.box_list[self.box].lower_milestoning_rates_path, 'a')
            self.data_file = open(self.box_list[self.box].data_path, 'a')



    def create_fixed_boxes(self, width, number_of_boxes, start_s):
        pass

    def read_exsisting_boundaries(self,file):
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
            box = BXDBox(lower_bound, upper_bound, "fixed", True)
            box_list.append(box)
        return box_list

    def reached_end(self):
        if self.box == self.end_box and self.reverse is False:
            if not self.converge_ends or self.criteria_met(self.box_list[self.box].upper):
                self.reverse = True
                self.progress_metric.set_bxd_reverse(self.reverse)
                print('reversing')
        elif self.box == self.start_box and self.reverse is True:
            if not self.converge_ends or self.criteria_met(self.box_list[self.box].lower):
                self.reverse = False
                self.progress_metric.set_bxd_reverse(self.reverse)
                self.completed_runs += 1

    def stuck_fix(self):
        pass

    def criteria_met(self, boundary):
        return boundary.hits >= self.number_of_hits

    def get_free_energy(self,T, boxes=1, milestoning = False, directory = 'Converging_Data', decorrelation_limit = 1):
        # Multiply T by the gas constant in kJ/mol
        T *= (8.314 / 1000)
        profile_high_res = []
        profile_low_res = []
        total_probability = 0
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
        if boxes == 1:
            profile = []
            for i in range(0, len(self.box_list)):
                profile.append((str(i), self.box_list[i].gibbs, self.box_list[i].gibbs_err))
            return profile
        else:
            try:
                profile = []
                total_probability =0
                for i in range(0, len(self.box_list)):
                    self.box_list[i].eq_population = np.exp(-1.0 * (self.box_list[i].gibbs / T))
                    self.box_list[i].eq_population_err = self.box_list[i].eq_population * (1 / T) * self.box_list[i].gibbs_err
                    total_probability += self.box_list[i].eq_population


                for i in range(0, len(self.box_list)):
                    self.box_list[i].eq_population /= total_probability
                    self.box_list[i].eq_population_err /= total_probability
                last_s = 0
                for i in range(0, len(self.box_list)):
                    s, dens = self.box_list[i].get_full_histogram(boxes)
                    for j in range(0, len(dens)):
                        d_err = 1/np.sqrt(float(dens[j]))
                        d = float(dens[j]) / float(len(self.box_list[i].data))
                        p = d * self.box_list[i].eq_population
                        p_err = p * np.sqrt((d_err / d) ** 2 + (self.box_list[i].eq_population_err / self.box_list[i].eq_population) ** 2)
                        p = -1.0 * np.log(p) * T
                        p_err = (T * p_err) / p
                        s_path = s[j] + last_s
                        profile.append((s_path, p, p_err))
                    last_s += s[-1]
                return profile
            except:
                print('couldnt find histogram data for high resolution profile')

    def collate_free_energy_data(self, prefix = 'Converging_Data', outfile = 'Combined_converging'):
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




    def boundary_check(self):
        self.path_bound_hit = self.progress_metric.reflect_back_to_path()
        self.bound_hit = 'none'
        if self.path_bound_hit:
            self.box_list[self.box].decorrelation_count = 0
        else:
            self.box_list[self.box].decorrelation_count += 1
        self.box_list[self.box].milestoning_count += 1
        self.box_list[self.box].non_milestoning_count += 1

        #Check for hit against upper boundary
        if self.box_list[self.box].upper.hit(self.s, 'up'):
            if self.progress_metric.outside_path():
                self.bound_hit = 'upper'
                return True
            elif self.box_list[self.box].upper.transparent and not self.path_bound_hit:
                self.box_list[self.box].upper.transparent = False
                self.box_list[self.box].milestoning_count = 0
                self.box_list[self.box].non_milestoning_count = 0
                self.upper_rates_file.close()
                self.upper_milestoning_rates_file.close()
                self.lower_rates_file.close()
                self.lower_milestoning_rates_file.close()
                self.data_file.close()
                self.box += 1
                self.upper_rates_file = open(self.box_list[self.box].upper_rates_path, 'a')
                self.upper_milestoning_rates_file = open(self.box_list[self.box].upper_milestoning_rates_path, 'a')
                self.lower_rates_file = open(self.box_list[self.box].lower_rates_path, 'a')
                self.lower_milestoning_rates_file = open(self.box_list[self.box].lower_milestoning_rates_path, 'a')
                self.data_file = open(self.box_list[self.box].data_path, 'a')
                self.box_list[self.box].milestoning_count = 0
                self.box_list[self.box].non_milestoning_count = 0
                self.box_list[self.box].last_hit = 'lower'
                return False
            else:
                self.bound_hit = 'upper'
                if self.box_list[self.box].decorrelation_count > self.decorrelation_limit:
                    self.box_list[self.box].upper.hits += 1
                    self.upper_rates_file.write(str(self.box_list[self.box].non_milestoning_count) + '\n')
                    self.box_list[self.box].non_milestoning_count = 0
                    if self.box_list[self.box].last_hit == 'lower':
                        self.upper_milestoning_rates_file.write(str(self.box_list[self.box].milestoning_count) + '\n')
                        self.box_list[self.box].milestoning_count = 0
                self.box_list[self.box].decorrelation_count = 0
                self.box_list[self.box].last_hit = 'upper'
                return True
        elif self.box_list[self.box].lower.hit(self.s, 'down'):
            if self.progress_metric.outside_path():
                self.bound_hit = 'lower'
                return True
            if self.box_list[self.box].lower.transparent and not self.path_bound_hit:
                self.box_list[self.box].lower.transparent = False
                self.box_list[self.box].milestoning_count = 0
                self.box_list[self.box].non_milestoning_count = 0
                self.box_list[self.box].decorrelation_count = 0
                self.upper_rates_file.close()
                self.upper_milestoning_rates_file.close()
                self.lower_rates_file.close()
                self.lower_milestoning_rates_file.close()
                self.data_file.close()
                self.box -= 1
                self.upper_rates_file = open(self.box_list[self.box].upper_rates_path, 'a')
                self.upper_milestoning_rates_file = open(self.box_list[self.box].upper_milestoning_rates_path, 'a')
                self.lower_rates_file = open(self.box_list[self.box].lower_rates_path, 'a')
                self.lower_milestoning_rates_file = open(self.box_list[self.box].lower_milestoning_rates_path, 'a')
                self.data_file = open(self.box_list[self.box].data_path, 'a')
                self.box_list[self.box].milestoning_count = 0
                self.box_list[self.box].non_milestoning_count = 0
                self.box_list[self.box].last_hit = 'upper'
                if self.box == 0:
                    self.reverse = False
                    self.completed_runs += 1
                return False
            else:
                self.bound_hit = 'lower'
                if self.box_list[self.box].decorrelation_count > self.decorrelation_limit:
                    self.box_list[self.box].lower.hits += 1
                    self.lower_rates_file.write(str(self.box_list[self.box].non_milestoning_count) + '\n')
                    self.box_list[self.box].non_milestoning_count = 0
                    if self.box_list[self.box].last_hit == 'upper':
                        self.lower_milestoning_rates_file.write(str(self.box_list[self.box].milestoning_count) + '\n')
                        self.box_list[self.box].milestoning_count = 0
                self.box_list[self.box].last_hit = 'lower'
                self.box_list[self.box].decorrelation_count = 0
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
        out = " box = " + str(self.box) + ' Lower Bound Hits = ' + str(self.box_list[self.box].lower.hits) + \
              ' Upper Bound Hits = ' + str(self.box_list[self.box].upper.hits)
        return out

class BXDBox:

    def __init__(self, lower, upper, type, act):
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
        self.non_milestoning_count = 0
        self.decorrelation_count = 0

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
            cumulative_probability += h / len(data)
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

    def get_full_histogram(self, boxes=10):
        data = [float(d[2]) for d in self.data]
        data = np.sort(data)
        data = data[0:-50]
        top = max(data)
        edges = []
        for i in range(0, boxes + 1):
            edges.append(i * (top / boxes))
        hist = np.zeros(boxes)
        for d in data:
            for j in range(0, boxes):
                if d > edges[j] and d <= edges[j + 1]:
                    hist[j] += 1
        return edges, hist

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

    def average_rates(self, milestoning, bound, path, decorrelation_limit):
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
        rates = rates[rates > decorrelation_limit]
        self.rates = 1 / rates
        self.average_rate = np.mean(self.rates)
        self.rate_error = np.std(self.rates) / np.sqrt(len(self.rates))
