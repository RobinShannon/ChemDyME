from abc import ABCMeta, abstractmethod
import numpy as np


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
        self.complete_runs = 0
        self.delta = 0

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

    @abstractmethod
    def convert_s_to_bound(self, lower, upper):
        pass

    def getDefaultBox(self,lower, upper):
        b = BXDBox(lower, upper,"fixed",True)
        return b

    # Determine whether boundary should be transparent or reflective
    def update_bounds(self):
        if self.reverse and (self.criteria_met(self.box_list[self.box].lower)):
            self.box_list[self.box].lower.transparent = True
        elif self.reverse is False and (self.criteria_met(self.box_list[self.box].upper)):
            self.box_list[self.box].upper.transparent = True

    def del_constraint(self, mol):
        if self.bound_hit == 'upper':
            norm = self.box_list[self.box].upper.n
        if self.bound_hit == 'lower':
            norm = self.box_list[self.box].upper.n
        self.delta = self.progress_metric.get_delta(mol, norm)
        return self.delta

    def path_del_constraint(self, mol):
        return self.progress_metric.get_path_delta(mol)

    def get_s(self, mol):
        return self.progress_metric.get_s(mol)


class Adaptive(BXD):

    def __init__(self, progress_metric, stuck_limit=20,  fix_to_path=False,
                 adaptive_steps=1000, epsilon=0.9, reassign_rate=2):

        super(Adaptive, self).__init__(progress_metric, stuck_limit)
        self.fix_to_path = fix_to_path
        self.adaptive_steps = adaptive_steps
        self.histogram_boxes = int(np.sqrt(adaptive_steps))
        self.epsilon = epsilon
        self.reassign_rate = reassign_rate
        self.completed_runs = 0
        s1, s2 = self.progress_metric.get_start_s()
        b1, b2 = self.get_starting_bounds(s1, s2)
        box = self.get_default_box(b1, b2)
        self.box_list.append(box)


    def update(self, mol):
        # update current and previous s(r) values
        self.s = self.get_s(mol)
        projected_data = self.progress_metric.project_point_on_path(self.s)
        self.inversion = False
        self.bound_hit = "none"

        # Check whether BXD direction should be changed and update accordingly
        self.reached_end(projected_data)

        # Check whether we are in an adaptive sampling regime.
        # If so update_adaptive_bounds checks current number of samples and controls new boundary placement
        if self.box_list[self.box].type == "adap":
            self.update_adaptive_bounds()

        # If we have sampled for a while and not hot the upper bound then reassign the boundary.
        # Only how often the boundary is reassigned depends upon the reasign_rate parameter
        if self.box_list[self.box].type == "normal" and len(self.box_list[self.box].data) > \
                self.reassign_rate * self.adaptive_steps:
            self.reassign_boundary()

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
                self.box_list[self.box].data.append((self.s, projected_data))

        if self.stuck_count > self.stuck_limit:
            self.stuck = True
            self.stuck_count = 0

    def update_adaptive_bounds(self):
        if len(self.box_list[self.box].data) > self.adaptive_steps:
            # If adaptive sampling has ended then add a boundary based up sampled data
            self.box_list[self.box].type = "normal"
            if not self.reverse:
                self.box_list[self.box].get_s_extremes(self.histogram_boxes, self.epsilon)
                bottom = self.box_list[self.box].bot
                top = self.box_list[self.box].top
                b1 = self.convert_s_to_bound(bottom, top)
                b2 = b1
                b3 = BXDBound(self.box_list[self.box].upper.n, self.box_list[self.box].upper.d)
                b3.invisible = True
                self.box_list[self.box].upper = b1
                self.box_list[self.box].upper.transparent = True
                new_box = self.get_default_box(b2, b3)
                self.box_list.append(new_box)
            elif self.reverse:
                # at this point we partition the box into two and insert a new box at the correct point in the boxList
                self.box_list[self.box].get_s_extremes_reverse(self.histogram_boxes, self.epsilon)
                bottom = self.box_list[self.box].bot
                top = self.box_list[self.box].top
                b1 = self.convert_s_to_bound(bottom, top)
                b2 = self.convert_s_to_bound(bottom, top)
                b3 = BXDBound(self.box_list[self.box].lower.norm, self.box_list[self.box].lower.D)
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
                    self.reverse = True
                    self.progress_metric.set_bxd_reverse(self.reverse)
            elif self.progress_metric.end_type == 'boxes':
                if self.box >= self.progress_metric.end_point and self.box_list[self.box].type != 'adap':
                    self.reverse = True
                    self.progress_metric.set_bxd_reverse(self.reverse)
        else:
            if self.box == 0:
                self.completed_runs += 1
                self.reverse = False
                self.progress_metric.set_bxd_reverse(self.reverse)

    def boundary_check(self):
        self.path_bound_hit = self.progress_metric.reflect_back_to_path()
        self.bound_hit = 'none'

        #Check for hit against upper boundary
        if self.box_list[self.box].upper.hit(self.s, 'up'):
            if self.box_list[self.box].upper.transparent and not self.path_bound_hit:
                self.box_list[self.box].upper.transparent = False
                self.box_list[self.box].data = []
                self.box += 1
                return False
            else:
                self.bound_hit = 'upper'
                return True
        elif self.box_list[self.box].lower.hit(self.s, 'down'):
            if self.box_list[self.box].lower.transparent and not self.path_bound_hit:
                self.box_list[self.box].lower.transparent = False
                self.box_list[self.box].data = []
                self.box -= 1
                if self.box == 0:
                    self.reverse = False
                    self.complete_runs += 1
                self.box_list[self.box].type = 'adap'
                self.box_list[self.box].data = []
                self.box_list[self.box].lower.transparent = True
                return False
            else:
                self.bound_hit = 'lower'
                return True
        else:
            return False

    def reassign_boundary(self):
        fix = self.fix_to_path
        self.fixToPath = False
        if self.reverse:
            self.box_list[self.box].get_s_extremes_reverse(self.histogram_boxes, self.epsilon)
        else:
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
        pass

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
        n2 = (s2 - s1) / np.linalg.norm(s1 - s2)
        if self.reverse:
            d2 = -1 * np.vdot(n2, s1)
        else:
            d2 = -1 * np.vdot(n2, s2)
        b2 = BXDBound(n2, d2)
        return b2

    def convert_s_to_bound_on_path(self, s):
        n = self.progress_metric.get_norm_to_path()
        d = -1 * np.vdot(n, s)
        b = BXDBound(n, d)
        return b

    def getDefaultBox(self, lower, upper):
        b = BXDBox(lower, upper, 'adap', False)
        return b

    def output(self):
        out = " box = " + str(self.box) + ' path segment = ' + str(self.progress_metric.path_segment) +\
              ' total projection = ' + str(self.progress_metric.project_point_on_path(self.s)) + " bound hit = " \
              + str(self.bound_hit) + " distance from path  = " + str(self.progress_metric.distance_from_path)
        return out


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

    def __init__(self, collective_variable, progress_metric, lower_s, upper_s, stuck_limit=20, bound_file="bounds.txt",
                 read_from_file=True, box_width=0, number_of_boxes=0):

        super(Converging, self).__init__(collective_variable, stuck_limit)
        if read_from_file:
            self.box_list = self.read_exsisting_boundaries(bound_file)
        else:
            self.create_fixed_boxes(box_width, number_of_boxes, progress_metric.start)
        self.old_s = 0
        self.s = progress_metric.start

    @classmethod
    def convert_from_adaptive(self, adaptive_bxd, bound_hits=100):
        super(Converging, self).__init__(adaptive_bxd.progress_metric, adaptive_bxd.stuck_limit)
        for b in self.box_list:
            b.reset('fixed', True)


    def update(self, mol):
        # update current and previous s(r) values
        self.s = self.get_s(mol)
        self.inversion = False
        self.bound_hit = "none"

        # Check whether BXD direction should be changed and update accordingly
        self.reached_end()

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

    def create_fixed_boxes(self, width, number_of_boxes, start_s):
        lower = self.convert_s_to_bound(start_s)
        for i in range(0,number_of_boxes):
            upper_s = lower.s + (width * lower.n)
            upper = self.convert_s_to_bound(upper_s)
            self.boxList.append(self.get_default_box(lower, upper))
            lower = upper

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
            box = BXDBox(lower_bound,upper_bound, "fixed", True)
            box_list.append(box)
        box_list.pop(0)
        return box_list

    def reached_end(self):
        if self.box == len(self.box_list):
            self.reverse = True
            self.projection
            return True
        else:
            return False

    def boundary_check(self):
        pass

    def stuck_fix(self):
        pass

    def criteria_met(self, boundary):
        return False

    def convert_s_to_bound(self, s1):
        n1 = (s2 - s1) / np.linalg.norm(s1 - s2)
        n2 = (s2 - s1) / np.linalg.norm(s1 - s2)
        D1 = -1 * np.vdot(n1, s1)
        D2 = -1 * np.vdot(n2, s2)
        b1 = BXDBound(n1, D1)
        b2 = BXDBound(n2, D2)
        b2.invisible = True
        return b1, b2

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
        self.gibbs = 0

    def rest(self, type, active):
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


    def get_s_extremes(self, b, eps):
        self.top_data = []
        self.bot_bata = []
        data = [d[1] for d in self.data]
        hist, edges = np.histogram(data, bins=b)
        cumulative_probability = 0
        limit = 0
        for h in range(0, len(hist)):
            cumulative_probability += hist[h] / len(data)
            if cumulative_probability > eps:
                limit = h
                break
        if limit == 0:
            limit = len(data) - 1
        for d in self.data:
            if d[1] > edges[limit] and d[1] <= edges[limit + 1]:
                self.top_data.append(d[0])
        self.top = np.mean(self.top_data, axis=0)
        for d in self.data:
            if d[1] >= edges[0] and d[1] < edges[1]:
                self.bot_data.append(d[0])
        self.bot = np.mean(self.bot_data, axis=0)

    def get_s_extremes_reverse(self, b, eps):
        self.top_data = []
        self.bot_data = []
        data = [d[2] for d in self.data]
        hist, edges = np.histogram(data, bins=b)
        cumulative_probability = 0
        limit = 0
        for h in range(0, len(hist)):
            cumulative_probability += hist[h] / len(data)
            if cumulative_probability > (1 - eps):
                limit = h
                break
        if limit == 0:
            limit = len(data) - 1
        for d in self.data:
            if d[2] > edges[-2] and d[2] <= edges[-1]:
                self.top_data.append(d[0])
        self.top = np.mean(self.top_data, axis=0)
        for d in self.data:
            if d[2] >= edges[limit] and d[2] < edges[limit + 1]:
                self.bot_data.append(d[0])
        self.bot = np.mean(self.bot_data, axis=0)

    def getFullHistogram(self):
        del self.data[0]
        data = [d[1] for d in self.data]
        top = max(data)
        edges = []
        for i in range(0, 11):
            edges.append(i * (top / 10))
        hist = np.zeros(10)
        for d in data:
            for j in range(0, 10):
                if d > edges[j] and d <= edges[j + 1]:
                    hist[j] += 1
        return edges, hist


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

    def average_rates(self):
        self.average_rate = np.mean(self.rates)
        self.rate_error = np.std(self.rates)
