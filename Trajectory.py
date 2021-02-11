import numpy as np
from ase.md import velocitydistribution as vd
import ase.io as io
import os
from concurrent.futures import ProcessPoolExecutor as PE
import copy
from ChemDyME.OpenMMCalc import OpenMMCalculator
from statistics import mean

class Trajectory:
    """
    Controls the running of a BXD trajectory. This class interfaces with a BXD object to track whether or not a
    boundary has been hit and then uses an attatched md_integrator object to propagate the dynamics.
    :param mol: ASE atoms object
    :param bxd: A BXDconstraint object containing the details of the BXD bounds etc
    :param md_integrator: An MDintegrator object controlling the propagation of the trajectory
    :param geo_print_frequency: Frequency at which the geometry is printed to file
    :param data_print_freqency: Frequency at which collective variable data is printed to file
    :param plot_update_frequency: If a plotter is attached this determines the frequency at which the plot is update
                                  The attached plotter is EXPERIMENTAL currently
    :param no_text_output: If True this disable printing to stdout
    :param plot_output: If True the trajectory will try to interface with a BXD_plotter object EXPERIMENTAL
    :param plotter: BXD plotter object
    :param calc: Type of calculator to be used. This currently doesnt do anything and only openMM canb be used
    :param calcMethod: The name of the openMM xml file for setting up a calculator. TODO The calculator should be
                       add outside the trajectory class but is temperarily added here while playing with
                       multiprocess
    """

    def __init__(self, mol, bxd, md_integrator, geo_print_frequency=1000, data_print_frequency=1000,log_print_frequency = 1000,
                 plot_update_frequency=100, no_text_output=False, plot_output=False, plotter=None, calc = 'openMM',
                 calcMethod = 'sys.xml', initialise_velocities = True, decorrelation_limit = 0, check_decorrelation=False,
                 decorrelation_length = 5000, number_of_decorrelation_runs = 5, max_runs = 1):
        self.decorrelation_length = decorrelation_length
        self.number_of_decorrelation_runs = number_of_decorrelation_runs
        self.check_decorrelation = check_decorrelation
        self.decorrelation_limit = decorrelation_limit
        self.bxd = bxd
        self.max_runs = max_runs
        self.calc = calc
        self.calcMethod = calcMethod
        self.md_integrator = md_integrator
        self.geo_print_frequency = geo_print_frequency
        self.data_print_frequency = data_print_frequency
        self.log_print_frequency = log_print_frequency
        self.forces = np.zeros(mol.get_positions().shape)
        self.mol = mol.copy()
        self.mol._calc = mol.get_calculator()
        if initialise_velocities == True:
            initial_temperature = md_integrator.temperature
            vd.MaxwellBoltzmannDistribution(self.mol, initial_temperature, force_temp=True)
        self.md_integrator.current_velocities = self.mol.get_velocities()
        self.md_integrator.half_step_velocity = self.mol.get_velocities()
        self.ReactionCountDown = 0
        self.bounds = [None] * 2
        self.no_text_output = no_text_output
        self.plot = False
        if plot_output:
            self.plot = True
            self.bxd_plotter = plotter
        self.plot_update_frequency = plot_update_frequency

    def __copy__(self):
        """
        Function to copy a trajectory object. OpenMM calculators objects are not picklable so for now the calculator is
        assigned in the run_trajectory method
        :return: A trajectory object
        """
        mol = self.mol.copy()
        return Trajectory(mol, copy.deepcopy(self.bxd), copy.deepcopy(self.md_integrator), calcMethod = self.calcMethod)


    def run_trajectory(self, max_steps = np.inf, parallel = False, print_to_file = True, print_directory = 'BXD_data'):
        """
        Runs a bxd trajectory until either the attached BXDconstraint indicates sufficient information has been obtained
        or the max_steps parameter is exceeded
        :param max_steps: DEFAULT np.inf
                          Maximum number of steps in MD trajectory
        :param parallel: Boolean DEFAULT False
                         If True then multiple copies of the trajectory are being run by multiprocess and this affects
                         how the calculator is attached
        :param print_to_file: Boolean DEFAULT False
                              Determines whether to print trajectory data to a file
        :param print_directory: String DEFAULT 'BXD_data'
                                Prefix for output directory for printing data
        :return:
        """
        hit = False
        previous_hit = 'none'
        if self.calc=='openMM':
            if parallel:
                self.mol.set_calculator(OpenMMCalculator(self.calcMethod, self.mol, parallel=True))
            else:
                self.mol.set_calculator(OpenMMCalculator(self.calcMethod, self.mol))

        print(str(self.mol.get_potential_energy))
        # If print_to_file = True then setup an output directory. If the print_directory already exists then append
        # consecutive numbers to the "print_directory" prefix until a the name does not correspond to an existing
        # directory.

        dir = str(print_directory)
        i=1
        temp_dir = dir
        while os.path.isdir(temp_dir):
            temp_dir = dir + ("_" + str(i))
            i += 1
        os.mkdir(temp_dir)
        data_file = open(temp_dir+'/data.txt', 'w')
        geom_file = open(temp_dir+'/geom.xyz', 'w')
        bound_file = open(temp_dir+'/bound_file.txt', 'w')
        log_file = open(temp_dir+'/logfile','w')
        hit_file = open(temp_dir+'/hits','w')
        double_hit_file = open(temp_dir + '/double_hits', 'w')
        # depending upon the type of BXD object this function does some initial setup
        self.bxd.initialise_files()

        decorrelated = True
        old_box = -1
        # Set up boolean for while loop to determine whether the trajectory loop should keep going
        keep_going = True

        # Get forces from atoms
        try:
            self.forces = self.mol.get_forces()
        except:
            print('forces error')

        # Want to make sure the trajectory doesnt try to perform and BXD inversion on the first MD step. This shouldnt happen.
        # While first_run = True, we will override the bxd inversion.
        first_run = True
        steps_since_last_hit = 0
        iterations = 0
        Ts = []
        # Run MD trajectory for specified number of steps or until BXD reaches its end point
        while keep_going:

            if self.bxd.box > old_box and self.check_decorrelation:
                velocities = copy.deepcopy(self.md_integrator.current_velocities)
                positions = copy.deepcopy(self.md_integrator.current_positions)
                vaf_ar = []
                turning_point_ar = []
                for i in range(0,self.number_of_decorrelation_runs):
                    vaf, turning_point = self.get_correlation_time(self.decorrelation_length)
                    vaf_ar.append(vaf)
                    turning_point_ar.append(turning_point)
                    self.md_integrator.current_positions = copy.deepcopy(positions)
                    self.md_integrator.old_positions = copy.deepcopy(positions)
                    self.md_integrator.current_velocities = copy.deepcopy(velocities)
                    self.md_integrator.old_velocities = copy.deepcopy(velocities)
                t_point = np.mean(np.asarray(turning_point_ar))
                self.bxd.box_list[self.bxd.box].decorrelation_time = t_point
                correlation_file = open(self.bxd.box_list[self.bxd.box].temp_dir + '/correlation.txt', 'w')
                correlation_file.write("Decorrelation time = " + str(t_point) + str("\n"))
                for i,tp in enumerate(vaf_ar):
                    correlation_file.write("Time profile " + str(i) + "\n")
                    for j in tp:
                        correlation_file.write(str(j) + "\n")
                correlation_file.close()
            else:
                for b in  self.bxd.box_list:
                    b.decorrelation_time = self.decorrelation_limit


            old_box = self.bxd.box

            del_phi = []
            # update the BXDconstraint with the current geometry
            self.bxd.update(self.mol, decorrelated)
            # If the trajectory has been stuck in a box for too long then the bxd object will set skip_box = True
            # In that case try to alter the molecular geometry so that it moves to the next BXD box
            if self.bxd.skip_box:
                if self.bxd.reverse:
                    self.mol.set_positions(self.bxd.box_list[self.bxd.box - 1].geometry.get_positions())
                else:
                    self.mol.set_positions(self.bxd.box_list[self.bxd.box + 1].geometry.get_positions())
                vd.MaxwellBoltzmannDistribution(self.mol, self.md_integrator.temperature, force_temp=True)
                self.md_integrator.current_velocities = self.mol.get_velocities()
                self.md_integrator.half_step_velocity = self.mol.get_velocities()
            # Check whether the bxd object indicates a boundary hit
            bounded = self.bxd.inversion
            if bounded and first_run:
                print("BXD bound hit on first step, either there is a small rounding error or there is something wrong with the initial geometry or bound. Proceed with caution")
                bounded = False
            if bounded:
                # If we have hit a bound we need to determine whether we have hit the upper / lower boundary of the box
                # the path boundary or whether both are hit. The del_phi list is populated with each constraint.
                if self.bxd.bound_hit != 'none':
                    del_phi.append(self.bxd.del_constraint(self.mol))
                    decorrelated = True
                if self.bxd.path_bound_hit:
                    del_phi.append(self.bxd.path_del_constraint(self.mol))
                    steps_since_last_hit = 0
                    decorrelated = False
                prior_pos = copy.deepcopy(self.bxd.get_s(self.md_integrator.current_positions))
                prior_vel =  copy.deepcopy(self.bxd.get_s(self.md_integrator.current_positions+(self.md_integrator.old_velocities * self.md_integrator.timestep)))
                prior_v_vel = copy.deepcopy(self.bxd.get_s(self.md_integrator.current_positions+(self.md_integrator.old_velocities * self.md_integrator.timestep)))
                prior_v_pos = copy.deepcopy(self.bxd.get_s(self.md_integrator.verlet_positions))
                prior_accel = copy.deepcopy(self.bxd.get_s(self.md_integrator.old_positions+(self.md_integrator.timestep * 0.5 * self.md_integrator.accel)))
                self.bxd.get_s(self.md_integrator.current_velocities)
                # If we have hit a bound get the md object to modify the velocities / positions appropriately.
                self.md_integrator.constrain(del_phi)
                post_pos = copy.deepcopy(self.bxd.get_s(self.md_integrator.current_positions))
                post_vel =  copy.deepcopy(self.bxd.get_s(self.md_integrator.current_positions+(self.md_integrator.current_velocities * self.md_integrator.timestep)))
                hit_post_contraint = self.bxd.box_list[self.bxd.box].lower.hit(self.bxd.get_s(self.md_integrator.current_positions), 'down') or self.bxd.box_list[self.bxd.box].upper.hit(self.bxd.get_s(self.md_integrator.current_positions), 'up')
            if self.bxd.path_bound_hit and self.bxd.bound_hit != 'none' and previous_hit == 'none':
                previous_hit = 'path and ' + str(self.bxd.bound_hit)
            elif self.bxd.path_bound_hit and previous_hit == 'none':
                previous_hit = 'path'
            else:
                previous_hit = str(self.bxd.bound_hit)
            # Now we have gone through the first inversion section we can set first_run to false
            first_run = False

            # Now get the md object to propagate the dynamics according to the standard Velocity Verlet / Langevin
            # procedure:
            # 1. md_step_pos: Get the half step velocity v(t + 1/2 * delta_t) and then new positions x(t + delta_t)
            # 2. Get forces at new positions
            # 3. md_step_vel : Get the  new velocities v(t + delta_t)
            self.md_integrator.md_step_pos(self.forces, self.mol)
            next_pos = self.bxd.get_s(self.md_integrator.current_positions)
            next_vel = self.bxd.get_s(self.md_integrator.current_positions+self.md_integrator.current_velocities)
            next_v_vel = self.bxd.get_s(self.md_integrator.current_positions+self.md_integrator.verlet_velocity)
            next_v_pos = self.bxd.get_s(self.md_integrator.verlet_positions)
            next_accel = self.bxd.get_s(self.md_integrator.current_positions+self.md_integrator.accel)
            #Check whether we are stuck at a boundary
            new_hit = self.bxd.box_list[self.bxd.box].lower.hit(self.bxd.get_s(self.mol), 'down') or self.bxd.box_list[self.bxd.box].upper.hit(self.bxd.get_s(self.mol), 'up')

            if bounded and not new_hit:
                hit_file.write('new hit at step ' + str(iterations) + '\n')
                hit_file.write('bound ' + str(previous_hit) + '\n')
                hit_file.write(
                    str(prior_pos[0]) + '\t' + str(prior_pos[1]) + '\t' + str(prior_vel[0]) + '\t' + str(
                        prior_vel[1]) + '\t' + str(prior_vel[1]) + '\n')
                hit_file.write(
                    str(prior_v_pos[0]) + '\t' + str(prior_v_pos[1]) + '\t' + str(prior_v_vel[0]) + '\t' + str(
                        prior_v_vel[1]) + '\t' + str(prior_v_vel[1]) + '\n')
                hit_file.write(str(prior_accel[0]) + '\t' + str(prior_accel[1]) + '\n')
                hit_file.write(
                    str(post_pos[0]) + '\t' + str(post_pos[1]) + '\t' + str(post_vel[0]) + '\t' + str(
                        post_vel[1]) + '\t' + str(post_vel[1]) + '\n')
                hit_file.write(
                    str(next_pos[0]) + '\t' + str(next_pos[1]) + '\t' + str(next_vel[0]) + '\t' + str(
                        next_vel[1]) + '\t' + str(next_vel[1]) + '\n')
                hit_file.write(
                    str(next_v_pos[0]) + '\t' + str(next_v_pos[1]) + '\t' + str(next_v_vel[0]) + '\t' + str(
                        next_v_vel[1]) + '\t' + str(next_v_vel[1]) + '\n')
                hit_file.write(str(next_accel[0]) + '\t' + str(next_accel[1]) + '\n')
                if str(previous_hit) == 'lower':
                    hit_file.write(str(self.bxd.box_list[self.bxd.box].lower.d) + '\t' + str(
                        self.bxd.box_list[self.bxd.box].lower.n[0]) + '\t' + str(
                        self.bxd.box_list[self.bxd.box].lower.n[1]) + '\n')
                elif str(previous_hit) == 'upper':
                    hit_file.write(str(self.bxd.box_list[self.bxd.box].upper.d) + '\t' + str(
                        self.bxd.box_list[self.bxd.box].upper.n[0]) + '\t' + str(
                        self.bxd.box_list[self.bxd.box].upper.n[1]) + '\n')
                hit_file.write(str(hit_post_contraint) + '\n')
                hit_file.flush()

            while bounded and new_hit:
                if self.bxd.box_list[self.bxd.box].lower.hit(self.bxd.get_s(self.mol), 'down'):
                    new_bound = 'lower'
                else:
                    new_bound='upper'
                if previous_hit == new_bound:
                    log_file.write("oops problem with inversion at multiple boundaries" +str('\n'))
                    self.mol.set_positions(self.md_integrator.old_positions)
                    new_hit = self.bxd.box_list[self.bxd.box].lower.hit(self.bxd.get_s(self.mol), 'down') or self.bxd.box_list[self.bxd.box].upper.hit(self.bxd.get_s(self.mol), 'up')
                    self.md_integrator.retry_pos(self.mol)
                    new_pos = self.bxd.get_s(self.md_integrator.current_positions)
                    new_vel = self.bxd.get_s(self.md_integrator.half_step_velocity)
                    double_hit_file.write('new hit at step ' + str(iterations) + '\n')
                    double_hit_file.write('previous bound ' + str(previous_hit) + '\n')
                    double_hit_file.write('new bound ' + str(new_bound) + '\n')
                    double_hit_file.write(
                        str(prior_pos[0]) + '\t' + str(prior_pos[1]) + '\t' + str(prior_vel[0]) + '\t' + str(
                            prior_vel[1]) + '\t' + str(prior_vel[1]) + '\n')
                    double_hit_file.write(
                        str(prior_v_pos[0]) + '\t' + str(prior_v_pos[1]) + '\t' + str(prior_v_vel[0]) + '\t' + str(
                            prior_v_vel[1]) + '\t' + str(prior_v_vel[1]) + '\n')
                    double_hit_file.write(str(prior_accel[0]) + '\t' + str(prior_accel[1]) + '\n')
                    double_hit_file.write(
                        str(post_pos[0]) + '\t' + str(post_pos[1]) + '\t' + str(post_vel[0]) + '\t' + str(
                            post_vel[1]) + '\t' + str(post_vel[1]) + '\n')
                    double_hit_file.write(
                        str(next_pos[0]) + '\t' + str(next_pos[1]) + '\t' + str(next_vel[0]) + '\t' + str(
                            next_vel[1]) + '\t' + str(next_vel[1]) + '\n')
                    double_hit_file.write(
                        str(next_v_pos[0]) + '\t' + str(next_v_pos[1]) + '\t' + str(next_v_vel[0]) + '\t' + str(
                            next_v_vel[1]) + '\t' + str(next_v_vel[1]) + '\n')
                    double_hit_file.write(str(next_accel[0]) + '\t' + str(next_accel[1]) + '\n')
                    double_hit_file.write(
                        str(new_pos[0]) + '\t' + str(new_pos[1]) + '\t' + str(new_vel[0]) + '\t' + str(
                            new_vel[1]) + '\n')
                    if new_bound == 'lower':
                        double_hit_file.write(str(self.bxd.box_list[self.bxd.box].lower.d) + '\t' + str(
                            self.bxd.box_list[self.bxd.box].lower.n[0]) + '\t' + str(
                            self.bxd.box_list[self.bxd.box].lower.n[1]) + '\n')
                    else:
                        double_hit_file.write(str(self.bxd.box_list[self.bxd.box].upper.d) + '\t' + str(
                            self.bxd.box_list[self.bxd.box].upper.n[0]) + '\t' + str(
                            self.bxd.box_list[self.bxd.box].upper.n[1]) + '\n')
                    double_hit_file.flush()
                else:
                    new_hit = False

            try:
                self.forces = self.mol.get_forces()
            except:
                print('forces error')
            self.md_integrator.md_step_vel(self.forces, self.mol)

            if not decorrelated and steps_since_last_hit > self.bxd.box_list[self.bxd.box].decorrelation_time:
                decorrelated = True

            # Now determine what to print at the current MD step. TODO improve data writing / reporting mechanism
            # First check if we are due to print the geometry
            if iterations % self.geo_print_frequency == 0 and decorrelated:
                if print_to_file:
                    io.write(geom_file,self.mol, format='xyz', append=True)
                    geom_file.flush()


            # Now see if we are due to print the BXD current value of the collective variable to file.
            if iterations % self.data_print_frequency == 0 and self.bxd.steps_since_any_boundary_hit > decorrelated:
                if print_to_file:
                    string = str(self.bxd.s)
                    # remove all newline and tab characters
                    string = string.replace('\n', '')
                    string = string.replace('\t', '')
                    string = ' '.join(string.split())
                    data_file.write(string+'\n')
                    data_file.flush()

            # Check if we are due to print to stdout
            if iterations % self.log_print_frequency == 0:
                if bounded == False:
                    log_file.write(self.md_integrator.output(self.mol) + self.bxd.output() + '\n')
                else:
                    log_file.write('HIT\t' + self.md_integrator.output(self.mol) + self.bxd.output() + '\n')
                log_file.flush()
                print(self.md_integrator.output(self.mol) + self.bxd.output())

            # TODO this is confusing and need re-writing.
            # This section writes the boundary data to file but in a convoluted way
            # It is all tied in with the live updating of the plot which is still quite experimental and need a review
            if iterations % self.plot_update_frequency == 0:
                self.bounds[0] = self.bxd.box_list[self.bxd.box].lower.get_data()
                self.bounds[1] = self.bxd.box_list[self.bxd.box].upper.get_data()
                if print_to_file:
                    bound_file.seek(0)
                    bound_file.truncate()
                    for b in self.bxd.box_list:
                        string = str(b.upper.get_data())
                        string = string.replace('\n', '')
                        string = string.replace('\t', '')
                        string = ' '.join(string.split())
                        bound_file.write(string + '\n')
                        bound_file.flush()
                if self.plot:
                    self.bxd_plotter.plot_bxd_from_array(self.points, self.bounds)

            # Check whether BXD has gathered all the info it needs, if so signal that the trajectory should stop
            if self.bxd.completed_runs == self.max_runs or iterations > max_steps:
                keep_going = False
                try:
                    self.bxd.final_printing(temp_dir,self.mol)
                except:
                    print("couldnt do final BXD printing")
            iterations += 1
            steps_since_last_hit +=1

    def get_correlation_time(self, max_time):
        """
        Runs a bxd trajectory until either the attached BXDconstraint indicates sufficient information has been obtained
        or the max_steps parameter is exceeded
        :param max_steps: DEFAULT np.inf
                          Maximum number of steps in MD trajectory
        :param parallel: Boolean DEFAULT False
                         If True then multiple copies of the trajectory are being run by multiprocess and this affects
                         how the calculator is attached
        :param print_to_file: Boolean DEFAULT False
                              Determines whether to print trajectory data to a file
        :param print_directory: String DEFAULT 'BXD_data'
                                Prefix for output directory for printing data
        :return:
        """
        first_run = True
        vac_array = []
        vac_array.append(self.md_integrator.current_velocities)
        for i in range(0,max_time):
            s = self.bxd.get_s(self.mol)
            bounded = self.bxd.box_list[self.bxd.box].upper.hit(s, 'up') or self.bxd.box_list[self.bxd.box].lower.hit(s, 'down')
            del_phi = []
            if bounded and first_run:
                print(
                    "BXD bound hit on first step, either there is a small rounding error or there is something wrong with the initial geometry or bound. Proceed with caution")
                bounded = False
            if bounded:
                # If we have hit a bound we need to determine whether we have hit the upper / lower boundary of the box
                # the path boundary or whether both are hit. The del_phi list is populated with each constraint.
                if self.bxd.bound_hit != 'none':
                    del_phi.append(self.bxd.del_constraint(self.mol))
                # If we have hit a bound get the md object to modify the velocities / positions appropriately.
                self.md_integrator.constrain(del_phi)

            self.md_integrator.md_step_pos(self.forces, self.mol)
            first_run = False
            try:
                self.forces = self.mol.get_forces()
            except:
                print('forces error')
            self.md_integrator.md_step_vel(self.forces, self.mol)
            i+=1
            vac_array.append(self.md_integrator.current_velocities)

        vac_array = np.asarray(vac_array)
        vaf2 = np.zeros((max_time) * 2 + 1)
        for l in range(len(self.mol.get_atomic_numbers())):
            for m in range(3):
                vaf2 += np.correlate(vac_array[:,l,m],vac_array[:,l,m],'full')
        vaf = vaf2[max_time:]
        vaf /= copy.deepcopy(vaf[0])

        turning_point = np.inf
        vaf_reduced = []
        for i,v in enumerate(vaf):
            if i > turning_point:
                break
            if vaf[i] > vaf[i-1] and i != 0 and turning_point > max_time:
                turning_point = i*5
            vaf_reduced.append(v)

        x = np.arange(1,len(vaf_reduced)+1)
        x = np.log(x)
        vaf_reduced = np.asarray(vaf_reduced)
        fit = np.polyfit(x, vaf_reduced, 1)
        decorrelation_time = (-np.log(2)/fit[0])*5
        return vaf, decorrelation_time



    def converging_trajectory_pool(self, root = 'Converging_data', processes=1 ):
        """
        Uses multiprocessing to run multiple copies of a converging trajectory and gather the data at the end.
        THIS CURRENTLY DOESNT WORK AS DESIRED
        :param root:
        :param processes:
        :return:
        """
        pool = PE(max_workers=processes)
        self_list = []
        for i in range(0, processes):
            t = self.__copy__()
            t.bxd.reset(str(root) + str(i))
            t.md_integrator.current_velocities = t.mol.get_velocities()
            t.md_integrator.current_positions = t.mol.get_positions()
            t.md_integrator.half_step_velocity = t.mol.get_velocities()
            self_list.append(t)
        for _ in pool.map(run_pool, self_list):
            pass
        self_list[0].bxd.collate_free_energy_data(prefix=root, outfile=root)

        def converging_trajectory_pool_split_boxes(self, box_geometries='BXD/box_geoms.xyz', processes=1):
            pool = PE(max_workers=processes)
            self_list = []
            start = self.bxd.start_box
            end = self.bxd.end_box
            increment = (end - start) / processes
            for i in range(0, processes):
                t = self.__copy__()
                t.bxd.reset('converging_'+str(i))
                t.bxd.start_box = int(start + i * increment)
                t.bxd.end_box = int(start + (i + 1) * increment)
                t.bxd.box = int(start + i * increment)
                temp_mol = io.read(box_geometries, index=int(start + i * increment))
                t.mol.set_positions(temp_mol.get_positions())
                vd.MaxwellBoltzmannDistribution(t.mol, t.md_integrator.temperature, force_temp=True)
                t.md_integrator.current_velocities = t.mol.get_velocities()
                t.md_integrator.current_positions = t.mol.get_positions()
                t.md_integrator.half_step_velocity = t.mol.get_velocities()
                self_list.append(t)
            self_list[-1].bxd.end_box = int(end)
            for _ in pool.map(run_pool, self_list):
                pass
            self_list[0].bxd.collate_free_energy_data(prefix = 'Converging_Data', outfile = 'Combined_converging')

    def write_traj_to_file(self, file_name = 'geom.xyz'):
        io.write(file_name,self.ase_traj)

def run_pool(traj):
    traj.run_trajectory()
    return 1

def run_pool_test(int):
    print(str(int))



