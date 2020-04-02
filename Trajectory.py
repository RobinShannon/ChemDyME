import numpy as np
from ase.md import velocitydistribution as vd
import ase.io as io
import os
from concurrent.futures import ProcessPoolExecutor as PE
import copy
from ChemDyME.OpenMMCalc import OpenMMCalculator

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

    def __init__(self, mol, bxd, md_integrator, geo_print_frequency=1000, data_print_freqency=100,
                 plot_update_frequency=100, no_text_output=False, plot_output=False, plotter=None, calc = 'openMM',
                 calcMethod = 'sys.xml', initialise_velocities = True):

        self.bxd = bxd
        self.calc = calc
        self.calcMethod = calcMethod
        self.md_integrator = md_integrator
        self.geo_print_frequency = geo_print_frequency
        self.data_print_freqency = data_print_freqency
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


    def run_trajectory(self, max_steps = np.inf, parallel = False, print_to_file = False, print_directory = 'BXD_data'):
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
        if parallel:
            self.mol.set_calculator(OpenMMCalculator(self.calcMethod, self.mol, parallel=True))
        else:
            self.mol.set_calculator(OpenMMCalculator(self.calcMethod, self.mol))

        print(str(self.mol.get_potential_energy))
        # If print_to_file = True then setup an output directory. If the print_directory already exists then append
        # consecutive numbers to the "print_directory" prefix until a the name does not correspond to an exsisting
        # directory.
        if print_to_file:
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

        # depending upon the type of BXD object this function does some initial setup
        self.bxd.initialise_files()



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
        iterations = 0

        # Run MD trajectory for specified number of steps or until BXD reaches its end point
        while keep_going:
            del_phi = []
            # update the BXDconstraint with the current geometry
            self.bxd.update(self.mol)
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
                if self.bxd.path_bound_hit:
                    del_phi.append(self.bxd.path_del_constraint(self.mol))
                # If we have hit a bound get the md object to modify the velocities / positions appropriately.
                self.md_integrator.constrain(del_phi)
                hit = True
            # Now we have gone through the first inversion section we can set first_run to false
            first_run = False

            # Now get the md object to propagate the dynamics according to the standard Velocity Verlet / Langevin
            # procedure:
            # 1. md_step_pos: Get the half step velocity v(t + 1/2 * delta_t) and then new positions x(t + delta_t)
            # 2. Get forces at new positions
            # 3. md_step_vel : Get the  new velocities v(t + delta_t)
            self.md_integrator.md_step_pos(self.forces, self.mol)
            # new_hit = self.bxd.box_list[self.bxd.box].lower.hit(self.bxd.get_s(self.mol), 'down') or self.bxd.box_list[
            #     self.bxd.box].upper.hit(self.bxd.get_s(self.mol), 'up')
            # while hit and new_hit:
            #     self.md_integrator.retry_pos(self.mol)
            #     new_hit = self.bxd.box_list[self.bxd.box].lower.hit(self.bxd.get_s(self.mol), 'down') or \
            #               self.bxd.box_list[
            #                   self.bxd.box].upper.hit(self.bxd.get_s(self.mol), 'up')
            try:
                self.forces = self.mol.get_forces()
            except:
                print('forces error')
            self.md_integrator.md_step_vel(self.forces, self.mol)
            #if hit:
                # #print("hit!\tOld/New s\t=\t" +str(self.bxd.get_s(self.md_integrator.old_positions)) +'\t' +  str(self.bxd.get_s(self.md_integrator.current_positions)) + "Old/New vel\t=\t"+str(self.bxd.get_s(self.md_integrator.discarded_velocities)) +'\t' +str(self.bxd.get_s(self.md_integrator.old_velocities)) +'\t' +  str(self.bxd.get_s(self.md_integrator.current_velocities)))
                #
                # if self.bxd.bound_hit=='upper':
                #     if self.bxd.box_list[self.bxd.box].upper.hit(self.bxd.get_s(self.mol), 'up'):
                #         print("upper still hit after inversion!")
                #         print("hit!\tOld/New s\t=\t" +str(self.bxd.get_s(self.md_integrator.old_positions)) +'\t' +  str(self.bxd.get_s(self.md_integrator.current_positions)) + "Old/New vel\t=\t"+str(self.bxd.get_s(self.md_integrator.discarded_velocities)) +'\t' +str(self.bxd.get_s(self.md_integrator.old_velocities)) +'\t' +  str(self.bxd.get_s(self.md_integrator.current_velocities)))
                #         print(str(self.bxd.box_list[self.bxd.box].upper.n) + '\t' + str(
                #             self.bxd.box_list[self.bxd.box].upper.d))
                # elif self.bxd.bound_hit=='lower':
                #     if self.bxd.box_list[self.bxd.box].lower.hit(self.bxd.get_s(self.mol), 'down'):
                #         print("lower still hit after inversion!")
                #         print("hit!\tOld/New s\t=\t" +str(self.bxd.get_s(self.md_integrator.old_positions)) +'\t' +  str(self.bxd.get_s(self.md_integrator.current_positions)) + "Old/New vel\t=\t"+str(self.bxd.get_s(self.md_integrator.discarded_velocities)) +'\t' +str(self.bxd.get_s(self.md_integrator.old_velocities)) +'\t' +  str(self.bxd.get_s(self.md_integrator.current_velocities)))
                #         print(str(self.bxd.box_list[self.bxd.box].lower.n) + '\t' + str(self.bxd.box_list[self.bxd.box].lower.d))
            hit = False

            #new_hit =  self.bxd.box_list[self.bxd.box].lower.hit(self.bxd.get_s(self.mol), 'down') or self.bxd.box_list[self.bxd.box].upper.hit(self.bxd.get_s(self.mol), 'up')
            #old_hit = self.bxd.box_list[self.bxd.box].lower.hit(self.bxd.get_s(self.md_integrator.old_positions), 'down') or self.bxd.box_list[self.bxd.box].upper.hit(self.bxd.get_s(self.md_integrator.old_positions), 'up')
            #print(str(old_hit) + '\t' + str(new_hit))
            # Now determine what to print at the current MD step. TODO improve data writing / reporting mechanism
            # First check if we are due to print the geometry
            if iterations % self.geo_print_frequency == 0 and self.bxd.steps_since_any_boundary_hit > self.bxd.decorrelation_limit:
                if print_to_file:
                    io.write(geom_file,self.mol, format='xyz', append=True)
                    geom_file.flush()


            # Now see if we are due to print the BXD current value of the collective variable to file.
            if iterations % 10 == 0 and self.bxd.steps_since_any_boundary_hit > self.bxd.decorrelation_limit:
                if print_to_file:
                    string = str(self.bxd.s)
                    # remove all newline and tab characters
                    string = string.replace('\n', '')
                    string = string.replace('\t', '')
                    string = ' '.join(string.split())
                    data_file.write(string+'\n')
                    data_file.flush()

            # Check if we are due to print to stdout
            if iterations % self.data_print_freqency == 0:
                if not self.no_text_output:
                    px = 0
                    py = 0
                    pz = 0
                    for atom in self.mol:
                        px += atom.momentum[0]
                        py += atom.momentum[1]
                        pz += atom.momentum[2]
                    # Call md integrator and BXD to get the apropriate output string
                    if bounded == False:
                        print(self.md_integrator.output(self.mol) + self.bxd.output() + '\tMomenta\t' + str(px) + '\t' + str(py) +'\t' + str(pz))
                    else:
                        print('HIT\t' + self.md_integrator.output(self.mol) + self.bxd.output() + '\tMomenta\t' + str(
                            px) + '\t' + str(py) + '\t' + str(pz))

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
            if self.bxd.completed_runs == 1 or iterations > max_steps:
                keep_going = False
                try:
                    self.bxd.final_printing(temp_dir,self.mol)
                except:
                    print("couldnt do final BXD printing")
            iterations += 1

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



