import numpy as np
from ase.md import velocitydistribution as vd
import ase.io as io
import os
from concurrent.futures import ProcessPoolExecutor as PE
import copy
from ChemDyME.OpenMMCalc import OpenMMCalculator

class Trajectory:

    def __init__(self, mol, bxd, md_integrator, bimolecular=False, process_number=0, geo_print_frequency=1000,
                 data_print_freqency=100, plot_update_frequency=100, mixed_timestep=False, initial_temperature=np.nan,
                 no_text_output=False, plot_output=False, plotter=None, calc = 'openMM', calcMethod = 'sys.xml', parallel = False):

        self.bxd = bxd
        self.calc = calc
        self.calcMethod = calcMethod
        self.md_integrator = md_integrator
        self.process_number = process_number
        self.bimolecular = bimolecular
        self.geo_print_frequency = geo_print_frequency
        self.data_print_freqency = data_print_freqency
        self.mixed_timestep = mixed_timestep
        self.forces = np.zeros(mol.get_positions().shape)
        self.mol = mol.copy()
        self.mol._calc = mol.get_calculator()
        if np.isnan(initial_temperature):
            initial_temperature = md_integrator.temperature
        vd.MaxwellBoltzmannDistribution(self.mol, initial_temperature, force_temp=True)
        self.md_integrator.current_velocities = self.mol.get_velocities()
        self.md_integrator.half_step_velocity = self.mol.get_velocities()
        self.ReactionCountDown = 0
        self.ase_traj = []
        self.points = []
        self.bounds = [None] * 2
        self.no_text_output = no_text_output
        self.plot = False
        if plot_output:
            self.plot = True
            self.bxd_plotter = plotter
        self.plot_update_frequency = plot_update_frequency

    def __copy__(self):
        mol = self.mol.copy()
        return Trajectory(mol, copy.deepcopy(self.bxd), copy.deepcopy(self.md_integrator), calcMethod = self.calcMethod)


    def run_trajectory(self, max_steps = np.inf, parallel = False, reset = False, print_to_file = False, print_directory = 'BXD_data'):

        if parallel:
            self.mol.set_calculator(OpenMMCalculator(self.calcMethod, self.mol, parallel=True))
        else:
            self.mol.set_calculator(OpenMMCalculator(self.calcMethod, self.mol))

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

        if reset:
            self.ase_traj = []
            self.points = []

        self.bxd.initialise_files()

        keep_going = True

        # Get forces from atoms
        try:
            self.forces = self.mol.get_forces()
        except:
            print('forces error')

        iterations = 0
        # Run MD trajectory for specified number of steps
        while keep_going:

            del_phi = []
            self.bxd.update(self.mol)
            if self.bxd.skip_box:
                if self.bxd.reverse:
                    self.mol.set_positions(self.bxd.box_list[self.bxd.box - 1].geometry.get_positions())
                else:
                    self.mol.set_positions(self.bxd.box_list[self.bxd.box + 1].geometry.get_positions())
                vd.MaxwellBoltzmannDistribution(self.mol, self.md_integrator.temperature, force_temp=True)
                self.md_integrator.current_velocities = self.mol.get_velocities()
                self.md_integrator.half_step_velocity = self.mol.get_velocities()
            bounded = self.bxd.inversion
            if bounded:
                if self.bxd.path_bound_hit:
                    del_phi.append(self.bxd.path_del_constraint(self.mol))
                if self.bxd.bound_hit != 'none':
                    del_phi.append(self.bxd.del_constraint(self.mol))
                # Perform inversion if required
                self.md_integrator.constrain(del_phi)
            self.md_integrator.md_step_pos(self.forces, self.mol)
            try:
                self.forces = self.mol.get_forces()
            except:
                print('forces error')
            self.md_integrator.md_step_vel(self.forces, self.mol)

            if iterations % self.geo_print_frequency == 0:
                self.ase_traj.append(self.mol.copy())
                if print_to_file:
                    io.write(geom_file,self.mol, format='xyz', append=True)
                    geom_file.flush()

            if iterations % 10 == 0:
                self.points.append(self.bxd.s)
                if print_to_file:
                    string = str(self.bxd.s)
                    string = string.replace('\n', '')
                    string = string.replace('\t', '')
                    string = ' '.join(string.split())
                    data_file.write(string+'\n')
                    data_file.flush()

            if iterations % self.data_print_freqency == 0:
                if not self.no_text_output:
                    print(self.md_integrator.output(self.mol) + self.bxd.output())

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

            #check if one full run is complete, if so stop the adaptive search
            if self.bxd.complete_runs == 1 or iterations > max_steps:
                keep_going = False
                try:
                    self.bxd.final_printing(temp_dir,self.mol)
                except:
                    print("couldnt do final BXD priniting")
            iterations += 1

    def converging_trajectory_pool(self, box_geometries = 'BXD/box_geoms.xyz', processes=1 ):
        pool = PE(max_workers=processes)
        self_list = []
        start = self.bxd.start_box
        end = self.bxd.end_box
        increment = ( end - start) / processes
        for i in range(0,processes):
            t = self.__copy__()
            t.bxd.start_box = int(start + i * increment)
            t.bxd.end_box = int(start + (i+1) * increment)
            t.bxd.box = int(start + i * increment)
            temp_mol = io.read(box_geometries, index = int(start + i * increment))
            t.mol.set_positions(temp_mol.get_positions())
            vd.MaxwellBoltzmannDistribution(t.mol, t.md_integrator.temperature, force_temp=True)
            t.md_integrator.current_velocities = t.mol.get_velocities()
            t.md_integrator.current_positions = t.mol.get_positions()
            t.md_integrator.half_step_velocity = t.mol.get_velocities()
            self_list.append(t)
        self_list[-1].bxd.end_box = int(end)
        for _ in pool.map(run_pool,self_list):
            pass

    def write_traj_to_file(self, file_name = 'geom.xyz'):
        io.write(file_name,self.ase_traj)

def run_pool(traj):
    traj.run_trajectory()
    return 1

def run_pool_test(int):
    print(str(int))



