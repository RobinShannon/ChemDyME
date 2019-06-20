import numpy as np
from ase.md import velocitydistribution as vd
import ase.io as io
import os, datetime
from ase.visualize import view

class Trajectory:

    def __init__(self, mol, bxd, md_integrator, bimolecular=False, process_number=0, geo_print_frequency=1000,
                 data_print_freqency=100, plot_update_frequency=100, mixed_timestep=False, initial_temperature=np.nan,
                 no_text_output=False, plot_output=False, plotter=None):

        self.bxd = bxd
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


    def run_trajectory(self, max_steps = np.inf, save_ase_traj = False, reset = False, print_to_file = False, print_directory = 'BXD_data'):

        if print_to_file:
           dir = str(print_directory + str(datetime.datetime.now().isoformat(timespec='minutes')))
           os.mkdir(dir)
           data_file = open(dir+'/data.txt', 'w')
           geom_file = open(dir+'/geom.xyz', 'w')
           bound_file = open(dir+'/bound_file.txt', 'w')


        if reset:
            self.ase_traj = []
            self.points = []

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
                    io.write(geom_file,self.mol,format='xyz', append=True)
                    geom_file.flush()

            if iterations % 10 == 0:
                self.points.append(self.bxd.s)
                if print_to_file:
                    data_file.write(str(self.bxd.s)+'\n')
                    data_file.flush()

            if iterations % self.data_print_freqency == 0:
                if not self.no_text_output:
                    print(self.md_integrator.output(self.mol) + self.bxd.output())

            if iterations % self.plot_update_frequency == 0:
                self.bounds[0] = self.bxd.box_list[self.bxd.box].lower.get_data()
                self.bounds[1] = self.bxd.box_list[self.bxd.box].upper.get_data()
                if print_to_file:
                    bound_file.write(str(self.bxd.box_list[self.bxd.box].lower.get_data()) + '\n')
                    bound_file.flush()
                if self.plot:
                    self.bxd_plotter.plot_bxd_from_array(self.points, self.bounds)

            #check if one full run is complete, if so stop the adaptive search
            if self.bxd.complete_runs == 1 or iterations > max_steps:
                keep_going = False

            iterations += 1

    def write_traj_to_file(self, file_name = 'geom.xyz'):
        io.write(file_name,self.ase_traj)



