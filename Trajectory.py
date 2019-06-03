import numpy as np
from ase.md import velocitydistribution as vd
from ase.visualize import view

class Trajectory:

    def __init__(self, mol, bxd, md_integrator, bimolecular=False, process_number=0, geo_print_frequency=100,
                  data_print_freqency=100, mixed_timestep=False, initial_temperature=np.nan):
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


    def run_trajectory(self, max_steps = np.inf, save_ase_traj = False, reset = False):

        if reset:
            self.ase_traj = []

        keep_going = True

        file = open("geo.xyz","w")

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
                self.mol.set_positions(self.md_integrator.old_positions)
                if self.bxd.bound_hit != 'none':
                    del_phi.append(self.bxd.del_constraint(self.mol))
                if self.bxd.path_bound_hit:
                    del_phi.append(self.bxd.path_del_constraint(self.mol))

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


            if iterations % self.data_print_freqency == 0:
                print(self.md_integrator.output(self.mol) + self.bxd.output())


            #check if one full run is complete, if so stop the adaptive search
            if self.bxd.complete_runs == 1 or iterations > max_steps:
                keep_going = False

            iterations += 1

