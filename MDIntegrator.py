from abc import abstractmethod
import numpy as np
from numpy.random import standard_normal


# Class to track constraints and calculate required derivatives
# Inversion procedure occurs in MDintegrator class
class MDIntegrator:

    def __init__(self, forces, velocities, mol):
        self.mol = mol
        self.masses = mol.get_masses()
        self.forces = forces
        self.old_forces = forces
        self.current_velocities = velocities
        self.half_step_velocity = velocities
        self.old_velocities = velocities
        self.positions = mol.get_positions()
        self.old_positions = mol.get_positions()
        self.current_positions = mol.get_positions()
        self.new_positions = mol.get_positions()
        self.constrained = True

    @abstractmethod
    def constrain(self, del_phi):
        pass

    @abstractmethod
    def constrain2(self, del_phi1, del_phi2):
        pass

    @abstractmethod
    def md_step_pos(self, forces, timestep, mol):
        pass

    @abstractmethod
    def md_step_vel(self, forces, timestep, mol):
        pass


class VelocityVerlet(MDIntegrator):

    # Modify the stored velocities to satisfy a single BXD constraint
    def constrain(self, del_phi):

        # Revert positions and forces to time prior to BXD inversion
        self.current_positions = self.old_positions
        self.current_velocities = self.old_velocities
        self.forces = self.old_forces

        # Temporarily flatten 3 by n matrices into vectors to get dot products.
        a = ((np.tile(self.masses, (3, 1))).transpose()).flatten()
        b = del_phi.flatten()
        c = self.current_velocities.flatten()

        # Get Lagrangian constraint
        lagrangian = (-2.0 * np.vdot(b, c)) / np.vdot(b, (b * (1 / a)))

        # Modify velocities
        self.current_velocities += (lagrangian * del_phi * (1/self.masses)[:, None])
        self.constrained = True

    def constrain2(self, del_phi1, del_phi2):

        # Revert positions and forces to time prior to BXD inversion
        self.current_positions = self.old_positions
        self.current_velocities = self.old_velocities
        self.forces = self.old_forces

        # Temporarily flatten 3 by n matrices into vectors to get dot products.
        a = ((np.tile(self.masses, (3, 1))).transpose()).flatten()
        b = del_phi1.flatten()
        c = del_phi2.flatten()
        d = self.current_velocities.flatten()

        # In the two constraint case the simulataneous equations can be solved analytically.
        # This should be gneralised in the future. Here we save the various coefficients from the various permuations of
        # dot products containing the two constraints
        c1 = np.vdot(b, (b * (1 / a)))
        c2 = np.vdot(b, (c * (1 / a)))
        c3 = np.vdot(c, (b * (1 / a)))
        c4 = np.vdot(c, (c * (1 / a)))
        c6 = np.vdot(c, d)

        lamb2 = ((c3 * c4) - (c1 * c6))/((c2 * c3) - (c1 * c4))
        lamb1 = (c3 - lamb2 * c2) / c1

        # Update velocities
        self.current_velocities = self.current_velocities + (lamb1 * del_phi1 * (1 / self.masses[:, None])) + \
                                  (lamb2 * del_phi2 * (1 / self.masses[:, None]))
        self.constrained = True

    # This method returns the new positions after a single md timestep
    def md_step_pos(self, forces, timestep, mol):

        # If we have been constrained then forces have already been reset
        # Otherwise store forces provided to function
        if self.constrained:
            self.constrained = False
        else:
            self.forces = forces

        #  Get Acceleration from masses and forces
        accel = self.forces[:] / self.masses[:, None]

        # keep track of position prior to update in case we need to revert
        self.old_positions = self.current_positions

        # Then get the next half step velocity and update the position.
        # NB currentVel is one full MD step behind currentPos
        self.half_step_velocity = self.current_velocities + accel * timestep * 0.5
        self.current_positions = self.current_positions + (self.half_step_velocity * timestep)

        # Return positions
        mol.set_positions(self.current_positions)
        if mol.pbc.any():
            mol.wrap()

    def md_step_vel(self, forces, timestep, mol):

        # Store forces from previous step and then update
        self.old_forces = self.forces
        self.forces = forces

        # Store old velocities
        self.old_velocities = self.current_velocities

        # Get Acceleration from masses and forces
        accel = self.forces[:] / self.masses[:, None]

        # Use recent half velocity to update the velocities
        self.current_velocities = self.half_step_velocity + (accel * timestep * 0.5)

        # Return positions
        mol.set_velocities(self.current_velocities)


class Langevin(MDIntegrator):

    def __init__(self, temperature, friction, forces, velocity, mol, timestep):
        self.friction = friction
        self.temp = temperature
        # Get coefficients
        super(Langevin, self).__init__(forces, velocity, mol)
        self.sigma = np.sqrt(2 * self.temp * self.friction / self.masses)
        self.c1 = timestep / 2. - timestep * timestep * self.friction / 8.
        self.c2 = timestep * self.friction / 2 - timestep * timestep * self.friction * self.friction / 8.
        self.c3 = np.sqrt(timestep) * self.sigma / 2. - timestep**1.5 * self.friction * self.sigma / 8.
        self.c5 = timestep**1.5 * self.sigma / (2 * np.sqrt(3))
        self.c4 = self.friction / 2. * self.c5
        self.xi = 0
        self.eta = 0

    def reset(self, timestep):
        self.c1 = timestep / 2. - timestep * timestep * self.friction / 8.
        self.c2 = timestep * self.friction / 2 - timestep * timestep * self.friction * self.friction / 8.
        self.c3 = np.sqrt(timestep) * self.sigma / 2. - timestep**1.5 * self.friction * self.sigma / 8.
        self.c5 = timestep**1.5 * self.sigma / (2 * np.sqrt(3))
        self.c4 = self.friction / 2. * self.c5

    # Modify the stored velocities to satisfy a single BXD constraint
    def constrain(self, del_phi):

        # Revert positions and forces to time prior to BXD inversion
        self.current_positions = self.old_positions
        self.current_velocities = self.old_velocities
        self.forces = self.old_forces

        # Temporarily flatten 3 by n matrices into vectors to get dot products.
        a = ((np.tile(self.masses, (3, 1))).transpose()).flatten()
        b = del_phi.flatten()
        c = self.current_velocities.flatten()

        # Get Lagrangian constraint
        lagrangian = (-2.0 * np.vdot(b, c)) / np.vdot(b, (b * (1 / a)))

        # Update velocities
        self.current_velocities = self.current_velocities + (lagrangian * del_phi * (1/self.masses)[:, None])
        self.constrained = True

    def constrain2(self, del_phi1, del_phi2):

        # Revert positions and forces to time prior to BXD inversion
        self.current_positions = self.old_positions
        self.current_velocities = self.old_velocities
        self.forces = self.old_forces

        # Temporarily flatten 3 by n matrices into vectors to get dot products.
        a = ((np.tile(self.masses, (3, 1))).transpose()).flatten()
        b = del_phi1.flatten()
        c = del_phi2.flatten()
        d = self.current_velocities.flatten()

        # In the two constraint case the simulataneous equations can be solved analytically.
        # This should be gneralised in the future. Here we save the various coefficients from the various permuations
        # of dot products containing the two constraints
        c1 = np.vdot(b, (b * (1 / a)))
        c2 = np.vdot(b, (c * (1 / a)))
        c3 = np.vdot(c, (b * (1 / a)))
        c4 = np.vdot(c, (c * (1 / a)))
        c6 = np.vdot(c, d)

        lamb2 = ((c3 * c4) - (c1 * c6))/((c2 * c3) - (c1 * c4))
        lamb1 = (c3 - lamb2 * c2) / c1

        # Update velocities
        self.current_velocities = self.current_velocities + (lamb1 * del_phi1 * (1 / self.masses[:, None])) + \
                                  (lamb2 * del_phi2 * (1 / self.masses[:, None]))

        # Let MDstep method know a constraint has occured and not to update the forces
        self.constrained = True

    # This method returns the new positions after a single md timestep
    def md_step_pos(self, forces, timestep, mol):

        # If we have been constrained then forces have already been reset
        # Otherwise store forces provided to function
        if self.constrained:
            self.constrained = False
        else:
            self.forces = forces

        # Get Acceleration from masses and forces
        accel = self.forces[:] / self.masses[:, None]

        # Keep track of position prior to update in case we need to revert
        self.old_positions = self.current_positions

        # Get two normally distributed variables
        self.xi = standard_normal(size=(len(self.masses), 3))
        self.eta = standard_normal(size=(len(self.masses), 3))

        # Then get the next half step velocity and update the position.
        # NB currentVel is one full MD step behind currentPos
        self.half_step_velocity = self.current_velocities + \
                                  (self.c1 * accel - self.c2 * self.half_step_velocity + self.c3[:, None] * self.xi -
                                   self.c4[:, None] * self.eta)

        self.current_positions = self.current_positions + timestep * self.half_step_velocity + self.c5[:, None] * self.eta

        # Return positions
        mol.set_positions(self.current_positions)
        if mol.pbc.any():
            mol.wrap()

    def md_step_vel(self, forces, timestep, mol):

        # Store forces from previous step and then update
        self.old_forces = self.forces
        self.forces = forces

        # Store old velocities
        self.old_velocities = self.current_velocities

        # Get Acceleration from masses and forces
        accel = self.forces[:] / self.masses[:, None]

        # Use recent half velocity to update the velocities
        self.current_velocities = self.half_step_velocity + (self.c1 * accel - self.c2 * self.half_step_velocity +
                                                             self.c3[:, None] * self.xi - self.c4[:, None] * self.eta)

        # Return positions
        mol.set_velocities(self.current_velocities)

