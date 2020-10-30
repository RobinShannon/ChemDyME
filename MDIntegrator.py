from abc import abstractmethod
import numpy as np
from numpy.random import standard_normal
from ase import units
import warnings

class MDIntegrator:
    """
    Base class defining the integrator used to propagate the molecular dynamics. Currently there are two derived
    classes VelocityVerlet or Langevin. Each derived class must implement the following methods:
    constrain1: Perform a BXD velocity inversion at a single boundary
    constrain2: Perform a BXD velocity inversion when two boundaries are hit simultaneously
    md_step_pos: Update the atom positions.
    md_step_vel: Update the atom velocities.
    """

    def __init__(self, mol, temperature=298, timestep=0.5):
        self.temperature = temperature * units.kB
        self.timestep = timestep * units.fs
        self.masses = mol.get_masses()
        self.forces = 0
        self.old_forces = 0
        self.current_velocities = mol.get_velocities()
        self.half_step_velocity = mol.get_velocities()
        self.old_velocities = mol.get_velocities()
        self.positions = mol.get_positions()
        self.old_positions = mol.get_positions()
        self.current_positions = mol.get_positions()
        self.new_positions = mol.get_positions()
        self.constrained = False

    def constrain(self,del_phi):
        """
        Determines whether one or two BXD boundaries have been hit depending upon the length of the del_phi argument and
        calls the appropriate constrain function.
        :param del_phi: List containing one or more values for del_phi, the derivative of a given BXD constraint
        :return:
        """
        if len(del_phi) > 2:
            warnings.warn('three or more BXD constraints have been hit simultaneously, currently only a maximum of two '
                          'can be dealt with')
        elif len(del_phi) == 2:
            self.constrain2(del_phi[0], del_phi[1])
        elif len(del_phi) == 1:
            self.constrain1(del_phi[0])

    @abstractmethod
    def constrain1(self, del_phi):
        pass

    @abstractmethod
    def constrain2(self, del_phi1, del_phi2):
        pass

    @abstractmethod
    def md_step_pos(self, forces, mol):
        pass

    @abstractmethod
    def md_step_vel(self, forces, mol):
        pass


class VelocityVerlet(MDIntegrator):
    """
    Sets up an MD integrator for performing Velocity Verlet dynamics
    :param mol: ASE atoms object
    :param temperature: Starting temperature in Kelvin for the MD used to initialise random velocities
    :param timestep: Timestep for use in the MD integration in femtoseconds
    """

    def constrain1(self, del_phi):
        """
        Modify the stored velocities to satisfy a single BXD constraint
        :param del_phi: derivative of the constraint with respect to the cartesian coordinates
        :return:
        """

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
        #self.forces = del_phi * lagrangian
        self.constrained = True

    def constrain2(self, del_phi1, del_phi2):
        """
        Modify the stored velocities to satisfy two BXD constraints
        :param del_phi1: derivative of the first constraint with respect to the cartesian coordinates
        :param del_phi2: derivative of the seccond constraint with respect to the cartesian coordinates
        :return:
        """

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
        c5 = np.vdot(b, d)
        c6 = np.vdot(c, d)

        lamb2 = (-2*c5*c3 + 2*c1**c6)/(c2*c3-c1*c4)
        lamb1 = (-2*c5-c2*lamb2)/c1

        # Update velocities
        self.current_velocities = self.current_velocities + (lamb1 * del_phi1 * (1 / self.masses[:, None])) + \
                                  (lamb2 * del_phi2 * (1 / self.masses[:, None]))
        self.constrained = True

    def md_step_pos(self, forces, mol):
        """
        This method returns the new positions after a single md timestep
        :param forces: array containing the forces at the current geometry
        :param mol: ASE atoms object
        :return:
        """

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
        self.half_step_velocity = self.current_velocities + accel * self.timestep * 0.5
        self.current_positions = self.current_positions + (self.half_step_velocity * self.timestep)

        # Return positions
        mol.set_positions(self.current_positions)

        # If we have periodic boundary conditions then make sure the new positions respect these
        if mol.pbc.any():
            mol.wrap()

    def md_step_vel(self, forces, mol):
        """
        This method returns the new velocities
        :param forces: array containing the forces at the current geometry
        :param mol: ASE atoms object
        :return:
        """
        # Store forces from previous step and then update
        self.old_forces = self.forces
        self.forces = forces

        # Store old velocities
        self.old_velocities = self.current_velocities

        # Get Acceleration from masses and forces
        accel = self.forces[:] / self.masses[:, None]

        # Use recent half velocity to update the velocities
        self.current_velocities = self.half_step_velocity + (accel * self.timestep * 0.5)

        # Return positions
        mol.set_velocities(self.current_velocities)

    def output(self, mol):
        """
        Generates string of output text appropriate to the velocity verlet case
        :param mol: ASE atoms object
        :return: output string
        """
        out = " total energy = " + str(mol.get_potential_energy()) +'\t' +str(mol.get_kinetic_energy())
        return out

class Langevin(MDIntegrator):

    def __init__(self, mol, temperature=298, friction=1.0, timestep=0.5):
        """
        Set up an instance of a Langevin MD integrator.
        :param mol: ASE atoms object
        :param temperature: Temperature the MD should be run at in Kelvin
        :param friction: Langevin friction parameter
        :param timestep: MD timestep in femtosecconds
        """
        self.friction = friction
        # Get coefficients
        super(Langevin, self).__init__(mol, temperature=temperature, timestep=timestep)
        self.sigma = np.sqrt(2 * self.temperature * self.friction / self.masses)
        self.c1 = self.timestep / 2. - self.timestep * self.timestep * self.friction / 8.
        self.c2 = self.timestep * self.friction / 2 - self.timestep * self.timestep * self.friction * self.friction / 8.
        self.c3 = np.sqrt(self.timestep) * self.sigma / 2. - self.timestep**1.5 * self.friction * self.sigma / 8.
        self.c5 = self.timestep**1.5 * self.sigma / (2 * np.sqrt(3))
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
    def constrain1(self, del_phi):
        """
        Modify the stored velocities to satisfy a single BXD constraint
        :param del_phi: derivative of the constraint with respect to the cartesian coordinates
        :return:
        """
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

        self.discarded_velocities = self.current_velocities

        # Update velocities
        self.current_velocities = (self.current_velocities + (lagrangian * del_phi * (1/self.masses)[:, None]))
        self.half_step_velocity = self.current_velocities
        self.constrained = True

    def constrain2(self, del_phi1, del_phi2):
        """
        Modify the stored velocities to satisfy two BXD constraints
        :param del_phi1: derivative of the first constraint with respect to the cartesian coordinates
        :param del_phi2: derivative of the seccond constraint with respect to the cartesian coordinates
        :return:
        """
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
        c5 = np.vdot(b, d)
        c6 = np.vdot(c, d)

        lamb2 = (-2*c5*c3 + 2*c1**c6)/(c2*c3-c1*c4)
        lamb1 = (-2*c5-c2*lamb2)/c1

        # Update velocities
        self.current_velocities = self.current_velocities + (lamb1 * del_phi1 * (1 / self.masses[:, None])) + (lamb2 * del_phi2 * (1 / self.masses[:, None]))
        self.half_step_velocity = self.current_velocities
        # Let MDstep method know a constraint has occured and not to update the forces
        self.constrained = True

    # This method returns the new positions after a single md timestep
    def md_step_pos(self, forces, mol):
        """
        This method returns the new positions after a single md timestep
        :param forces: array containing the forces at the current geometry
        :param mol: ASE atoms object
        :return:
        """

        # If we have been constrained then forces have already been reset
        # Otherwise store forces provided to function
        if self.constrained:
            self.constrained = False
        else:
            self.forces = forces

        # keep track of position prior to update in case we need to revert
        self.very_old_positions = self.old_positions
        self.old_positions = self.current_positions
        self.old_forces = self.forces

        # Get Acceleration from masses and forces
        accel = self.forces[:] / self.masses[:, None]

        # Get two normally distributed variables
        self.xi = standard_normal(size=(len(self.masses), 3))
        self.eta = standard_normal(size=(len(self.masses), 3))

        # Then get the next half step velocity and update the position.
        # NB currentVel is one full MD step behind currentPos

        self.half_step_velocity = self.current_velocities + (self.c1 * accel - self.c2 * self.half_step_velocity + self.c3[:, None] * self.xi - self.c4[:, None] * self.eta)
        self.current_positions = self.current_positions + self.timestep * self.half_step_velocity + self.c5[:, None] * self.eta

        # Return positions
        mol.set_positions(self.current_positions)

    def md_step_vel(self, forces, mol):
        """
        This method returns the new velocities
        :param forces: array containing the forces at the current geometry
        :param mol: ASE atoms object
        :return:
        """

        # Store forces from previous step and then update

        self.forces = forces

        # Store old velocities
        self.old_velocities = self.current_velocities

        # Get Acceleration from masses and forces
        accel = self.forces[:] / self.masses[:, None]

        # Use recent half velocity to update the velocities
        self.current_velocities = self.half_step_velocity + (
                    self.c1 * accel - self.c2 * self.half_step_velocity + self.c3[:, None] * self.xi - self.c4[:, None] * self.eta)

        # Return positions
        mol.set_velocities(self.current_velocities)

    def retry_pos(self, mol):
            """
            This method returns the new positions after a single md timestep
            :param forces: array containing the forces at the current geometry
            :param mol: ASE atoms object
            :return:
            """

            self.current_positions = mol.get_positions()


            # Get Acceleration from masses and forces
            accel = self.forces[:] / self.masses[:, None]


            # Get two normally distributed variables
            self.xi = standard_normal(size=(len(self.masses), 3))
            self.eta = standard_normal(size=(len(self.masses), 3))

            changes = self.current_velocities - self.discarded_velocities

            # Then get the next half step velocity and update the position.
            # NB currentVel is one full MD step behind currentPos
            self.half_step_velocity = self.current_velocities + (self.c1 * accel - self.c2 * self.half_step_velocity + self.c3[:, None] * self.xi - self.c4[:, None] * self.eta)
            self.current_positions = self.current_positions + 0.01 * (self.timestep * self.half_step_velocity + self.c5[:,
                        None] * self.eta)

            mol.set_positions(self.current_positions)
            return

    def output(self, mol):
        """
        Generates string of output text appropriate to the Langevin case
        :param mol: ASE atoms object
        :return: output string
        """
        out = " Temperature = " + str(mol.get_temperature()) + ' Total_energy = ' +str(mol.get_potential_energy() + mol.get_kinetic_energy())
        return out