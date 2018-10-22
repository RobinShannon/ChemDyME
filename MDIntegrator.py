from abc import ABCMeta, abstractmethod
import numpy as np
from numpy.random import standard_normal
from ase import units


# Class to track constraints and calculate required derivatives
# Inversion procedure occurs in MDintegrator class
class MDIntegrator:

    def __init__(self, forces, velocity, mol):
        self.mol = mol
        self.masses = mol.get_masses()
        self.forces = forces
        self.oldForces = forces
        self.currentVel = velocity
        self.HalfVel = velocity
        self.oldVel = velocity
        self.pos = mol.get_positions()
        self.oldPos = mol.get_positions()
        self.currentPos = mol.get_positions()
        self.newPos = mol.get_positions()
        self.vHalfPresent = False
        self.constrained = False

    @abstractmethod
    def constrain(self, del_phi):
        pass

    @abstractmethod
    def constrain2(self, del_phi1, del_phi2):
        pass

    @abstractmethod
    def mdStep(self, forces, timestep, mol):
        pass


class VelocityVerlet(MDIntegrator):

    # Modify the stored velocities to satisfy a single BXD constraint
    @abstractmethod
    def constrain(self, del_phi):

        # Revert positions and forces to time prior to BXD inversion
        self.currentPos = self.oldPos

        # Have to start VV procedure again as we loose the half steps
        self.vHalfPresent = False

        # Temporarily flatten 3 by n matrices into vectors to get dot products.
        a = ((np.tile(self.masses, (3, 1))).transpose()).flatten()
        b = del_phi.flatten()
        c = self.currentVel.flatten()

        # Get Lagrangian constraint
        lamb =  (-2.0 * np.vdot(b,c)) / np.vdot(b , (b * (1 / a)))


        # Update velocities
        self.HalfVel = self.currentVel + ( lamb * del_phi * (1/self.masses)[:,None])
        self.constrained = True

    @abstractmethod
    def constrain2(self, del_phi1, del_phi2):

        # Revert postions and forces to time prior to BXD inversion
        self.currentPos = self.oldPos

        # Have to start VV procedure again as we loose the half steps
        self.vHalfPresent = False

        # Temporarily flatten 3 by n matrices into vectors to get dot products.
        a = ((np.tile(self.masses, (3, 1))).transpose()).flatten()
        b = np.flatten(del_phi1)
        c = np.flatten(del_phi2)
        d = np.flatten(self.oldVel)

        # In the two constraint case the simulataneous equations can be solved analytically. This should be gneralised in the future
        # Here we save the various coefficients from the various permuations of dot products containing the two constraints
        c1 = np.vdot(b,(b * (1 / a)))
        c2 = np.vdot(b,(c * (1 / a)))
        c3 = np.vdot(c,(b * (1 / a)))
        c4 = np.vdot(c,(c * (1 / a)))
        c5 = np.vdot(b,d)
        c6 = np.vdot(c,d)

        lamb2 = (( c3 * c4) - ( c1 * c6 ))/(( c2 * c3 ) - ( c1 * c4 ))
        lamb1 = (c3 - lamb2 * c2) / c1


        # Update velocities
        self.HalfVel = self.oldVel + ( lamb1 * del_phi1 * ( 1 / self.masses[:,None] ) ) + ( lamb2 * del_phi2 * ( 1 / self.masses[:,None] ) )
        self.oldVel = self.HalfVel

        # Let MDstep method know a constraint has occured and not to update the forces
        self.constrained = True


    #This method returns the new positions after a single md timestep
    @abstractmethod
    def mdStep(self, forces, timestep, mol):

        if self.constrained == False:
            self.forces = forces
        else:
            self.constrained = False

        #check the below works. Get Acceleration from masses and forces
        accel = self.forces[:] / self.masses[:,None]

        # keep track of position prior to update in case we need to revert
        self.oldPos = self.currentPos

        #Then start Velocity Verlet procedure to get the current velocity from previous halfstep
        if self.vHalfPresent is True:
            self.currentVel = self.HalfVel + ( accel * timestep * 0.5 )
        else:
            self.currentVel = self.HalfVel
            self.vHalfPresent = True

        # Then get the next half step velocity and update the position.
        # NB currentVel is one full MD step behind currentPos
        self.HalfVel = self.currentVel + accel * timestep * 0.5
        self.currentPos = self.currentPos + (self.HalfVel * timestep)

        # Return positions
        mol.set_positions(self.currentPos)
        mol.set_velocities(self.currentVel)

class Langevin(MDIntegrator):

    def __init__(self, temperature, friction, forces, velocity, mol):
        self.friction = friction
        self.temp = temperature
        super(Langevin,self).__init__(forces, velocity, mol)
        self.sigma = np.sqrt(2 * self.temp * self.friction / self.masses)

    @abstractmethod
    def mdStep(self, forces, timestep, mol):

        if self.constrained == False:
            self.forces = forces
        else:
            self.constrained = False

        # Get coefficients
        c1 = timestep / 2. - timestep * timestep * self.friction / 8.
        c2 = timestep * self.friction / 2 - timestep * timestep * self.friction * self.friction / 8.
        c3 = np.sqrt(timestep) * self.sigma / 2. - timestep**1.5 * self.friction * self.sigma / 8.
        c5 = timestep**1.5 * self.sigma / (2 * np.sqrt(3))
        c4 = self.friction / 2. * c5

        # Get two normally distributed variables
        xi = standard_normal(size=(len(self.masses), 3))
        eta = standard_normal(size=(len(self.masses), 3))

        # Check the below works. Get Acceleration from masses and forces
        accel = self.forces[:] / self.masses[:,None]

        # Keep track of position prior to update in case we need to revert
        self.oldPos = self.currentPos

        # Then start Velocity Verlet procedure to get the current velocity from previous halfstep
        if self.vHalfPresent is True:
            self.currentVel = self.HalfVel + (c1 * accel - c2 * self.HalfVel + c3[:,None] * xi - c4[:,None] * eta)
        else:
            self.currentVel = self.HalfVel
            self.vHalfPresent = True

        # Then get the next half step velocity and update the position.
        # NB currentVel is one full MD step behind currentPos
        self.HalfVel = self.currentVel + (c1 * accel - c2 * self.HalfVel + c3[:,None] * xi - c4[:,None] * eta)
        self.currentPos = self.currentPos + timestep * self.HalfVel + c5[:,None] * eta

        # Return positions
        mol.set_positions(self.currentPos)
        mol.set_velocities(self.currentVel)

        # Modify the stored velocities to satisfy a single BXD constraint
    @abstractmethod
    def constrain(self, del_phi):

        # Revert positions and forces to time prior to BXD inversion
        self.currentPos = self.oldPos

        # Have to start VV procedure again as we loose the half steps
        self.vHalfPresent = False

        # Temporarily flatten 3 by n matrices into vectors to get dot products.
        a = ((np.tile(self.masses, (3, 1))).transpose()).flatten()
        b = del_phi.flatten()
        c = self.currentVel.flatten()

        # Get Lagrangian constraint
        lamb =  (-2.0 * np.vdot(b,c)) / np.vdot(b , (b * (1 / a)))


        # Update velocities
        self.HalfVel = self.currentVel + ( lamb * del_phi * (1/self.masses)[:,None])
        self.constrained = True

    @abstractmethod
    def constrain2(self, del_phi1, del_phi2):

        # Revert postions and forces to time prior to BXD inversion
        self.currentPos = self.oldPos

        # Have to start VV procedure again as we loose the half steps
        self.vHalfPresent = False

        # Temporarily flatten 3 by n matrices into vectors to get dot products.
        a = ((np.tile(self.masses, (3, 1))).transpose()).flatten()
        b = np.flatten(del_phi1)
        c = np.flatten(del_phi2)
        d = np.flatten(self.oldVel)

        # In the two constraint case the simulataneous equations can be solved analytically. This should be gneralised in the future
        # Here we save the various coefficients from the various permuations of dot products containing the two constraints
        c1 = np.vdot(b,(b * (1 / a)))
        c2 = np.vdot(b,(c * (1 / a)))
        c3 = np.vdot(c,(b * (1 / a)))
        c4 = np.vdot(c,(c * (1 / a)))
        c5 = np.vdot(b,d)
        c6 = np.vdot(c,d)

        lamb2 = (( c3 * c4) - ( c1 * c6 ))/(( c2 * c3 ) - ( c1 * c4 ))
        lamb1 = (c3 - lamb2 * c2) / c1


        # Update velocities
        self.HalfVel = self.oldVel + ( lamb1 * del_phi1 * ( 1 / self.masses[:,None] ) ) + ( lamb2 * del_phi2 * ( 1 / self.masses[:,None] ) )
        self.oldVel = self.HalfVel

        # Let MDstep method know a constraint has occured and not to update the forces
        self.constrained = True
