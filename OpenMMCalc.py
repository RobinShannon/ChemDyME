try:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
except:
    print("no openMM version found")

import numpy as np
import ForceField as ff
from ase.calculators.calculator import Calculator, all_changes


class OpenMMCalculator(Calculator):
    """
    Simple implementation of a ASE calculator for OpenMM.

    Parameters:
        input : PDB file with topology.
        nonbondedMethod : The nonbonded method to use (see https://simtk.org/api_docs/openmm/api10/classOpenMM_1_1NonbondedForce.html). Defaults to CutoffNonPeriodic.
        nonbondedCutoff : The nonbonded cutoff distance to use (in Angstroms). Default : 10 Angstroms.
    """
    implemented_properties = ['energy', 'forces']
    default_parameters = {'input' : "openmm.pdb",
                          'nonbondedMethod' : CutoffNonPeriodic,
                          'fileType' : "xyz",
                          'ASEmol': 0,
                          'nonbondedCutoff' : 10 * angstrom,
                          'atomTypes' : []}

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)
        input = self.parameters.input
        fileType = self.parameters.fileType
        if fileType == "pdb":
            pdb = PDBFile(input)
            forcefield = ForceField('amber99sb.xml')
            topology = pdb.topology
            positions = pdb.positions
            print("Generating OpenMM system")
            self.system = forcefield.createSystem(topology, nonbondedMethod=self.parameters.nonbondedMethod,
                                              nonbondedCutoff=self.parameters.nonbondedCutoff)
        if fileType == "xyz":
            print("Generating OpenMM system")
            self.system = self.setUpMM3(self.parameters.ASEmol, self.parameters.atomTypes)
            positions = [x for x in self.parameters.ASEmol.get_positions()]


        # Create a dummy integrator, this doesn't really matter.
        self.integrator = VerletIntegrator(0.001 * picosecond)
        self.platform = Platform.getPlatformByName("CPU")
        self.context = openmm.Context(self.system, self.integrator)
        self.context.setPositions(positions * angstrom)
        state = self.context.getState(getEnergy=True)
        print("Energy: ", state.getPotentialEnergy(), len(positions))
        self.n_atoms = len(positions)

    def calculate(self, atoms=None,
                  properties=['energy', 'forces'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        positions = [x for x in atoms.positions]
        self.context.setPositions(positions * angstrom)
        state = self.context.getState(getEnergy=True, getForces=True)
        energyKJMol = state.getPotentialEnergy()
        kjMol2ev = 0.01036; # ...roughly
        energy = energyKJMol.value_in_unit(kilojoules_per_mole) * kjMol2ev
        forcesOpenmm = state.getForces()
        # There must be a more elegant way of doing this
        forces = [[f.value_in_unit(kilojoule_per_mole/angstrom) * kjMol2ev for f in force] for force in
                  forcesOpenmm]
        self.results['energy'] = energy
        self.results['forces'] = np.array(forces)

    def setUpMM3(input, mol, types):

        masses = mol.get_masses() * amu
        nAtoms = len(masses)
        f = ff.MM3(mol,types)

        # Create a system and add particles to it
        system = openmm.System()
        for index in range(nAtoms):
            # Particles are added one at a time
            # Their indices in the System will correspond with their indices in the Force objects we will add later
            system.addParticle(masses[index])

        # Add Lennard-Jones interactions using a NonbondedForce. Only required so that openMM can set up exclusions.
        force = openmm.NonbondedForce()
        force.setUseDispersionCorrection(False) # use long-range isotropic dispersion correction
        force.addParticle(0.5,1,4.184)

        # Add custom bond term
        bondForce = openmm.CustomBondForce("k*0.5*(r-r_eq)*(r-r_eq)*(1.0+cs*(r-r_eq) + (7.0/12.0)*cs*cs*(r-r_eq)*(r-r_eq))")
        bondForce.addPerBondParameter("k")#Force constant
        bondForce.addPerBondParameter("r_eq")#Equilibrium distance
        #cs converted from 1/A to 1/nm
        bondForce.addGlobalParameter("cs",-2.55 *10.0)
        # Add custom force to system
        system.addForce(bondForce)
        # Iterate through bond list
        bondPairs = []
        for bond in f.bondList:
            #Add bond term to forces. Fields 1 and 2 are the atom indicies and fields 3 and 4 are parameters k and r_eq
            bondForce.addBond(bond[0],bond[1], [bond[2]*602.3*AngstromsPerNm*AngstromsPerNm, bond[3]*NmPerAngstrom])
            bondPairs.append((bond[0],bond[1]))

        # Custom angle term
        angleForce = openmm.CustomAngleForce("k *0.5 *dtheta*dtheta*expansion;""expansion= 1.0 -0.014*dtor*dtheta+ 5.6e-5*dtor^2*dtheta^2-1.0e-6*dtor^3*dtheta^3+2.2e-8*dtor^4*dtheta^4;""dtor=57.295779;""dtheta = theta- theta_eq")
        angleForce.addPerAngleParameter("k")
        angleForce.addPerAngleParameter("theta_eq")
        system.addForce(angleForce)
        for angle in f.angleList:
            #Add bond term to forces. Fields 1, 2 and 3 are the atom indicies and fields 4 and 5 are parameters k and r_eq
            angleForce.addAngle(angle[0],angle[1],angle[2], [angle[3]*602.2,angle[4]*RadiansPerDegree])

        LJforce = openmm.CustomNonbondedForce("4.0*epsilon*(sigma^12/r^12 - sigma^6/r^6);"
																				 "sigma=((sig1+sig2)/2.0);"
																				 "epsilon=sqrt(eps1*eps2)")
        #Add any required parameters
        LJforce.addPerParticleParameter("sig")
        LJforce.addPerParticleParameter("eps")

        #Pass each particle params
        for LJ in f.LJList:
            LJforce.addParticle([LJ[0]*NmPerAngstrom,LJ[1]*4.184])

        #After we've defined the LJ, we need to tell OpenMM to skip calculation where a bond or angle exists between to the two particles
        num_exceptions = force.getNumExceptions()

        LJforce.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffNonPeriodic)
        #Set cutoff distance in nm
        LJforce.setCutoffDistance(3.0*NmPerAngstrom)

        #Create exceptions for non-bonded interactions
        for b in bondPairs:
            force.addException(b[0], b[1], 1, 1.0/1.2, 1.0/2.0)
            LJforce.addExclusion(b[0],b[1])

        system.addForce(LJforce)

        #dihedralForce = openmm.CustomTorsionForce()
        #for index in range(nAtoms): # all particles must have parameters assigned for the NonbondedForce
            # Particles are assigned properties in the same order as they appear in the System object
        #    force4.addParticle(charge, sigma, epsilon)
        #force_index4 = system.addForce(force4)
        return system
