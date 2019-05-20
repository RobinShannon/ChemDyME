try:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
    import simtk.openmm.app as app
    from simtk.openmm import XmlSerializer
except:
    print("no openMM version found")
import numpy as np
import time
import ForceField as ff
from io import StringIO
from tempfile import TemporaryDirectory
import os
from xml.dom.minidom import getDOMImplementation, parseString
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
        if fileType == "xyz":
            print("Generating OpenMM system")
            self.system,self.topology = self.setUpMM3(self.parameters.ASEmol, self.parameters.atomTypes)
            positions = [x for x in self.parameters.ASEmol.get_positions()]
        if fileType == "xml":
            print("Generating OpenMM system")
            f = open('OpenMM.xml','r')
            sys = f.read()
            #self.system = forcefield.createSystem(topology, nonbondedMethod=self.parameters.nonbondedMethod,nonbondedCutoff=self.parameters.nonbondedCutoff)
            self.system = XmlSerializer.deserialize(sys)
            #box_vec = self.system.getDefaultPeriodicBoxVectors()
            #self.parameters.ASEmol.set_cell([box_vec[0]._value[0]*10,box_vec[1]._value[1]*10,box_vec[2]._value[2]*10])
            #self.parameters.ASEmol.pbc = (True,True,True)
            #self.parameters.ASEmol.wrap()
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
        atoms.wrap()
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
        topology = app.topology.Topology()
        f = ff.MM3(mol,types)

        chain = topology.addChain('Main')
        res = topology.addResidue('mm3', chain)
        atoms = []
        for a,t in zip(masses, f.types):
            elem = app.element.Element.getByMass(a)
            atoms.append(topology.addAtom(str('mm3_')+str(t),elem,res))

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
            id1,id2 = bond[0],bond[1]
            topology.addBond(atoms[id1],atoms[id2])


        # Custom angle term
        angleForce = openmm.CustomAngleForce("k *0.5 *dtheta*dtheta*expansion;""expansion= 1.0 -0.014*dtor*dtheta+ 5.6e-5*dtor^2*dtheta^2-1.0e-6*dtor^3*dtheta^3+2.2e-8*dtor^4*dtheta^4;""dtor=57.295779;""dtheta = theta- theta_eq")
        angleForce.addPerAngleParameter("k")
        angleForce.addPerAngleParameter("theta_eq")
        system.addForce(angleForce)
        for angle in f.angleList:
            #Add bond term to forces. Fields 1, 2 and 3 are the atom indicies and fields 4 and 5 are parameters k and r_eq
            angleForce.addAngle(angle[0],angle[1],angle[2], [angle[3]*602.3,angle[4]*RadiansPerDegree])

        # Custom angle term
        # dihedralForce = openmm.CustomTorsionForce("0.5*V1*(1-cos(theta)) + 0.5*V2*(1-cos(2* (theta-3.141592)))+0.5*V3*(1-cos(3*theta))")
        # dihedralForce.addPerTorsionParameter("V1")
        # dihedralForce.addPerTorsionParameter("V2")
        # dihedralForce.addPerTorsionParameter("V3")
        # system.addForce(dihedralForce)
        # for dihedral in f.dihedralList:
        #     #Add bond term to forces. Fields 1, 2 and 3 are the atom indicies and fields 4 and 5 are parameters k and r_eq
        #     dihedralForce.addTorsion(dihedral[0],dihedral[1],dihedral[2],dihedral[3], [dihedral[4]*4.184,dihedral[5]*4.184,dihedral[6]*4.184])

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
        return system,topology

    """
    Serialize and deserialize OpenMM simulations to and from XML files.

    A simulation is described as the concatenation of a starting structure as a PDB
    file, an OpenMM serialized system, an OpenMM serialized integrator, and,
    optionally, an OpenMM serialized state. The resulting XML file looks like:

    ::
        <OpenMMSimulation>
            <pdb>
                // pasted content of the PDB file
            </pdb>
            <System ...>
                // XML content of the OpenMM serialized system
            </System>
            <Integrator ...>
                // XML content of the OpenMM serialized integrator
            </Integrator>
        </OpenMMSimulation>

    The ``System`` and ``Integrator`` tags are the roots of the serialized system
    and integrator, respectively.

    This module provides a function :fun:`serialize_simulation` that generates an
    XML file from an existing instance of :class:`simtk.openmm.app.Simulation`, and
    a function :fun:`deserialize_simulation` that creates an instance of simulation
    from an XML file.
    """

    def serialize_simulation(self) -> str:

        ROOT_TAG = 'OpenMMSimulation'

        """
        Generate an XML string from a simulation.

        :param simulation: The simulation to serialize.
        :return: A string with the content of an XML file describing the simulation.
        """
        implementation = getDOMImplementation()
        document = implementation.createDocument(None, ROOT_TAG, None)

        # Extract the PDB
        positions = self.context.getState(getPositions=True).getPositions()
        pdb_content = StringIO()
        app.PDBFile.writeFile(self.topology, positions, pdb_content)
        pdb_node = document.createElement('pdb')
        pdb_node.appendChild(document.createTextNode(pdb_content.getvalue()))

        # Extract the system
        system_xml_str = XmlSerializer.serialize(self.system)
        system_document = parseString(system_xml_str)

        # Extract the integrator
        integrator_xml_str = XmlSerializer.serialize(self.integrator)
        integrator_document = parseString(integrator_xml_str)

        # Combine the element in a single
        root = document.documentElement
        root.appendChild(pdb_node)
        root.appendChild(system_document.documentElement)
        root.appendChild(integrator_document.documentElement)

        return root.toprettyxml()
