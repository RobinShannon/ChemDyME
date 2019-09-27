from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import xml.etree.ElementTree as ET
#import os
from sys import stdout, exit, stderr

def get_openmm_amber(prmtop_file, crd_file, out_xml):
    prmtop = AmberPrmtopFile(prmtop_file)
    inpcrd = AmberInpcrdFile(crd_file)
    positions = inpcrd.positions
    system = prmtop.createSystem(nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=10 * angstrom)
    out_xml.write(system)


def get_openmm_pdb(pdb,out_xml):
    pdb = PDBFile(pdb)
    forcefield = ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff,constraints=None, rigidWater=False)
    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    out_xml.write(system)

def get_openmm_narupa2(narupa, out_xml, out_pdb):
    file_in = open(narupa, "r")
    sys_out = open(out_xml, 'w')
    pdb_out = open(out_pdb, 'w')
    elem = ET.parse(file_in)
    root = elem.getroot()
    for child in root:
        if child.tag == "System":
            sys = ET.tostring(child).decode("utf-8")
            sys_out.write(sys)
        if child.tag == "pdb":
            pdb = ET.tostring(child).decode("utf-8")
            pdb_out.write(str(pdb))


def get_openmm_charmm(psf_file,crd_file,out_xml,params_dir):
    psf = CharmmPsfFile(psf_file)
    crd = CharmmCrdFile(crd_file)
    params = []
    for file in os.listdir(params_dir):
        if not file.startswith('.'):
          full_path = params_dir+'/'+file
          params.append(full_path)
    params = CharmmParameterSet(*params)
    system = psf.createSystem(params,nonbondedMethod=NoCutoff,nonbondedCutoff=1*nanometer, constraints=None, implicitSolvent=HCT )
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picosecond)
    simulation = Simulation(psf.topology, system,integrator)
    simulation.context.setPositions(crd.positions)
    simulation.minimizeEnergy()
    simulation.step(10000)
    serialized_system = XmlSerializer.serialize(system)
    out_file = open(out_xml, 'w')
    out_file.write(serialized_system)


#get_openmm_charmm(/Users/cm14sjm/Documents/BXD/BXD_CHARMM/step1_pdbreader.psf,/Users/cm14sjm/Documents/BXD/BXD_CHARMM/1tit_min_equi_final.crd, params_files = '/Users/cm14sjm/Documents/BXD/BXD_CHARMM/toppar.str', '/Users/cm14sjm/Documents/BXD/BXD_CHARMM/1tit_01_001.vel', '/Users/cm14sjm/Documents/BXD/BXD_CHARMM/1tit_01_001.dcd',/Users/cm14sjm/Documents/BXD/BXD_CHARMM/tit_out.xml)