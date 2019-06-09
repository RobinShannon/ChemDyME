from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import xml.etree.ElementTree as ET


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




