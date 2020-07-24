import os
import ChemDyME.Globals as g
import ChemDyME.Main as m
import ChemDyME.GenBXDMain as gm
import ChemDyME.DosMain  as dm
import scine_sparrow

calculation = scine_sparrow.Calculation('AM1')
calculation.set_elements(['H', 'H'])
calculation.set_positions([[0, 0, 0], [1, 0, 0]])
ene = calculation.calculate_energy()
print(str(ene))


path = os.getcwd()
gl = g.Globals(path)
if gl.RunType == "MechGen":
    m.run(gl)
elif gl.RunType == "GenBXD":
    gm.run(gl)
elif gl.RunType == "DOS":
    dm.run(gl)
