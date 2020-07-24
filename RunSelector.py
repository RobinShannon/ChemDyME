import os
import ChemDyME.Globals as g
import ChemDyME.Main as m
import ChemDyME.GenBXDMain as gm
import ChemDyME.DosMain  as dm


path = os.getcwd()
gl = g.Globals(path)
if gl.RunType == "MechGen":
    m.run(gl)
elif gl.RunType == "GenBXD":
    gm.run(gl)
elif gl.RunType == "DOS":
    dm.run(gl)
