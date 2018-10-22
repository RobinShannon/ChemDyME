import os
import Globals as g
import Main as m
import GenBXDMain as gm
import DosMain  as dm


path = os.getcwd()
gl = g.Globals(path)
if gl.RunType == "MechGen":
    m.run(gl)
elif gl.RunType == "GenBXD":
    gm.run(gl)
elif gl.RunType == "DOS":
    dm.run(gl)
