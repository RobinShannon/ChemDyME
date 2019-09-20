import os
import Reaction as rxn
import Trajectory
import inout as io
import MasterEq
import ConnectTools as CT
from shutil import copyfile
import multiprocessing
import numpy as np
from ase import Atoms


class ReactionNetwork:

    def __init__(self, starting_species, exploring_trajectory, master_equation, level_list,
                 bimolecular_start = False, cores = 1, iterations = 4):

        self.start = starting_species
        self.trajectory = exploring_trajectory
        self.master_equation = master_equation
        self.level_list = level_list
        self.bimolecular_start = bimolecular_start
        self.cores = cores
        self.iterations = iterations
        self.path = os.getcwd()
        self.restart = False
        self.mechanism_run_time = 0.0

        # Check whether there is a directory for putting calcuation data in. If not create it
        if not os.path.exists(self.path + '/Raw'):
            os.mkdir(self.path + '/Raw')

        # Make working directories for each core
        for i in range(0, self.cores):
            if not os.path.exists(self.path + '/Raw/' + str(i)):
                os.mkdir(self.path + '/Raw/' + str(i))

    def run(glo):




        # Create Molecule object for the current reactant
        reacs = dict(("reac_" + str(i), rxn.Reaction(glo.cartesians, glo.species, i, glo)) for i in range(glo.cores))

        # Initialise Master Equation object
        me = MasterEq.MasterEq()

        # Open files for saving summary
        mainsumfile = open(('mainSummary.txt'), "a")

        while mechanismRunTime < glo.maxSimulationTime:

            # Minimise starting Geom and write summary xml for channel
            if reacs['reac_0'].have_reactant == False:
                outputs = []
                if __name__ == 'Main':
                    arguments = []
                    for i in range(0, glo.cores):
                        name = 'reac_' + str(i)
                        arguments.append(reacs[name])
                    p = multiprocessing.Pool(glo.cores)
                    results = p.map(minReac, arguments)
                    outputs = [result for result in results]

                for i in range(0, glo.cores):
                    name = 'reac_' + str(i)
                    reacs[name] = outputs[i]

            else:
                for i in range(0, glo.cores):
                    name = 'reac_' + str(i)
                    reacs[name].have_reactant = False

            # Update path for new minima
            minpath = syspath + '/' + reacs['reac_0'].ReacName

            # Get smiles name for initial geom and create directory for first minimum
            if not os.path.exists(minpath):
                os.makedirs(minpath)

            # Copy MESMER file from mes folder
            MESpath = syspath + '/MESMER/'
            symb = "".join(reacs[name].CombReac.get_chemical_symbols())
            if reacs['reac_0'].energyDictionary[symb] == 0.0:
                for i in range(0, glo.cores):
                    name = 'reac_' + str(i)
                    d = {symb: reacs[name].reactantEnergy}
                    reacs[name].energyDictionary.update(d)

            # If a MESMER file has not been created for the current minima then create one
            if not os.path.exists(MESpath):
                os.makedirs(MESpath)
                copyfile('mestemplate.xml', MESpath + 'mestemplate.xml')
                copyfile('mestemplate.xml', MESpath + 'mestemplateFull.xml')
                MESFullPath = MESpath + 'mestemplateFull.xml'
                MESpath = MESpath + 'mestemplate.xml'
                io.writeMinXML(reacs['reac_0'], MESpath, True, False)
                io.writeMinXML(reacs['reac_0'], MESFullPath, True, False)
                if reacs['reac_0'].is_bimol_reac == True:
                    io.writeMinXML(reacs['reac_0'], MESpath, True, True)
                    io.writeMinXML(reacs['reac_0'], MESFullPath, True, True)
                glo.restart = False
            else:
                MESFullPath = MESpath + 'mestemplateFull.xml'
                MESpath = MESpath + 'mestemplate.xml'

            # If this is a restart then need to find the next new product from the ME, otherwise start trajectories
            if glo.restart == False:
                # Open files for saving summary
                sumfile = open((minpath + '/summary.txt'), "w")

                reacs['reac_0'].printReac(minpath)
                for r in range(0, glo.ReactIters):
                    tempPaths = dict(
                        ("tempPath_" + str(i), minpath + '/temp' + str(i) + '_' + str(r)) for i in range(glo.cores))
                    # Now set up tmp directory for each thread
                    for i in range(0, glo.cores):
                        if not os.path.exists(tempPaths[('tempPath_' + str(i))]):
                            os.makedirs(tempPaths[('tempPath_' + str(i))])

                    if r % 2 == 0:
                        glo.trajMethod = glo.trajMethod1
                        glo.trajLevel = glo.trajLevel1
                    else:
                        glo.trajMethod = glo.trajMethod2
                        glo.trajLevel = glo.trajLevel2

                    # If this is the first species and it is a bimolecular channel, then initialise a bimolecular trajectory
                    # Otherwise initialise unimolecular trajectory at minima
                    if glo.InitialBi == True:
                        trajs = dict(("traj_" + str(i), Trajectory.Trajectory(reacs[('reac_' + str(i))].CombReac, glo,
                                                                              tempPaths[('tempPath_' + str(i))], str(i),
                                                                              True)) for i in range(glo.cores))
                    else:
                        trajs = dict(("traj_" + str(i), Trajectory.Trajectory(reacs[('reac_' + str(i))].CombReac, glo,
                                                                              tempPaths[('tempPath_' + str(i))], str(i),
                                                                              False)) for i in range(glo.cores))

                    results2 = []
                    outputs2 = []
                    if __name__ == "Main":
                        arguments1 = []
                        arguments2 = []
                        for i in range(0, glo.cores):
                            name = 'reac_' + str(i)
                            name2 = 'traj_' + str(i)
                            arguments1.append(reacs[name])
                            arguments2.append(trajs[name2])
                        arguments = list(
                            zip(arguments1, arguments2, [minpath] * glo.cores, [MESpath] * glo.cores, range(glo.cores),
                                [glo] * glo.cores))
                        p = multiprocessing.Pool(glo.cores)
                        results2 = p.map(runNormal, arguments)
                        outputs2 = [result for result in results2]

                    for i in range(0, glo.cores):
                        name = 'reac_' + str(i)
                        reacs[name] = outputs2[i][0]
                        sumfile.write(str(reacs[name].ProdName) + '_' + str(reacs[name].biProdName) + '\t' + str(
                            reacs[name].forwardBarrier) + '\t' + str(outputs2[i][1].numberOfSteps))
                        sumfile.flush()

                # run a master eqution to estimate the lifetime of the current species
                me.runTillReac(MESpath)
                me.newSpeciesFound = False

                # check whether there is a possible bimolecular rection for current intermediate
                if len(glo.BiList) > 0 and glo.InitialBi == False:
                    for i in range(0, len(glo.BiList)):
                        baseXYZ = reacs['reac_0'].CombReac.get_chemical_symbols()
                        if me.time > (1 / float(glo.BiRates[i])):
                            print("assessing whether or not to look for bimolecular channel. Rate = " + str(
                                float(glo.BiRates[i])) + "Mesmer reaction time = " + str(me.time))
                            glo.InitialBi = True
                            xyz = CT.get_bi_xyz(reacs['reac_0'].ReacName, glo.BiList[i])
                            spec = np.append(baseXYZ, np.array(glo.BiList[i].get_chemical_symbols()))
                            combinedMol = Atoms(symbols=spec, positions=xyz)
                            # Set reaction instance
                            for j in range(0, glo.cores):
                                name = 'reac_' + str(j)
                                d = {symb: reacs[name].reactantEnergy}
                                reacs[name].re_init_bi(xyz, spec)
                                biTrajs = dict(("traj_" + str(k),
                                                Trajectory.Trajectory(combinedMol, glo, tempPaths[('tempPath_' + str(k))],
                                                                      str(k), True)) for k in range(glo.cores))
                                biTempPaths = dict(
                                    ("tempPath_" + str(k), minpath + '/temp' + str(j)) for k in range(glo.cores))
                            if __name__ == "Main":
                                arguments1 = []
                                arguments2 = []
                                for j in range(0, glo.cores):
                                    name = 'reac_' + str(j)
                                    name2 = 'traj_' + str(j)
                                    biTrajs[name2].fragIdx = (len(baseXYZ), len(xyz))
                                    arguments1.append(reacs[name])
                                    arguments2.append(biTrajs[name2])
                                arguments = list(zip(arguments1, arguments2, [minpath] * glo.cores, [MESpath] * glo.cores,
                                                     range(glo.cores), [glo] * glo.cores, [glo.BiList[i]] * glo.cores))
                                p = multiprocessing.Pool(glo.cores)
                                p.map(runNormal, arguments)
                                glo.InitialBi = False

                # Run ME from the given minimum. While loop until species formed is new
                sumfile.close()
                glo.restart = False
            glo.InitialBi = False
            while me.newSpeciesFound == False:
                me.runTillReac(MESpath)
                mechanismRunTime += me.time
                out = me.prodName + '     ' + str(mechanismRunTime) + '\n'
                me.visitedList.append(me.prodName)
                mainsumfile.write(out)
                mainsumfile.flush()
                if not os.path.exists(syspath + '/' + me.prodName):
                    os.makedirs(syspath + '/' + me.prodName)
                    for i in range(0, glo.cores):
                        if os.path.exists(syspath + '/' + reacs[('reac_' + str(i))].ReacName + '/' + me.prodName):
                            reacs[('reac_' + str(i))].newReac(
                                syspath + '/' + reacs[('reac_' + str(i))].ReacName + '/' + me.prodName, me.prodName, False)
                        else:
                            print("cant find path " + str(
                                syspath + '/' + reacs[('reac_' + str(i))].ReacName + '/' + me.prodName))
                            try:
                                reacs[('reac_' + str(i))].newReac(syspath + '/' + me.prodName, me.prodName, True)
                            except:
                                reacs[('reac_' + str(i))].newReacFromSMILE(me.prodName)
                    io.update_me_start(me.prodName, me.ene, MESpath)
                    me.newSpeciesFound = True
                else:
                    if me.repeated() == True:
                        me.equilCount += 1
                        if me.equilCount >= 20:
                            mainsumfile.write(
                                'lumping' + ' ' + str(reacs['reac_0'].ReacName) + ' ' + str(me.prodName) + '\n')
                            me.prodName = io.lumpSpecies(reacs['reac_0'].ReacName, me.prodName, MESpath, MESpath)
                            mainsumfile.flush()
                            me.equilCount = 1
                    minpath = syspath + '/' + me.prodName
                    for i in range(0, glo.cores):
                        if os.path.exists(syspath + '/' + reacs[('reac_' + str(i))].ReacName + '/' + me.prodName):
                            reacs[('reac_' + str(i))].newReac(
                                syspath + '/' + reacs[('reac_' + str(i))].ReacName + '/' + me.prodName, me.prodName, False)
                        else:
                            try:
                                reacs[('reac_' + str(i))].newReac(syspath + '/' + me.prodName, me.prodName, True)
                            except:
                                reacs[('reac_' + str(i))].newReacFromSMILE(me.prodName)
                    io.update_me_start(me.prodName, me.ene, MESpath)

            me.newspeciesFound = False
            glo.restart = False

        mainsumfile.close()











