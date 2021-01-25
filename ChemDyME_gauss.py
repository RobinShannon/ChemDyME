import os
import copy
from collections.abc import Iterable
from shutil import which
from typing import Dict, Optional
from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator, EnvironmentError
from pathlib import Path
from shutil import copyfile
import re

class GaussianDynamics:
    calctype = 'optimizer'
    delete = ['force']
    keyword: Optional[str] = None
    special_keywords: Dict[str, str] = dict()

    def __init__(self, atoms, calc=None):
        self.atoms = atoms
        if calc is not None:
            self.calc = calc
        else:
            if self.atoms.calc is None:
                raise ValueError("{} requires a valid Gaussian calculator "
                                 "object!".format(self.__class__.__name__))

            self.calc = self.atoms.calc

    def todict(self):
        return {'type': self.calctype,
                'optimizer': self.__class__.__name__}

    def delete_keywords(self, kwargs):
        """removes list of keywords (delete) from kwargs"""
        for d in self.delete:
            kwargs.pop(d, None)

    def set_keywords(self, kwargs):
        args = kwargs.pop(self.keyword, [])
        if isinstance(args, str):
            args = [args]
        elif isinstance(args, Iterable):
            args = list(args)

        for key, template in self.special_keywords.items():
            if key in kwargs:
                val = kwargs.pop(key)
                args.append(template.format(val))

        kwargs[self.keyword] = args

    def run(self, **kwargs):
        calc_old = self.atoms.calc
        params_old = copy.deepcopy(self.calc.parameters)

        self.delete_keywords(kwargs)
        self.delete_keywords(self.calc.parameters)
        self.set_keywords(kwargs)

        self.calc.set(**kwargs)
        self.atoms.calc = self.calc

        try:
            self.atoms.get_potential_energy()
        except OSError:
            converged = False
        else:
            converged = True

        atoms = read(self.calc.label + '.log')
        self.atoms.cell = atoms.cell
        self.atoms.positions = atoms.positions

        self.calc.parameters = params_old
        self.calc.reset()
        if calc_old is not None:
            self.atoms.calc = calc_old

        return converged


class GaussianOptimizer(GaussianDynamics):
    keyword = 'opt'
    special_keywords = {
        'fmax': '{}',
        'steps': 'maxcycle={}',
    }


class GaussianIRC(GaussianDynamics):
    keyword = 'irc'
    special_keywords = {
        'direction': '{}',
        'steps': 'maxpoints={}',
    }


class Gaussian(FileIOCalculator):
    implemented_properties = ['energy', 'forces', 'dipole']
    command = 'GAUSSIAN < PREFIX.com > PREFIX.log'
    discard_results_on_any_change = True

    def __init__(self, *args, label='', **kwargs):
        label = label + 'gaussian'
        FileIOCalculator.__init__(self, *args, label=label, **kwargs)

    def calculate(self, *args, **kwargs):
        gaussians = ('g16', 'g09', 'g03')
        if 'GAUSSIAN' in self.command:
            for gau in gaussians:
                if which(gau):
                    self.command = self.command.replace('GAUSSIAN', gau)
                    break
            else:
                raise EnvironmentError('Missing Gaussian executable {}'
                                       .format(gaussians))
        try:
            FileIOCalculator.calculate(self, *args, **kwargs)
        except:
            print('Gaussian error')
            i = 0
            while Path('gauserror'+str(i)+'.log').exists():
                i += 1
            copyfile(self.label + '.log', 'gauserror'+str(i)+'.log')
    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        write(self.label + '.com', atoms, properties=properties,
              format='gaussian-in', **self.parameters)

    def read_results(self):
        output = read(self.label + '.log', format='gaussian-out')
        try:
            self.calc = output.calc
            self.results = output.calc.results
        except:
            print('Gaussian error')
            i = 0
            while Path('gauserror'+str(i)+'.log').exists():
                i += 1
            copyfile(self.label + '.log', 'gauserror'+str(i)+'.log')


    # Method(s) defined in the old calculator, added here for
    # backwards compatibility
    def clean(self):
        for suffix in ['.com', '.chk', '.log']:
            try:
                os.remove(os.path.join(self.directory, self.label + suffix))
            except OSError:
                pass

    def get_version(self):
        raise NotImplementedError  # not sure how to do this yet

    def minimise_stable(self, path = os.getcwd(), atoms: Optional[Atoms] = None):
        opt = GaussianOptimizer(atoms, self)
        opt.run(steps=100, opt='calcall cartesian')



    def minimise_ts_only(self, atoms):
        opt = GaussianOptimizer(atoms, self)
        opt.run(steps=100, opt='calcall, ts, noeigentest, cartesian')

    def read_vibs(self):
        vibs = []
        zpe = 0
        inp = open(str(self.label) + '.log', "r")
        for line in inp:
            if re.search("Frequencies", line):
                l = line.split()
                vibs.append(float(l[2]))
                zpe += float(l[2])
                try:
                    vibs.append(float(l[3]))
                    zpe += float(float(l[3]))
                except:
                    pass
                try:
                    vibs.append(float(l[4]))
                    zpe += float(float(l[4]))
                except:
                    pass
        zpe *= 0.00012
        zpe /= 2
        return vibs, zpe

    def read_ts_vibs(self):
        vibs = []
        zpe = 0
        inp = open(str(self.label) + '.log', "r")
        for line in inp:
            if re.search("Frequencies", line):
                try:
                    l = line.split()
                    vibs.append(float(l[2]))
                    zpe += float(l[2])
                    vibs.append(float(l[3]))
                    zpe += float(l[3])
                    vibs.append(l[4])
                    zpe += float(float(l[4]))
                except:
                    pass
            if re.search("Error termination"):
                return 0
        if vibs[0] > -250:
            print("GaussianTS has no imaginary Frequency")
            return
        if vibs[1] < 0:
            print("GaussianTS has more than 1 imaginary Frequency")
            return

        zpe -= vibs[0]
        imaginaryFreq = abs((vibs[0]))
        vibs.pop(0)
        zpe *= 0.00012
        zpe /= 2
        return vibs, zpe, imaginaryFreq