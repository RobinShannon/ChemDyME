from abc import ABCMeta, abstractmethod
import numpy as np
import ConnectTools as ct
import xml.etree.ElementTree as etree

class ForceField:

    def __init__(self, mol, types):
        self.types = types
        self.vdwList  = []
        self.bondList,self.angleList,self.dihedralList,self.LJList = self.getConectivity(mol)

    @abstractmethod
    def getConnectivity(self, mol):
        pass

class Narupa(ForceField):
    def getConnectivity(self, mol):
        file = 'template.xml'
        tree = etree.parse(file)
        if tree.getroot().find('AtomTypes') is not None:
            for type in tree.getroot().find('AtomTypes').findall('Type'):
                self.registerAtomType(type.attrib)



class MM3(ForceField):

    @abstractmethod
    def getConectivity(self, mol):
        size =len(mol.get_positions())
        dRef = ct.refBonds(mol)
        bondMat = ct.bondMatrix(dRef,mol)
        bondList = []
        angleList = []
        angleListTemp = []
        dihedralList = []
        LJList=[]
        fullBonds = []
        #Loop through bond matrix and add any bonds to the list
        for i in range(0,size):
            bondedToAtom = []
            for j in range(i,size):
                if bondMat[i][j] == 1.0:
                    K,re_eq = self.getBondParams(i,j,self.types)
                    bondList.append((i,j,K,re_eq))
                    bondedToAtom.append(j)
            fullBonds.append(bondedToAtom)

        #Loop through bond matrix and add any angles to the list
        #Must be a cleverer way to do this
        for i in range(0,size):
            for j in range(0,size):
                for k in range(0,size):
                    if bondMat[i][j] == 1.0 and (bondMat[j][k] == 1.0) and i != k:
                        if self.types[i] != self.types[k]:
                            a,b = self.types[i],self.types[k]
                        else:
                            a,b = i,k
                        if a <= b:
                            K,theta_eq = self.getAngleParams(i,j,k,self.types)
                            angleList.append((i,j,k,K,theta_eq))
                        if a > b:
                            K,theta_eq = self.getAngleParams(k,j,i,self.types)
                            angleList.append((k,j,i,K,theta_eq))
                        angleList = self.f7(angleList)
        for i in range(0,size):
            sigma,epsilon = self.getLJParams(i,self.types)
            LJList.append((sigma,epsilon))
        #Loop through bond matrix and add any dihedral to the list
        for i in range(0,size):
            for j in range(i,size):
                for k in range(0,size):
                    for l in range(0,size):
                        if bondMat[i][j] == 1.0 and bondMat[j][k] == 1.0 and bondMat[k][l] == 1.0:
                            dihedralList.append((i,j,k,l))
        return bondList,angleList,dihedralList,LJList

    def getBondParams(self,i,j,types):
        inp = open("mm3.prm", "r")
        K = 0
        r_eq = 0
        if types[i] <= types[j]:
            type1 = types[i]
            type2 = types[j]
        else:
            type1 = types[j]
            type2 = types[i]
        for line in inp:
            words = line.split()
            try:
                if words[0] == "bond" and float(words[1]) == type1 and float(words[2]) == type2:
                    K,r_eq = float(words[3]),float(words[4])
            except:
                pass
        return K,r_eq

    def getAngleParams(self,i,j,k,types):
        inp = open("mm3.prm", "r")
        K = 0
        theta_eq = 0
        for line in inp:
            words = line.split()
            try:
                if words[0] == "angle" and float(words[1]) == types[i] and float(words[2]) == types[j] and float(words[3]) == types[k]:
                    K,theta_eq = float(words[4]),float(words[5])
            except:
                pass
        return K,theta_eq

    def getLJParams(self,i,types):
        inp = open("mm3.prm", "r")
        sigma = 0
        epsilon = 0
        for line in inp:
            words = line.split()
            try:
                if words[0] == "vdw" and float(words[1]) == types[i]:
                    sigma,epsilon = float(words[2]),float(words[3])
            except:
                pass
        return sigma,epsilon

    def f7(self,seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

