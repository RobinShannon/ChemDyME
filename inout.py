import xml.etree.cElementTree as ET
import ChemDyME.Tools
import numpy as np
from xml.dom import minidom

def writeTSXML(React, path):
    spinMult = 2

    cml = ChemDyME.Tools.getCML(React.TS,'TS_' + React.ReacName + '_' + React.ProdName)

    ET.register_namespace('me', 'http://www.chem.leeds.ac.uk/mesmer')
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    ET.register_namespace('', 'http://www.xml-cml.org/schema')

    imaginaryFreq = str(React.imaginaryFreq)
    freqs = str(React.TSFreqs)
    freqs = freqs[1:-1]
    freqs = freqs.replace('[', '')
    freqs = freqs.replace(']', '')
    freqs = freqs.replace(',','')


    with open(path, 'r') as myfile:
        data=myfile.read().replace('\n', '')
    tree = ET.fromstring(cml)
    bigTree = ET.fromstring(data)

    prop = ET.Element("propertyList")
    vib = ET.SubElement(prop, "property", dictRef ="me:vibFreqs")
    vibarrays = ET.SubElement(vib,"array", units="cm-1")
    vibarrays.text = freqs
    Imagvib = ET.SubElement(prop, "property", dictRef ="me:imFreq")
    Imag = ET.SubElement(Imagvib,"scalar", units="cm-1")
    Imag.text = imaginaryFreq
    Mult = ET.SubElement(prop, "property", dictRef ="me:spinMultiplicity")
    Multi = ET.SubElement(Mult,"scalar", units="cm-1")
    Multi.text = str(spinMult)
    ene = ET.SubElement(prop, "property", dictRef ="me:ZPE")
    enedata = ET.SubElement(ene,"scalar", units="kJ/mol")
    symb  = "".join(React.CombReac.get_chemical_symbols())
    enedata.text = str((React.forwardBarrier - React.energyDictionary[symb]) * 96.45)

    # Append the new "data" elements to the root element of the XML document
    tree.append(prop)

    children = bigTree.getchildren()
    children[1] = children[1].append(tree)

    # Now we have a new well-formed XML document. It is not very nicely formatted...
    out = ET.tostring(bigTree)

    # ...so we'll use minidom to make the output a little prettier
    dom = minidom.parseString(out)
    ChemDyME.Tools.prettyPrint(dom, path)

def writeTSXML2(React, path):
    spinMult = 2
    
    cml = ChemDyME.Tools.getCML(React.TS2,'TS2_' + React.ReacName + '_' + React.ProdName)
    
    ET.register_namespace('me', 'http://www.chem.leeds.ac.uk/mesmer')
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    ET.register_namespace('', 'http://www.xml-cml.org/schema')
    
    imaginaryFreq = str(React.imaginaryFreq2)
    freqs = str(React.TS2Freqs)
    freqs = freqs[1:-1]
    freqs = freqs.replace(',','')
    
    with open(path, 'r') as myfile:
        data=myfile.read().replace('\n', '')
    tree = ET.fromstring(cml)
    bigTree = ET.fromstring(data)

    prop = ET.Element("propertyList")
    vib = ET.SubElement(prop, "property", dictRef ="me:vibFreqs")
    vibarrays = ET.SubElement(vib,"array", units="cm-1")
    vibarrays.text = freqs
    Imagvib = ET.SubElement(prop, "property", dictRef ="me:imFreq")
    Imag = ET.SubElement(Imagvib,"scalar", units="cm-1")
    Imag.text = imaginaryFreq
    Mult = ET.SubElement(prop, "property", dictRef ="me:spinMultiplicity")
    Multi = ET.SubElement(Mult,"scalar", units="cm-1")
    Multi.text = str(spinMult)
    ene = ET.SubElement(prop, "property", dictRef ="me:ZPE")
    enedata = ET.SubElement(ene,"scalar", units="kJ/mol")
    symb  = "".join(React.CombReac.get_chemical_symbols())
    enedata.text = str((React.forwardBarrier2 - React.energyDictionary[symb]) * 96.45 )
    
    # Append the new "data" elements to the root element of the XML document
    tree.append(prop)
    
    children = bigTree.getchildren()
    children[1] = children[1].append(tree)
    
    # Now we have a new well-formed XML document. It is not very nicely formatted...
    out = ET.tostring(bigTree)
    
    # ...so we'll use minidom to make the output a little prettier
    dom = minidom.parseString(out)
    ChemDyME.Tools.prettyPrint(dom, path)

def writeMinXML(React, path, isReactant, isBi):

    spinMult = 2
    if isReactant and isBi:
        cml = ChemDyME.Tools.getCML(React.biReac,React.biReacName)
        spinMult = ChemDyME.Tools.getSpinMult(React.biReac, React.biReacName)
        freqs = str(React.biReacFreqs)
        potE = 0.0
    elif isReactant and not isBi:
        cml = ChemDyME.Tools.getCML(React.Reac,React.ReacName)
        spinMult = ChemDyME.Tools.getSpinMult(React.Reac, React.ReacName)
        freqs = str(React.ReacFreqs)
        symb  = "".join(React.CombReac.get_chemical_symbols())
        potE = React.reactantEnergy - React.energyDictionary[symb]
    elif not isReactant and isBi:
        cml = ChemDyME.Tools.getCML(React.biProd,React.biProdName)
        spinMult = ChemDyME.Tools.getSpinMult(React.biProd, React.biProdName)
        freqs = str(React.biProdFreqs)
        potE = 0.0
    elif not isReactant and not isBi:
        cml = ChemDyME.Tools.getCML(React.Prod,React.ProdName)
        spinMult = ChemDyME.Tools.getSpinMult(React.Prod, React.ProdName)
        freqs = str(React.ProdFreqs)
        symb  = "".join(React.CombProd.get_chemical_symbols())
        potE = React.productEnergy - React.energyDictionary[symb]

    #Convert from ev to kJ
    potE = potE * 96.485

    freqs = freqs[1:-1]
    freqs = freqs.replace(',','')

    ET.register_namespace('me', 'http://www.chem.leeds.ac.uk/mesmer')
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    ET.register_namespace('', 'http://www.xml-cml.org/schema')

    with open(path, 'r') as myfile:
        data=myfile.read().replace('\n', '')

    tree = ET.fromstring(cml)
    bigTree = ET.fromstring(data)

    prop = ET.Element("propertyList")
    lumpedSpecies = ET.SubElement(prop, "property", dictRef ="me:lumpedSpecies")
    lumpArrays = ET.SubElement(lumpedSpecies,"array")
    lumpArrays.text = " "
    vib = ET.SubElement(prop, "property", dictRef ="me:vibFreqs")
    vibarrays = ET.SubElement(vib,"array", units="cm-1")
    vibarrays.text = str(freqs)
    ene = ET.SubElement(prop, "property", dictRef ="me:ZPE")
    enedata = ET.SubElement(ene,"scalar", units="kJ/mol")
    enedata.text = str(potE)
    Mult = ET.SubElement(prop, "property", dictRef ="me:spinMultiplicity")
    multi = ET.SubElement(Mult,"scalar", units="cm-1")
    multi.text = str(spinMult)
    epsilon = ET.SubElement(prop, "property", dictRef ="me:epsilon")
    epsilondata = ET.SubElement(epsilon,"scalar")
    epsilondata.text = '473.17'
    sigma = ET.SubElement(prop, "property", dictRef ="me:sigma")
    sigmadata = ET.SubElement(sigma,"scalar")
    sigmadata.text = '5.09'

    eneTrans = ET.Element('me:energyTransferModel')
    eneTrans.set("{http://www.w3.org/2001/XMLSchema-instance}type","me:ExponentialDown")
    eTran = ET.SubElement(eneTrans,"scalar", units="cm-1")
    eTran.text = '250'

    # Append the new "data" elements to the root element of the XML document
    tree.append(prop)
    tree.append(eneTrans)

    children = bigTree.getchildren()
    children[1] = children[1].append(tree)

    if isReactant and not isBi:
        initPop = ET.Element('me:InitialPopulation')
        iPop = ET.SubElement(initPop,"me:molecule", ref=React.ReacName, population="1.0")
        children[3] = children[3].append(initPop)



    # Now we have a new well-formed XML document. It is not very nicely formatted...
    out = ET.tostring(bigTree)

    # ...so we'll use minidom to make the output a little prettier
    dom = minidom.parseString(out)
    ChemDyME.Tools.prettyPrint(dom, path)

def writeCombXML(React, path):

    spinMult = 2
    cml = ChemDyME.Tools.getCML(React.combProd,React.combProdName)
    spinMult = ChemDyME.Tools.getSpinMult(React.combProd, React.combProdName)
    freqs = str(React.combProdFreqs)
    symb = "".join(React.CombProd.get_chemical_symbols())
    potE = React.combProductEnergy - React.energyDictionary[symb]


    #Convert from ev to kJ
    potE = potE * 96.485

    freqs = freqs[1:-1]
    freqs = freqs.replace(',','')

    ET.register_namespace('me', 'http://www.chem.leeds.ac.uk/mesmer')
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    ET.register_namespace('', 'http://www.xml-cml.org/schema')

    with open(path, 'r') as myfile:
        data=myfile.read().replace('\n', '')

    tree = ET.fromstring(cml)
    bigTree = ET.fromstring(data)

    prop = ET.Element("propertyList")
    lumpedSpecies = ET.SubElement(prop, "property", dictRef ="me:lumpedSpecies")
    lumpArrays = ET.SubElement(lumpedSpecies,"array")
    lumpArrays.text = " "
    vib = ET.SubElement(prop, "property", dictRef ="me:vibFreqs")
    vibarrays = ET.SubElement(vib,"array", units="cm-1")
    vibarrays.text = str(freqs)
    ene = ET.SubElement(prop, "property", dictRef ="me:ZPE")
    enedata = ET.SubElement(ene,"scalar", units="kJ/mol")
    enedata.text = str(potE)
    Mult = ET.SubElement(prop, "property", dictRef ="me:spinMultiplicity")
    multi = ET.SubElement(Mult,"scalar", units="cm-1")
    multi.text = str(spinMult)
    epsilon = ET.SubElement(prop, "property", dictRef ="me:epsilon")
    epsilondata = ET.SubElement(epsilon,"scalar")
    epsilondata.text = '473.17'
    sigma = ET.SubElement(prop, "property", dictRef ="me:sigma")
    sigmadata = ET.SubElement(sigma,"scalar")
    sigmadata.text = '5.09'

    eneTrans = ET.Element('me:energyTransferModel')
    eneTrans.set("{http://www.w3.org/2001/XMLSchema-instance}type","me:ExponentialDown")
    eTran = ET.SubElement(eneTrans,"scalar", units="cm-1")
    eTran.text = '250'

    # Append the new "data" elements to the root element of the XML document
    tree.append(prop)
    tree.append(eneTrans)

    children = bigTree.getchildren()
    children[1] = children[1].append(tree)



    # Now we have a new well-formed XML document. It is not very nicely formatted...
    out = ET.tostring(bigTree)

    # ...so we'll use minidom to make the output a little prettier
    dom = minidom.parseString(out)
    ChemDyME.Tools.prettyPrint(dom, path)


def writeReactionXML(React,path, printTS2):

    ET.register_namespace('me', 'http://www.chem.leeds.ac.uk/mesmer')
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    ET.register_namespace('', 'http://www.xml-cml.org/schema')

    name=React.ReacName + '_' + React.ProdName

    with open(path, 'r') as myfile:
        data=myfile.read().replace('\n', '')

    bigTree = ET.fromstring(data)
    act = 0
    symb = "".join(React.CombReac.get_chemical_symbols())
    #buffer is added to rxn element as the print function removes the third word
    rxn = ET.Element("reaction", id=name, active='true' )
    if React.is_bimol_reac == False and React.is_bimol_prod == False:
        rct = ET.SubElement(rxn, "reactant")
        ml1 = ET.SubElement(rct, "molecule", ref = React.ReacName)
        ml1.set("{http://www.chem.leeds.ac.uk/mesmer}type","modelled")
    elif React.is_bimol_reac == False and React.is_bimol_prod == True:
        rct = ET.SubElement(rxn, "product")
        ml1 = ET.SubElement(rct, "molecule", ref =React.ReacName)
        ml1.set("{http://www.chem.leeds.ac.uk/mesmer}type","modelled")
    else:
        rct1 = ET.SubElement(rxn, "reactant")
        ml1_1 = ET.SubElement(rct1, "molecule", ref = React.ReacName)
        ml1_1.set("{http://www.chem.leeds.ac.uk/mesmer}type","modelled")
        rct2 = ET.SubElement(rxn, "reactant")
        ml1_2 = ET.SubElement(rct2, "molecule", ref = React.biReacName)
        ml1_2.set("{http://www.chem.leeds.ac.uk/mesmer}type","excessReactant")
        if React.barrierlessReaction:
            act = 0
        else:
            act = ((React.forwardBarrier - React.energyDictionary[symb]) * 96.45)
        if act < 0:
            act = 0
        React.barrierlessReaction = True
    if React.is_bimol_prod == False:
        prod = ET.SubElement(rxn, "product")
        pml1 = ET.SubElement(prod, "molecule", ref = React.ProdName)
        pml1.set("{http://www.chem.leeds.ac.uk/mesmer}type","modelled")
    elif React.is_bimol_reac == False:
        prod1 = ET.SubElement(rxn, "reactant")
        pml1_1 = ET.SubElement(prod1, "molecule", ref = React.ProdName)
        pml1_1.set("{http://www.chem.leeds.ac.uk/mesmer}type","modelled")
        prod2 = ET.SubElement(rxn, "reactant")
        pml1_2 = ET.SubElement(prod2, "molecule", ref = React.biProdName)
        pml1_2.set("{http://www.chem.leeds.ac.uk/mesmer}type","excessReactant")
        React.barrierlessReaction = True
        act = ((React.forwardBarrier - (React.productEnergy-React.energyDictionary[symb])) * 96.45)
        if act < 0:
            act = 0
    else:
        prod1 = ET.SubElement(rxn, "product")
        pml1_1 = ET.SubElement(prod1, "molecule", ref = React.ProdName)
        pml1_1.set("{http://www.chem.leeds.ac.uk/mesmer}type","modelled")

    # Transition state section, if not present then ILT
    if React.barrierlessReaction == False:
        TS = ET.SubElement(rxn, "{http://www.chem.leeds.ac.uk/mesmer}transitionState")
        if printTS2:
            ts1 = ET.SubElement(TS, "molecule", ref = ('TS2_' + React.ReacName + '_' + React.ProdName ))
        else:
            ts1 = ET.SubElement(TS, "molecule", ref = ('TS_' + React.ReacName + '_' + React.ProdName ))
        ts1.set("{http://www.chem.leeds.ac.uk/mesmer}type","transitionState")
        RRKM = ET.SubElement(rxn, "{http://www.chem.leeds.ac.uk/mesmer}MCRCMethod", name = "SimpleRRKM")
    else:
        ILT = ET.SubElement(rxn, "{http://www.chem.leeds.ac.uk/mesmer}MCRCMethod")
        ILT.set("{http://www.w3.org/2001/XMLSchema-instance}type","MesmerILT")
        preExp= ET.SubElement(ILT, "{http://www.chem.leeds.ac.uk/mesmer}preExponential", units="cm3 molecule-1 s-1")
        preExp.text = '1E-10'
        preExp= ET.SubElement(ILT, "{http://www.chem.leeds.ac.uk/mesmer}activationEnergy", units="kJ/mol")
        preExp.text = str(act)
        preExp= ET.SubElement(ILT, "{http://www.chem.leeds.ac.uk/mesmer}TInfinity")
        preExp.text = '298.0'
        preExp= ET.SubElement(ILT, "{http://www.chem.leeds.ac.uk/mesmer}nInfinity")
        preExp.text = '0.0'
        excess = ET.SubElement(rxn, "{http://www.chem.leeds.ac.uk/mesmer}excessReactantConc")
        excess.text = '1E18'

   # Append the new "data" elements to the root element of the XML document
    children = bigTree.getchildren()
    children[2] = children[2].append(rxn)

    # Now we have a new well-formed XML document. It is not very nicely formatted...
    out = ET.tostring(bigTree)

    # ...so we'll use minidom to make the output a little prettier
    dom = minidom.parseString(out)
    ChemDyME.Tools.prettyPrint(dom, path)

def lumpSpecies(rName, pName, iPath, oPath):

    doc = minidom.parse(iPath)
    rName_bi = False
    pName_bi = False
    biChecks = doc.getElementsByTagName("reaction")
    for biCheck in biChecks:
        reacs = biCheck.getElementsByTagName("reactant")
        if reacs[0].getElementsByTagName("molecule")[0].getAttribute("ref") == rName and len(reacs) == 2:
             rName_bi = True
        elif reacs[0].getElementsByTagName("molecule")[0].getAttribute("ref") == pName and len(reacs) == 2:
            pName_bi = True

    names = doc.getElementsByTagName("molecule")
    for name in names:
        nid = name.getAttribute("id")
        if nid == rName:
            props = name.getElementsByTagName("property")
            for prop in props:
                pid = prop.getAttribute("dictRef")
                if pid == "me:ZPE":
                    ene = prop.getElementsByTagName("scalar")
                    e1 = float(ene[0].firstChild.nodeValue)
                    if rName_bi == True:
                        e1 -= 1000000.0
        elif nid == pName:
            props = name.getElementsByTagName("property")
            for prop in props:
                pid = prop.getAttribute("dictRef")
                if pid == "me:ZPE":
                    ene = prop.getElementsByTagName("scalar")
                    e2 = float(ene[0].firstChild.nodeValue)
                    if pName_bi == True:
                        e2 -= 1000000.0

    if (e1 < e2):
        rName = rName
        lName = pName
    else:
        lName = rName
        rName = pName

    aLump = ""
    names = doc.getElementsByTagName("molecule")
    for name in names:
        nid = name.getAttribute("id")
        if nid == lName:
            props = name.getElementsByTagName("property")
            for prop in props:
                pid = prop.getAttribute("dictRef")
                if pid == "me:lumpedSpecies":
                    array = prop.getElementsByTagName("array")
                    aLump = str(array[0].firstChild.nodeValue)
    for name in names:
        nid = name.getAttribute("id")
        if nid == rName:
            props = name.getElementsByTagName("property")
            for prop in props:
                pid = prop.getAttribute("dictRef")
                if pid == "me:lumpedSpecies":
                    array = prop.getElementsByTagName("array")
                    a = str(array[0].firstChild.nodeValue)
                    array[0].firstChild.nodeValue = a + str(lName) + aLump

    ChemDyME.Tools.prettyPrint(doc,oPath)
    return rName

def update_me_start(name, ene, path):

    doc = minidom.parse(path)
    cond = doc.getElementsByTagName("me:conditions")
    init = cond[0].getElementsByTagName("me:molecule")
    init[0].setAttribute("ref", str(name))
    init[0].setAttribute("grain", str(ene))

    ChemDyME.Tools.prettyPrint(doc, path)
