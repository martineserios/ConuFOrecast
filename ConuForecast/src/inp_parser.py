# encoding: utf-8

#Viene de extraer2.

import swmmout
import subprocess
import numpy as np
import pandas as pd

swmmOutFileName = 'model.out'
places = {
    'links' : [
    ]
}


def parseSWMMFile(swmmOutFileName):

    nodos = {}
    links = {}

    with open(swmmOutFileName, 'r') as f:
        lines = f.readlines()

        def startsWith(linea, token):
            return linea[0:len(token)].lower() == token.lower()

        def procJunctions(linea):
            tokens = linea.split()
            if len(tokens) < 2:
                return
            nodos[tokens[0]] = {
                "elev": float(tokens[1]),
                "type": "junction",
                }

        def procStorage(linea):
            tokens = linea.split()
            if len(tokens) < 2:
                return
            nodos[tokens[0]] = {
                "elev": float(tokens[1]),
                "type": "storage",
                }

        def procCoordinates(linea):
            tokens = linea.split()
            if len(tokens) < 3:
                return
            name = tokens[0]
            if not name in nodos:
                return
            nodos[name]["p"] = np.array([
                    float(tokens[1]),
                    float(tokens[2])
                ])

        def procConduits(linea):
            tokens = linea.split()
            if len(tokens) < 7:
                return
            links[tokens[0]] = {
                "type": "conduit",
                "n1": tokens[1],
                "n2": tokens[2],
                "n": float(tokens[4]),
                "elev1": float(tokens[5]),
                "elev2": float(tokens[6]),
                }

        mode = None
        for line in lines:
            if startsWith(line, ';'):
                continue
            if len(line.strip()) == 0:
                continue
            if startsWith(line, '[JUNCTIONS]'):
                mode = procJunctions
            elif startsWith(line, '[STORAGE]'):
                mode = procStorage
            elif startsWith(line, '[COORDINATES]'):
                mode = procCoordinates
            elif startsWith(line, '[CONDUITS]'):
                mode = procConduits
            elif startsWith(line, '['):
                mode = None
            else:
                if mode is not None:
                    mode(line)

    return nodos, links


def extractFlowsOnAllConduits(swmmOutFileName, nodos, links):
    # Leer las coordenadas de los nodos

    conduits = []
    for name, link in links.items():
        if link["type"] != "conduit":
            continue
        try:
            conduits.append([name,
                nodos[link["n1"]]["p"][0],
                nodos[link["n1"]]["p"][1],
                nodos[link["n2"]]["p"][0],
                nodos[link["n2"]]["p"][1]])
        except:
            pass

    # Leer resultados
    outfile = swmmout.open(swmmOutFileName,'5.1.011')
    data = outfile.get_values('links', names=None, variables=['depth'])  #Se hace por separado para depth y para velocity

    fechas = []
    for dataline in data:
        fechas.append(dataline[0])
        # Each dataline is a time
        datadict = dict((str(dataitem[0]), dataitem[1]) for dataitem in dataline[1:])

        for conduit in conduits:
            name = conduit[0]
            conduit.append(abs(datadict[name]))

    with open("depth.txt", "w") as f:
        for conduit in conduits:
            line1 = [conduit[0]]
            line1.extend(conduit[5:])

            f.write('\t'.join([str(x) for x in line1]) + '\n')
            #f.write('\t'.join([str(x) for x in line2]) + '\n')
            f.write('\n')

            df={'date':fechas, 'depth': line1[1:]}
            df=pd.DataFrame(df)
            df.to_csv('Profundidades\\'+ str(line1[0]) +'.txt', sep=' ', header=('fechas', 'profundidad'))


if __name__ == '__main__':
    nodos, links = parseSWMMFile("model.inp")

    #extractDepthsOnAllNodes(swmmOutFileName, nodos, links)  Ver en extraer 2 si querés extraer en nodos hay otra función
    extractFlowsOnAllConduits(swmmOutFileName, nodos, links)