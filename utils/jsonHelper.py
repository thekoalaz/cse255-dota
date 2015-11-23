import json

def loadData(fileName):
    with open(fileName, 'r') as infile:
        data = json.load(infile)
    return data

def saveData(fileName, data):
    with open(fileName, 'wb') as outfile:
        json.dump(data, outfile)
