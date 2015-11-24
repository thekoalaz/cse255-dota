import json
import random

def loadData(fileName):
    with open(fileName, 'r') as infile:
        data = json.load(infile)
    random.shuffle(data)
    return data

def saveData(fileName, data):
    with open(fileName, 'wb') as outfile:
        json.dump(data, outfile)
