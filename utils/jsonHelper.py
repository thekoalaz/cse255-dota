import json
import random

import utils.logger
from utils.enums import GameMode

LOGGER = utils.logger.Logger("utils")
GAME_MODE_FILTER = [GameMode.ALL_PICK, GameMode.ALL_RANDOM, GameMode.CAPTAINS_DRAFT,
                    GameMode.SINGLE_DRAFT, GameMode.RANDOM_DRAFT, GameMode.CAPTAINS_MODE,
                    GameMode.ALL_DRAFT]

def loadData(fileName):
    with open(fileName, 'r') as infile:
        data = json.load(infile)
    before_filter_len = len(data)
    data = filter(data)
    LOGGER.debug("Filtered %d matches" % (before_filter_len - len(data)))
    random.shuffle(data)
    return data

def saveData(fileName, data):
    with open(fileName, 'wb') as outfile:
        json.dump(data, outfile)

def filter(data):
    filtered_data = []
    for datum in data:
        if datum['duration'] == 0:
            continue
        try:
            if GameMode(datum['game_mode']) not in GAME_MODE_FILTER:
                continue
        except ValueError:
            LOGGER.error("Unknown game mode encountered: %d" % datum['game_mode'])
        filtered_data.append(datum)

    return filtered_data
