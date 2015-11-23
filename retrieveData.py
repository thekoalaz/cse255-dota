import time
import jsonHelper
import dota2api

API_KEY = '8F8A5B69F2D638628736590086E7541B'
#API_KEY = '3218DD1190D7857B95363707E0C6A245'
def saveMatchIDs(start_id, fileName):
    api = dota2api.Initialise(API_KEY)
    matchIDs = [];
    while True:
        start_id = addMatchIDs(api,start_id,matchIDs)
        start_id -= 1 #avoid repeats
        if start_id == -1:
            break
    jsonHelper.saveData(fileName, matchIDs)
def addMatchIDs(api, start_id, matchIDs):
    history = api.get_match_history(skill = 3, start_at_match_id = start_id, matches_requested=100)
    matches = history['matches']
    ids = [m['match_id'] for m in matches]
    matchIDs.extend(ids)
    if len(ids) > 0:
        return ids[-1]
    else:
        return 0

def saveMatches(matchIDsFileName):
    api = dota2api.Initialise(API_KEY)

    matchIDs = jsonHelper.loadData(matchIDsFileName)

    matches = []
    i = 0
    count = 0
    for matchID in matchIDs:
        if(i > 109):
            addMatch(api,matchID,matches)
        count += 1
        if count >= 500:
            if(i > 109):
                jsonHelper.saveData('matches_' + str(i) + '.json', matches)
            matches = []
            count = 0
            print(i)
            i += 1

def addMatch(api, matchID, matches):
    match = api.get_match_details(match_id=matchID)
    matches.append(match)

def runMatchIDScan():
    for i in range(144,1000):
        saveMatchIDs(None, 'matchIDs_' + str(i) + '.json')
        if i > 0:
            oldData = jsonHelper.loadData('matchIDs_' + str(i-1) + '.json')
            newData = jsonHelper.loadData('matchIDs_' + str(i) + '.json')
            print(len(oldData + newData))
            print(len(set(oldData + newData)))
        print(i)
        time.sleep(900)

def combineMatchIDs(fileName):
    matchIDsSet = set()
    for i in range(0,144):
        matchIDs = jsonHelper.loadData('matchIDs_' + str(i) + '.json')
        matchIDsSet |= set(matchIDs)
    jsonHelper.saveData(fileName, list(matchIDsSet))

def combineMatches(fileName):
    matchSet = []
    for i in range(0,124):
        matches = jsonHelper.loadData('matchesReduced_' + str(i) + '.json')
        matchSet.extend(matches)
        print(i)
    jsonHelper.saveData(fileName, matchSet)

def reduceMatches():
    for i in range(0,124):
        matches = jsonHelper.loadData('matches_' + str(i) + '.json')
        matchesReduced = [reduceMatch(match) for match in matches]
        jsonHelper.saveData('matchesReduced_' + str(i) + '.json', matchesReduced)
        print(i)

def reduceMatch(match):
    for player in match['players']:
        player.pop('account_id', 0)
        player.pop('ability_upgrades', 0)
        player.pop('item_0', 0)
        player.pop('item_0_name', 0)
        player.pop('item_1', 0)
        player.pop('item_1_name', 0)
        player.pop('item_2', 0)
        player.pop('item_2_name', 0)
        player.pop('item_3', 0)
        player.pop('item_3_name', 0)
        player.pop('item_4', 0)
        player.pop('item_4_name', 0)
        player.pop('item_5', 0)
        player.pop('item_5_name', 0)
        player.pop('leaver_status', 0)
        player.pop('additional_units', 0)
        player.pop('tower_damage', 0)
        player.pop('hero_healing', 0)

    match.pop('barracks_status_dire', 0)
    match.pop('barracks_status_radiant', 0)
    match.pop('cluster', 0)
    match.pop('cluster_name', 0)
    match.pop('engine', 0)
    match.pop('human_players', 0)
    match.pop('leagueid', 0)
    match.pop('lobby_type', 0)
    match.pop('lobby_name', 0)
    match.pop('match_seq_num', 0)
    match.pop('negative_votes', 0)
    match.pop('positive_votes', 0)
    match.pop('start_time', 0)
    match.pop('season', 0)
    match.pop('tower_status_dire', 0)
    match.pop('tower_status_radiant', 0)

    match.pop('radiant_name', 0)
    match.pop('radiant_logo', 0)
    match.pop('radiant_team_complete', 0)
    match.pop('dire_name', 0)
    match.pop('dire_logo', 0)
    match.pop('dire_team_complete', 0)

    return match




#runMatchIDScan()
#combineMatchIDs('matchIDs.json')
#saveMatches('matchIDs.json')
#reduceMatches()
#combineMatches('matches.json')