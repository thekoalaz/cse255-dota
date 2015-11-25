import numpy

from collections import namedtuple
from collections import defaultdict

import utils.jsonHelper
import utils.logger

LOGGER = utils.logger.Logger("winrate")
LOGGER.setLevel(utils.logger.DEBUG)

def run(data):
    radiant_wins = 0
    side = {}
    PerMinuteInfo = namedtuple("PerMinuteInfo", ("gpm", "xpm", "kpm", "duration"))

    direInfos = []
    radiantInfos = []

    differenceInfos = []

    for datum in data:
        duration_in_min = datum['duration'] / 60
        if duration_in_min == 0:
            LOGGER.error("Duration was 0.")
            continue

        if datum['radiant_win']:
            radiant_wins += 1

        dire_info = [0, 0, 0]
        radiant_info = [0, 0, 0]
        for player in datum['players']:
            if player['side'] == 1:
                dire_info[0] += player['gold_per_min']
                dire_info[1] += player['xp_per_min']
                dire_info[2] += player['kills'] / duration_in_min
            else:
                radiant_info[0] += player['gold_per_min']
                radiant_info[1] += player['xp_per_min']
                radiant_info[2] += player['kills'] / duration_in_min
        direInfo = PerMinuteInfo(gpm = dire_info[0], xpm = dire_info[1], kpm = dire_info[2], duration = duration_in_min)
        radiantInfo = PerMinuteInfo(gpm = radiant_info[0], xpm = radiant_info[1], kpm = radiant_info[2], duration = duration_in_min)
        direInfos.append(direInfo)
        radiantInfos.append(radiantInfo)

        differenceInfo = PerMinuteInfo(gpm = radiantInfo.gpm - direInfo.gpm,
                                       xpm = radiantInfo.xpm - direInfo.xpm,
                                       kpm = radiantInfo.kpm - direInfo.kpm,
                                       duration = duration_in_min)
        differenceInfos.append((differenceInfo, datum['radiant_win']))


    total_games = len(data)
    radiant_win_rate = radiant_wins / total_games
    LOGGER.info("Radiant win rate: %f" % radiant_win_rate)
    durations = [info[0].duration for info in differenceInfos]
    LOGGER.info("Match duration: Mean: %f, StdDev: %f" % (numpy.mean(durations), numpy.std(durations)) )

    radiant_gpms = [info.gpm for info in radiantInfos]
    LOGGER.info("Radiant gpm: Mean: %f, StdDev:%f" % (numpy.mean(radiant_gpms), numpy.std(radiant_gpms)) )
    dire_gpms = [info.gpm for info in direInfos]
    LOGGER.info("Dire gpm: Mean: %f, StdDev:%f" % (numpy.mean(dire_gpms), numpy.std(dire_gpms)) )
    radiant_xpms = [info.xpm for info in radiantInfos]
    LOGGER.info("Radiant xpm: Mean: %f, StdDev:%f" % (numpy.mean(radiant_xpms), numpy.std(radiant_xpms)) )
    dire_xpms = [info.xpm for info in direInfos]
    LOGGER.info("Dire xpm: Mean: %f, StdDev:%f" % (numpy.mean(dire_xpms), numpy.std(dire_xpms)) )
    radiant_kpms = [info.kpm for info in radiantInfos]
    LOGGER.info("Radiant kpm: Mean: %f, StdDev:%f" % (numpy.mean(radiant_kpms), numpy.std(radiant_kpms)) )
    dire_kpms = [info.kpm for info in direInfos]
    LOGGER.info("Dire kpm: Mean: %f, StdDev:%f" % (numpy.mean(dire_kpms), numpy.std(dire_kpms)) )

    return radiantInfos, direInfos, differenceInfos

if __name__ == '__main__':
    LOGGER.debug("Reading data")
    data = utils.jsonHelper.loadData("data/matches.json")
    LOGGER.debug("Done reading data of length: %d" % len(data))
    run(data)
