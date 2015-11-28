﻿import numpy

from collections import namedtuple
from collections import defaultdict

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

import utils.jsonHelper
import utils.logger

LOGGER = utils.logger.Logger("PerMinutePredictor")
LOGGER.setLevel(utils.logger.DEBUG)

class PerMinutePredictor(object):
    PerMinuteInfo = namedtuple("PerMinuteInfo", ("gpm", "xpm", "kpm", "duration"))

    def __init__(self, filename):
        LOGGER.debug("Reading data")
        self.data = utils.jsonHelper.loadData(filename)
        LOGGER.debug("Done reading data of length: %d" % len(self.data))

        self.direInfos = []
        self.radiantINfos = []
        self.differenceInfos = []
        self.radiant_wins = 0

        self.count()
        self.stats()
        self.predict()

    def count(self):
        for datum in self.data:
            duration_in_min = datum['duration'] / 60
            if duration_in_min == 0:
                LOGGER.error("Duration was 0.")
                continue

            if datum['radiant_win']:
                self.radiant_wins += 1

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
            direInfo = self.PerMinuteInfo(gpm = dire_info[0], xpm = dire_info[1], kpm = dire_info[2], duration = duration_in_min)
            radiantInfo = self.PerMinuteInfo(gpm = radiant_info[0], xpm = radiant_info[1], kpm = radiant_info[2], duration = duration_in_min)
            self.direInfos.append(direInfo)
            self.radiantINfos.append(radiantInfo)

            differenceInfo = self.PerMinuteInfo(gpm = radiantInfo.gpm - direInfo.gpm,
                                           xpm = radiantInfo.xpm - direInfo.xpm,
                                           kpm = radiantInfo.kpm - direInfo.kpm,
                                           duration = duration_in_min)
            self.differenceInfos.append((differenceInfo, datum['radiant_win']))

        self.total_games = len(self.data)

    def stats(self):
        radiant_win_rate = self.radiant_wins / self.total_games
        LOGGER.info("Radiant win rate: %f" % radiant_win_rate)
        durations = [info[0].duration for info in self.differenceInfos]
        LOGGER.info("Match duration: Mean: %f, StdDev: %f" % (numpy.mean(durations), numpy.std(durations)) )

        radiant_gpms = [info.gpm for info in self.radiantINfos]
        LOGGER.info("Radiant gpm: Mean: %f, StdDev:%f" % (numpy.mean(radiant_gpms), numpy.std(radiant_gpms)) )
        dire_gpms = [info.gpm for info in self.direInfos]
        LOGGER.info("Dire gpm: Mean: %f, StdDev:%f" % (numpy.mean(dire_gpms), numpy.std(dire_gpms)) )
        radiant_xpms = [info.xpm for info in self.radiantINfos]
        LOGGER.info("Radiant xpm: Mean: %f, StdDev:%f" % (numpy.mean(radiant_xpms), numpy.std(radiant_xpms)) )
        dire_xpms = [info.xpm for info in self.direInfos]
        LOGGER.info("Dire xpm: Mean: %f, StdDev:%f" % (numpy.mean(dire_xpms), numpy.std(dire_xpms)) )
        radiant_kpms = [info.kpm for info in self.radiantINfos]
        LOGGER.info("Radiant kpm: Mean: %f, StdDev:%f" % (numpy.mean(radiant_kpms), numpy.std(radiant_kpms)) )
        dire_kpms = [info.kpm for info in self.direInfos]
        LOGGER.info("Dire kpm: Mean: %f, StdDev:%f" % (numpy.mean(dire_kpms), numpy.std(dire_kpms)) )

    def predict(self):
        def results(y_test, predictions):
            accuracy = 0
            for y,prediction in zip(y_test, predictions):
                if y == prediction:
                    accuracy += 1
            mse = numpy.mean((y_test - predictions) ** 2)
            LOGGER.info("MSE: %f" % mse)
            accuracy = 1 / len(y_test)
            LOGGER.info("Accuracy: %f" % accuracy)

        train_size = round(len(self.differenceInfos) * 0.8)
        validation_size = round(len(self.differenceInfos) * 0.1)

        LOGGER.debug("Train size: %d" % train_size)
        LOGGER.debug("Validation size: %d" % validation_size)

        train = self.differenceInfos[:train_size]
        validation = self.differenceInfos[train_size:train_size+validation_size]
        test = self.differenceInfos[train_size+validation_size:]
       
        X_train = [(1, info[0].duration, info[0].gpm, info[0].xpm, info[0].kpm) for info in train]
        y_train = [1 if info[1] else 0 for info in train]

        X_validation = [(1, info[0].duration, info[0].gpm, info[0].xpm, info[0].kpm) for info in validation]
        y_validation = [1 if info[1] else 0 for info in validation]

        X_test = [(1, info[0].duration, info[0].gpm, info[0].xpm, info[0].kpm) for info in test]
        y_test = [1 if info[1] else 0 for info in test]

        logistic_regressor = linear_model.LogisticRegression(C=1.0, fit_intercept=False)
        logistic_regressor.fit(X_train, y_train)

        predictions = logistic_regressor.predict(X_test)
        LOGGER.horizontal_rule()
        LOGGER.info("Logistics regression")
        results(y_test, predictions)

        rfc = RandomForestClassifier(n_estimators = 50, n_jobs = 4)
        rfc.fit(X_train, y_train)

        predictions = rfc.predict(X_test)
        LOGGER.horizontal_rule()
        LOGGER.info("RandomForestClassifier")
        results(y_test, predictions)


if __name__ == '__main__':
    PerMinutePredictor("data/matches.json")
