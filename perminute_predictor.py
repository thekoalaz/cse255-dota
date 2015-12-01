import numpy

from collections import namedtuple
from collections import defaultdict

import matplotlib
MATPLOT_EXTENSION = 'svg'
matplotlib.use(MATPLOT_EXTENSION)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

import utils.jsonHelper
import utils.logger

LOGGER = utils.logger.Logger("PerMinutePredictor")
LOGGER.setLevel(utils.logger.DEBUG)

class PerMinutePredictor(object):
    PerMinuteInfo = namedtuple("PerMinuteInfo", ("gpm", "xpm", "kpm", "duration"))

    def __init__(self, filename):
        LOGGER.horizontal_rule()
        LOGGER.debug("Reading data")
        self.data = utils.jsonHelper.loadData(filename)
        LOGGER.info("Done reading data of length: %d" % len(self.data))

        self.direInfos = []
        self.radiantINfos = []
        self.differenceInfos = []
        self.radiant_wins = 0

        self.count()
        self.stats()
        self.plot()
        LOGGER.horizontal_rule()
        LOGGER.info("GPM prediction")
        self.predict(lambda info: (1, info.duration, info.gpm))
        LOGGER.horizontal_rule()
        LOGGER.info("XPM prediction")
        self.predict(lambda info: (1, info.duration, info.xpm))
        LOGGER.horizontal_rule()
        LOGGER.info("KPM prediction")
        self.predict(lambda info: (1, info.duration, info.kpm))
        LOGGER.horizontal_rule()
        LOGGER.info("GPM, XPM prediction")
        self.predict(lambda info: (1, info.duration, info.gpm, info.xpm))
        LOGGER.horizontal_rule()
        LOGGER.info("GPM, KPM prediction")
        self.predict(lambda info: (1, info.duration, info.gpm, info.kpm))
        LOGGER.horizontal_rule()
        LOGGER.info("XPM, KPM prediction")
        self.predict(lambda info: (1, info.duration, info.xpm, info.kpm))
        LOGGER.horizontal_rule()
        LOGGER.info("GPM, XPM, KPM prediction")
        self.predict(lambda info: (1, info.duration, info.gpm, info.xpm, info.kpm))

    def count(self):
        LOGGER.horizontal_rule()
        LOGGER.debug("Counting...")
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
        LOGGER.debug("Done counting")

    def stats(self):
        radiant_win_rate = self.radiant_wins / self.total_games
        LOGGER.info("Radiant win rate: %f" % radiant_win_rate)
        durations = [info[0].duration for info in self.differenceInfos]
        LOGGER.info("Match duration: Mean: %f, StdDev: %f" % (numpy.mean(durations), numpy.std(durations)) )

        radiant_anomaly = [0, 0, 0]
        dire_anomaly = [0, 0, 0]
        for info in self.differenceInfos:
            if info[0].gpm < 0 and info[1]:
                radiant_anomaly[0] += 1
            elif info[0].gpm > 0 and not info[1]:
                dire_anomaly[0] += 1
            if info[0].xpm < 0 and info[1]:
                radiant_anomaly[1] += 1
            elif info[0].xpm > 0 and not info[1]:
                dire_anomaly[1] += 1
            if info[0].kpm < 0 and info[1]:
                radiant_anomaly[2] += 1
            elif info[0].kpm > 0 and not info[1]:
                dire_anomaly[2] += 1

        LOGGER.info("Radiant negative-gpm wins: %d, Percentage: %f" % (radiant_anomaly[0], radiant_anomaly[0] / self.total_games))
        LOGGER.info("Radiant negative-xpm wins: %d, Percentage: %f" % (radiant_anomaly[1], radiant_anomaly[1] / self.total_games))
        LOGGER.info("Radiant negative-kpm wins: %d, Percentage: %f" % (radiant_anomaly[2], radiant_anomaly[2] / self.total_games))

        LOGGER.info("Dire negative-gpm wins: %d, Percentage: %f" % (dire_anomaly[0], dire_anomaly[0] / self.total_games))
        LOGGER.info("Dire negative-xpm wins: %d, Percentage: %f" % (dire_anomaly[1], dire_anomaly[1] / self.total_games))
        LOGGER.info("Dire negative-kpm wins: %d, Percentage: %f" % (dire_anomaly[2], dire_anomaly[2] / self.total_games))

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


    def plot(self):
        def regression_plot(savename, xs, ys):
            def clamp(val):
                return max(min(val, 1), 0)

            Xs = [(x,) for x in xs]
            clf = linear_model.LogisticRegression().fit(Xs, ys)
            sparse_xs = numpy.array(xs)[0:-1:1000]
            sparse_ys = numpy.array(ys)[0:-1:1000]
            sparse_sorted_xs = numpy.sort(sparse_xs)
            sparse_log_ys = [1/ (1 + numpy.exp(-(clf.intercept_[0] + clf.coef_[0][0] * x))) for x in sparse_xs] 

            figure = plt.figure(figsize=(10,10))
            plt.subplots_adjust(top=0.85)
            plt.ylim(-0.1,1.1)
            plt.scatter(sparse_xs, sparse_ys, marker='.')
            plt.plot(sparse_sorted_xs, sparse_log_ys, linewidth = 1)
            plt.savefig(savename + '.' + MATPLOT_EXTENSION)
            figure.clf()
            

        LOGGER.horizontal_rule()
        LOGGER.debug("Plotting...")
        results = [1 if info[1] else 0 for info in self.differenceInfos]
        gpm_diffs = [info[0].gpm for info in self.differenceInfos]
        xpm_diffs = [info[0].xpm for info in self.differenceInfos]
        kpm_diffs = [info[0].kpm for info in self.differenceInfos]

        regression_plot('gpm', gpm_diffs, results)
        regression_plot('xpm', xpm_diffs, results)
        regression_plot('kpm', kpm_diffs, results)

        LOGGER.debug("Done plotting.")

    def predict(self, feature_func):
        def results(y_test, predictions):
            accuracy = 0
            for y,prediction in zip(y_test, predictions):
                if y == prediction:
                    accuracy += 1
            mse = numpy.mean((y_test - predictions) ** 2)
            LOGGER.info("MSE: %f" % mse)
            accuracy = accuracy / len(y_test)
            LOGGER.info("Accuracy: %f" % accuracy)

        train_size = round(len(self.differenceInfos) * 0.50)
        validation_size = round(len(self.differenceInfos) * 0.25)

        LOGGER.debug("Train size: %d" % train_size)
        LOGGER.debug("Validation size: %d" % validation_size)

        train = self.differenceInfos[:train_size]
        validation = self.differenceInfos[train_size:train_size+validation_size]
        test = self.differenceInfos[train_size+validation_size:]
       
        X_train = [feature_func(info[0]) for info in train]
        y_train = [1 if info[1] else 0 for info in train]

        X_validation = [feature_func(info[0]) for info in validation]
        y_validation = [1 if info[1] else 0 for info in validation]

        X_test = [feature_func(info[0]) for info in test]
        y_test = [1 if info[1] else 0 for info in test]

        LOGGER.horizontal_rule()
        LOGGER.info("Logistics regression")
        logistic_regressor = linear_model.LogisticRegression(C=1.0, fit_intercept=False)
        logistic_regressor.fit(X_train, y_train)
        LOGGER.info("Coefficients: %s" % logistic_regressor.coef_)

        predictions = logistic_regressor.predict(X_test)
        results(y_test, predictions)

        LOGGER.horizontal_rule()
        LOGGER.info("RandomForestClassifier")
        rfc = RandomForestClassifier(n_estimators = 50, n_jobs = 4)
        rfc.fit(X_train, y_train)
        LOGGER.info("Feature Importances: %s" % rfc.feature_importances_)

        predictions = rfc.predict(X_test)
        results(y_test, predictions)


if __name__ == '__main__':
    PerMinutePredictor("data/matches.json")
