import numpy as np
import time
import yaml
from multiprocessing import Queue  # , Process
import threading
import copy
# import coinclib as cl
# import analysis_library as al
# import singletimetagger as tt

try:
    import analysishelper.coinclib as cl
except ModuleNotFoundError:
    import coinclib as cl
try:
    import analysishelper.singletimetagger as tt
except ModuleNotFoundError:
    import singletimetagger as tt
try:
    import analysishelper.analysis_library as al
except ModuleNotFoundError:
    import analysis_library as al


class TimeTaggers():
    """
    Simple class to connect to the two time taggers, stream data, and find the
    delays/time tag offsets.
    """

    # configFile = 'client.yaml'):
    def __init__(self, config, offline=False, configFile=None):
        # self.configFile = configFile
        self.offline = offline
        self.config = config
        self.timeTaggers = {'alice': {'name': 'alice'}, 'bob': {'name': 'bob'}}
        self.configFile = configFile
        self.simple_init()

    def simple_init(self):

        if self.offline:
            if self.configFile is not None:
                self.load_config_data()
        else:
            ip1 = self.config['alice']['ip']
            port1 = self.config['alice']['port']

            ip2 = self.config['bob']['ip']
            port2 = self.config['bob']['port']

            if (ip1 == ip2) and (port1 == port2):
                self.singleServer = True
            else:
                self.singleServer = False

            det1 = self.config['alice']['channelmap']['detector']
            det2 = self.config['bob']['channelmap']['detector']

            self.timeTaggers['alice'] = tt.TimeTagger(ip1, port1, det1)
            self.timeTaggers['bob'] = tt.TimeTagger(ip2, port2, det2)

    def load_config_data(self):
        config_fp = open(self.configFile, 'r')
        config = yaml.safe_load(config_fp)
        config_fp.close()
        self.config = config

    def save_config_data(self):
        if self.config is not None:
            config_fp = open(self.configFile, 'w')
            yaml.dump(self.config, config_fp, default_flow_style=False)
            config_fp.close()
        else:
            print('Config not present')

    # def close(self):
    #     for key in ['alice', 'bob']:
    #         try:
    #             self.timeTaggers[key].close()
    #         except Exception:
    #             pass

    def reconnect(self):
        self.simple_init()
        # for key in ['alice','bob']:
        #     self.timeTaggers[key].connect(self.config[key]['ip'], self.config[key]['port'])

    def log_start_twottag(self, filestr):
        files = {}
        for key in self.timeTaggers.keys():
            filename = self.timeTaggers[key].start_logging_to_file(
                key + '_' + filestr)
            files[key] = filename
        return (files)

    def log_stop_twottag(self):
        files = {}
        for key in self.timeTaggers.keys():
            fn = self.timeTaggers[key].stop_logging_to_file()
            files[key] = fn
        return (files)

    def close(self):
        for key in self.timeTaggers.keys():
            self.timeTaggers[key].close()

    def get_stats(self, q='', dt=0.5):
        tableDict = {}
        keys = ['alice', 'bob']
        q = {}
        t = {}

        for key in keys:
            q[key] = Queue()
            ttagger = self.timeTaggers[key]
            t[key] = threading.Thread(
                target=self._fetch_stats, args=(ttagger, dt, q[key]))

        for key in keys:
            t[key].start()

        for key in keys:
            t[key].join(3 * dt)

        for key in keys:
            counts = q[key].get(timeout=dt * 5)
            tableDict[key] = counts

        # for key in self.timeTaggers.keys():
        #     counts = self.timeTaggers[key].get_stats(dt)
        #     tableDict[key] = counts

        if self.singleServer:
            tableDict['bob'] = tableDict['alice']

        return (tableDict)

    def _fetch_stats(self, ttagger, dt, q):
        counts = ttagger.get_stats(dt)
        q.put(counts)

    def get_data(self, ttager, dt, q=''):
        # t1 = time.time()
        data = ttager.stream_server(dt)
        # print(data)
        # t2 = time.time()
        # print('Fetch took', t2-t1, 'ms, dt:',dt)
        q.put(data)
        return data

    def fetch_data(self, intTime):
        q = {'alice': Queue(), 'bob': Queue()}
        t = {}
        for key in q:
            t[key] = threading.Thread(target=self.get_data, args=(
                self.timeTaggers[key], intTime, q[key]))

        for key in q:
            t[key].start()

        for key in q:
            t[key].join(3 * intTime)

        rawData = {}
        for key in q:
            # try:
            data = q[key].get(timeout=2.)
            data['ch'] = data['ch'] - 1
            rawData[key] = data

            # except Exception:
            #     rawData[key] = None
        return (rawData)

    def get_ch_settings(self):
        params = {'alice': {}, 'bob': {}}
        for key in params:
            params[key]['radius'] = self.config[key]['coin_radius'] - 1
            params[key]['channels'] = self.config[key]['channelmap']
            params[key]['pkIdx'] = self.config[key]['channelmap']['pkIdx'] * 1.
            # for ch in params[key]['channels']:
            #     params[key]['channels'][ch] -= 1
        # params['analysis'] = self.config['pockelProp']['analysis']
        params['divider'] = self.config['DIVIDER'] * 1.
        params['measureViol'] = self.config['measureViol']
        params['findPk'] = self.config['analysis']['findPk']
        return (params)

    def find_pc_turn_on_off(self, intTime):

        rawData = self.fetch_data(intTime)
        # rawData['bob'] = rawData['alice']

        # Next, with the data, trim it and find the offsets
        paramsCh = self.get_ch_settings()

        paramsS = copy.deepcopy(paramsCh)
        # only choose one of the detectors from Alice and Bob to start
        aliceDetCh = self.config['alignchannel']['alice']
        bobDetCh = self.config['alignchannel']['bob']

        paramsS['alice']['channels']['detector'] = aliceDetCh
        paramsS['bob']['channels']['detector'] = bobDetCh

        pcSS = al.calc_pc_on_off_slots(rawData, paramsS)
        # print('finished', pcSS)

        config_fp = open(self.configFile, 'r+')
        config = yaml.load(config_fp)
        config_fp.close()

        config['pockelProp']['analysis']['alice']['start'] = int(
            pcSS['alice'][0])
        config['pockelProp']['analysis']['alice']['stop'] = int(
            pcSS['alice'][1])
        config['pockelProp']['analysis']['bob']['start'] = int(pcSS['bob'][0])
        config['pockelProp']['analysis']['bob']['stop'] = int(pcSS['bob'][1])
        config['pockelProp']['start'] = int(pcSS['alice'][0])

        config_fp = open(self.configFile, 'w')
        yaml.dump(config, config_fp, default_flow_style=False)
        config_fp.close()
        return (pcSS)

    def find_sync_offset2(self, intTime, rawData=None):

        if rawData is None:
            rawData = self.fetch_data(intTime)

        # Next, with the data, trim it and find the offsets
        paramsCh = self.get_ch_settings()

        paramsS = copy.deepcopy(paramsCh)
        # only choose one of the detectors from Alice and Bob to start
        # aliceDetCh = self.config['alignchannel']['alice']
        # bobDetCh = self.config['alignchannel']['bob']
        # paramsS['alice']['channels']['detector'] = aliceDetCh
        # paramsS['bob']['channels']['detector'] = bobDetCh
        detectorName = 'V'
        paramsS['alice']['channels']['detector'] = paramsCh['alice']['channels']['detector'][detectorName]
        paramsS['bob']['channels']['detector'] = paramsCh['bob']['channels']['detector'][detectorName]
        # print(paramsCh)
        # print(paramsCh['alice']['channels']['detector'])
        divider = paramsCh['divider'] * 1.

        # print('starting')
        offsets = cl.calc_offset(rawData, paramsS, divider)
        abDelay = offsets['abDelay']
        offsetLaserPulse = offsets['offsetLaserPulse']
        ttagOffset = offsets['ttagOffset']
        syncTTagDiff = offsets['syncTTagDiff']
        pkIdx = offsets['pkIdx']

        # abDelay, offsetLaserPulse, ttagOffset, syncTTagDiff = cl.calc_offset(rawData, paramsS, divider)
        # print('finished finding offset')
        self.ttagOffset = ttagOffset
        self.offsetLaserPulse = offsetLaserPulse
        self.abDelay = abDelay

        self.config['analysis']['ttagOffset'] = int(ttagOffset)
        self.config['analysis']['pulseABDelay'] = int(offsetLaserPulse)
        self.config['analysis']['abDelay'] = int(abDelay)
        self.config['analysis']['syncTTagDiff'] = int(syncTTagDiff)

        # config_fp = open(self.configFile,'w')
        # yaml.dump(self.config, config_fp, default_flow_style=False)
        # config_fp.close()
        return self.config, pkIdx

    def update(self, dt='default'):
        # self.load_config_data()
        if dt == 'default':
            dt = self.config['INT_TIME']

        # self.ttagOffset = self.config['analysis']['ttagOffset']
        # self.abDelay = self.config['analysis']['abDelay']
        # usePockelsMask = self.config['pockelProp']['enable']

        margin = min(0.4 * dt, 0.2)
        margin = max(margin, 0.075)
        timeToFetch = dt + margin
        time.sleep(dt)
        rawData = self.fetch_data(timeToFetch)

        counts, params = self.analyze_data(rawData, dt)
        return (counts, params)

    def analyze_data(self, rawData, dt=None):

        self.ttagOffset = self.config['analysis']['ttagOffset']
        self.abDelay = self.config['analysis']['abDelay']
        self.syncTTagDiff = self.config['analysis']['syncTTagDiff']
        usePockelsMask = self.config['pockelProp']['enable']

        paramsCh = self.get_ch_settings()
        divider = paramsCh['divider'] * 1.
        findPk = paramsCh['findPk']
        isTrim = True
        # print(rawData)

        trimmedData, err = cl.trim_data(
            rawData, self.ttagOffset, self.abDelay, self.syncTTagDiff, paramsCh, dt=dt)
        isTrim = not err

        paramsSingle = copy.deepcopy(paramsCh)
        counts = {}
        counts['isTrim'] = int(isTrim)
        params = {'alice': {}, 'bob': {}}
        reducedDataSet = {}
        for detA in paramsCh['alice']['channels']['detector'].keys():
            for detB in paramsCh['bob']['channels']['detector'].keys():
                detKey = detA + detB
                detChA = paramsCh['alice']['channels']['detector'][detA]
                detChB = paramsCh['bob']['channels']['detector'][detB]
                paramsSingle['alice']['channels']['detector'] = detChA
                paramsSingle['bob']['channels']['detector'] = detChB

                try:
                    results = cl.calc_data_properties(
                        trimmedData, paramsSingle, divider, findPk=findPk)
                except Exception:
                    print('failed data properties')
                    results = {}
                    results['alice'] = None
                    results['bob'] = None

                detectionMask = self.get_window_mask(trimmedData, results)

                detectionDarkMask = self.get_darks_mask(trimmedData, results)

                pockelsMask, paramsPockels = self.get_pockels_mask(
                    trimmedData, results)

                detMask = [detectionMask]
                coincAndSingles = self.compute_coinc(
                    trimmedData, results, detMask)

                maskPC = [detectionMask, pockelsMask]
                coincAndSinglesPC = self.compute_coinc(
                    trimmedData, results, maskPC)

                maskDark = [detectionDarkMask]
                coincAndSinglesDark = self.compute_coinc(
                    trimmedData, results, maskDark)

                chStats, reducedData = self.compute_stats(trimmedData, results)
                # if isTrim:
                #     print(chStats, coincAndSingles)
                reducedDataSet[detKey] = reducedData
                counts[detKey + '_chStats'] = chStats
                counts[detKey] = coincAndSingles
                counts[detKey]['alice'] = detA
                counts[detKey]['bob'] = detB

                counts[detKey + '_PC'] = coincAndSinglesPC
                counts[detKey + '_Background'] = coincAndSinglesDark

                params['alice'][detA] = results['alice']
                params['bob'][detB] = results['bob']

                params['alice']['plotPA'] = {}
                params['bob']['plotPB'] = {}
                if paramsPockels is not None:
                    params['alice']['plotPA']['shadedRegion'] = paramsPockels['alice']
                    params['bob']['plotPB']['shadedRegion'] = paramsPockels['bob']
                else:
                    params['alice']['plotPA']['shadedRegion'] = None
                    params['bob']['plotPB']['shadedRegion'] = None
        # print(paramsPockels)
        return (counts, params, reducedDataSet)

    def get_pockels_mask(self, data, props):
        abDelay = self.config['analysis']['pulseABDelay']
        pockelsMask = {}
        offset = {}
        params = {}

        offset['alice'] = 1 * abDelay
        offset['bob'] = 0
        pcStart = int(self.config['pockelProp']['start'])
        pcLength = int(self.config['pockelProp']['length']) + 1

        for party in data.keys():
            # cl.get_processed_data(data[party], props[party], offset[party], pcStart, pcLength)

            pcMask, p = cl.get_pockels_mask(data[party], props[party],
                                            offset[party], pcStart, pcLength)
            pockelsMask[party] = pcMask
            params[party] = p

        return pockelsMask, params

    def get_window_mask(self, data, props):
        mask = {}
        for party in data:
            m = cl.get_window_mask(data[party], props[party])
            mask[party] = m

        return mask

    def get_darks_mask(self, data, props):
        mask = {}
        for party in data:
            m = cl.get_darks_mask(data[party], props[party])
            mask[party] = m
        return mask

    def get_settings_mask(self, data, props):
        mask = {}
        totalSettings = []
        sett = [1, 2]
        settings = []

        party = ['alice', 'bob']
        nSyncs = min(len(props[party[0]]['settingsSync']),
                     len(props[party[1]]['settingsSync']))
        settingSync = []
        for s1 in sett:
            for s2 in sett:
                sA = (props[party[0]]['settingsSync'] == s1)
                sB = (props[party[1]]['settingsSync'] == s2)
                sAsB = (sA[0:nSyncs] & sB[0:nSyncs])
                settingSync.append(sAsB)
                totalSettings.append(np.sum(sAsB))

        for p in party:
            mask[p] = []
            for s in settingSync:
                sAsBSync = s[props[p]['syncArray']]
                mask[p].append(sAsBSync)

        return mask, totalSettings

    def compute_stats(self, data, props):
        for party in data:
            if data[party] is None:
                return np.zeros((4, 4))

        detectionMask = self.get_window_mask(data, props)

        pockelsMask, paramsPockels = self.get_pockels_mask(data, props)

        settings, totalSettings = self.get_settings_mask(data, props)

        masksForReduced = [detectionMask, pockelsMask]
        reducedData, laserPeriod = self.get_reduced_data(
            data, props, masksForReduced)
        pcLength = int(self.config['pockelProp']['length']) + 1
        nPulses = 2 * pcLength

        res = []
        # for s in settings:
        party = []
        for p in data:
            party.append(p)

        for i in range(4):
            settingsMask = {}
            for p in party:
                settingsMask[p] = settings[p][i]
            mask = [detectionMask, pockelsMask, settingsMask]
            counts = self.compute_coinc(data, props, mask, nPulses=nPulses)
            trials = totalSettings[i]
            nullOutcomes = trials - \
                (counts['sAlice'] + counts['sBob'] + counts['coinc'])
            res += [nullOutcomes, counts['sAlice'],
                    counts['sBob'], counts['coinc']]

        res = np.array(res).reshape((4, 4))
        return res, reducedData

    def compute_coinc(self, data, props, masks, nPulses=None):
        countsDataDict, laserPeriod = self.get_reduced_data(data, props, masks)
        chData = []
        for party in countsDataDict:
            chData.append(countsDataDict[party])
        period = np.max(laserPeriod)
        if nPulses is None:
            radius = (period - 2) / 2
        else:
            radius = nPulses * (period - 2) / 2

        singlesA, singlesB, coinc = cl.find_coincidences(
            chData[0], chData[1], radius)
        counts = {'sAlice': singlesA,
                  'sBob': singlesB,
                  'coinc': coinc}
        return counts

    def get_reduced_data(self, data, props, masks):
        countsData = {}
        laserPeriod = []

        for party in data:
            totalMask = np.array([True] * len(data[party]))
            for m in masks:
                totalMask = totalMask & m[party]
            countsData[party] = data[party]['ttag'][totalMask]
            laserPeriod.append(props[party]['laserPeriod'])

        return countsData, laserPeriod


if __name__ == '__main__':
    pass
