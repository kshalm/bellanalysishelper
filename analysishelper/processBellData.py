import numpy as np
import copy
import yaml
import os
import base64
# import timetaggers as tt
# import coinclib as cl
try:
    import analysishelper.coinclib as cl
except ModuleNotFoundError:
    import coinclib as cl
try:
    import analysishelper.timetaggers as tt
except ModuleNotFoundError:
    import timetaggers as tt


def process_single_run(files, aggregate=True, findSync=False):
    print('value of findSync', findSync)
    errors = {}
    ttagDataStructure = np.dtype(
        [('ch', 'u1'), ('ttag', 'u8'), ('xfer', 'u2')])
    parties = {'alice', 'bob'}
    rawData = {}
    for p in parties:
        rawData[p] = np.fromfile(files[p], dtype=ttagDataStructure)
        rawData[p]['ch'] -= 1

    config = load_config_data(files['config'])
    timeTaggers = tt.TimeTaggers(config, offline=True)
    timeTaggers.configFile = files['config']

    if findSync:
        rawData, timeTaggers = find_ttag_offset(timeTaggers, rawData)

    print('Starting rollover detection')
    rollover = cl.check_for_timetagger_roll_over(rawData, timeTaggers.config)
    if rollover['err']:
        errors['rollover'] = True
        print('ERROR: Timetagger rollover detected')

    reducedData, err, info = analyze_data(rawData, timeTaggers.config)
    if err is not None:
        for k, v in err.items():
            errors[k] = v

    if 'output' in files:
        compressedData = cl.write_to_compressed_file(
            files['output'], reducedData, aggregate=aggregate)
    else:
        compressedData = cl.compress_binary_data(
            reducedData, aggregate=aggregate)

    counts = process_counts(reducedData)
    counts = counts.astype('int')
    # print(info)
    return counts, compressedData, errors, info


def find_ttag_offset(timeTaggers, rawData):
    nSyncs = 1000000
    dataSync = {}
    dataSync['alice'] = rawData['alice'][0:nSyncs]
    dataSync['bob'] = rawData['bob'][0:nSyncs]

    rawData['alice'] = rawData['alice'][nSyncs:-1]
    rawData['bob'] = rawData['bob'][nSyncs:-1]

    print('About to find offset')
    config, pkIdx = timeTaggers.find_sync_offset2(.2, rawData=dataSync)

    for p in pkIdx:
        config[p]['channelmap']['pkIdx'] = pkIdx[p]

    # print('')
    # print(config['analysis'])
    # print('')
    # timeTaggers.configFile = fConfig
    timeTaggers.config = config
    # timeTaggers.save_config_data()
    return rawData, timeTaggers


def process_multiple_data_runs(files, aggregate=True, findSync=False):
    parties = ['alice', 'bob']
    fAlice = files['alice']
    fBob = files['bob']
    fOut = files['output']
    if len(fAlice) != len(fBob):
        print('Alice and Bob need the same number of files')
        return None
    else:
        nFiles = len(fAlice)
    rawData = []

    chStatsAll = np.zeros((4, 4))
    errors = []

    for i in range(1, nFiles):
        print('Starting: ', fAlice[i])
        rawData = []
        filesForSingleRun = {}
        for key, fileArray in filesForSingleRun.items():
            f = fileArray[i]
            filesForSingleRun[key] = f
        # filesForSingleRun['alice'] = fAlice[i]
        # filesForSingleRun['bob'] = fBob[i]
        # filesForSingleRun['config'] = fConfig[i]
        # filesForSingleRun
        counts, compressedData, err = process_single_run(
            filesForSingleRun, aggregate=aggregate, findSync=findSync)
        errors.append(err)

        chStatsAll += counts.astype('int')
        print(chStatsAll.astype(int))
        print('')

    chStatsAll = chStatsAll.astype('int')
    return chStatsAll, errors


def check_for_detector_going_normal(ttags):
    pass


def trim_and_check_for_jumps(rawData, config):
    errors = None
    ttagOffset = config['analysis']['ttagOffset']
    abDelay = config['analysis']['pulseABDelay']
    syncTTagDiff = config['analysis']['syncTTagDiff']
    paramsCh = get_ch_settings(config)
    # Trim data
    print("Starting trim data procedure")
    trimmedData, err, heartbeat_bounds = cl.trim_data_to_heartbeat(
        rawData, ttagOffset, abDelay, syncTTagDiff, paramsCh)

    # # check for jumps
    print("Starting jump detection")
    # err, raw_data_list = cl.check_for_timetagger_jump(trimmedData, config)
    err = cl.check_for_timetagger_jump(trimmedData, config)

    if (err is None):
        return errors, trimmedData, heartbeat_bounds

    errors = {}
    errors['ttagJump'] = {}
    for k, v in err.items():
        errors['ttagJump'][k] = v
        if (k == 'err') and (v == True):
            print('ERROR: Timetagger jump detected.')

    '''
    If unique jumps are present, then split the data up at the jump points
    '''
    if err['err']:
        data_list_split_at_jumps = cl.split_data_at_jumps(trimmedData, err)
    else:
        return errors, trimmedData, heartbeat_bounds

    if (trimmedData is not None) & (data_list_split_at_jumps is not None):
        trimmed_data_list = []
        for data in data_list_split_at_jumps:
            # Trim each data segment. Here ttagoffset is 0 due to initial trimming
            td, err_trim = cl.trim_data(
                data, 0, abDelay, syncTTagDiff, paramsCh)
            trimmed_data_list.append(td)

        # concatenate back together each data segment.
        trimmedData = None
        for td in trimmed_data_list:
            if trimmedData is None:
                trimmedData = td
            else:
                for party in rawData:
                    trimmedData[party] = np.hstack(
                        (trimmedData[party], td[party]))

    return errors, trimmedData, heartbeat_bounds


def analyze_data(rawData, config):
    paramsCh = get_ch_settings(config)
    # paramsCh = get_ch_settings(config)
    divider = paramsCh['divider'] * 1.
    findPk = paramsCh['findPk']
    # usePockelsMask = config['pockelProp']['enable']
    abDelay = config['analysis']['pulseABDelay']

    errors, trimmedData, heartbeat_bounds = trim_and_check_for_jumps(
        rawData, config)

    paramsSingle = copy.deepcopy(paramsCh)
    # params = {'alice': {}, 'bob': {}}
    for detA in paramsCh['alice']['channels']['detector'].keys():
        for detB in paramsCh['bob']['channels']['detector'].keys():

            detChA = paramsCh['alice']['channels']['detector'][detA]
            detChB = paramsCh['bob']['channels']['detector'][detB]

            paramsSingle['alice']['channels']['detector'] = detChA
            paramsSingle['bob']['channels']['detector'] = detChB

            # settA1 = paramsCh['alice']['channels']['setting0']
            # settA2 = paramsCh['alice']['channels']['setting1']
            # settB1 = paramsCh['bob']['channels']['setting0']
            # settB2 = paramsCh['bob']['channels']['setting1']

            # paramsSingle['alice']['channels']['setting0'] = settA1
            # paramsSingle['alice']['channels']['setting1'] = settA2
            # paramsSingle['bob']['channels']['setting0'] = settB1
            # paramsSingle['bob']['channels']['setting1'] = settB2

    props = cl.calc_data_properties(
        trimmedData, paramsSingle, divider, findPk=findPk)

    reducedData = {}

    offset = {}

    offset['alice'] = 1 * abDelay
    offset['bob'] = 0
    pcStart = int(config['pockelProp']['start'])
    pcLength = int(config['pockelProp']['length']) + 1
    latency_info = {}
    latency_info['heartbeat_bounds'] = heartbeat_bounds
    #    latency_info['sync_bounds'] = heartbeat_bounds[1]
    for party in rawData.keys():
        reduced, latency_info_1_party =\
            cl.get_processed_data(trimmedData[party], props[party],
                                  offset[party], pcStart, pcLength)
        reducedData[party] = reduced
        latency_info[party] = latency_info_1_party
        latency_info[party]['sync_info'] = props[party]['info_array']

    return reducedData, errors, latency_info


def process_counts(data):
    SA = data['alice']['Setting']
    OA = data['alice']['Outcome']
    SB = data['bob']['Setting']
    OB = data['bob']['Outcome']

    sett = [1, 2]
    counts = []
    for s in sett:
        for j in sett:
            SAMask = SA == s
            SBMask = SB == j
            settingsMask = SAMask & SBMask
            OAMask = OA > 0
            OBMask = OB > 0
            # print(OA, OB)
            coinc = np.sum(settingsMask & OAMask & OBMask)
            singlesA = np.sum(settingsMask & OAMask)
            singlesB = np.sum(settingsMask & OBMask)
            nullOutcomes = np.sum(settingsMask & np.logical_not(
                OAMask) & np.logical_not(OBMask))
            res = [nullOutcomes, singlesA, singlesB, coinc]
            counts.append(res)

    counts = np.array(counts).astype(int)
    return counts


def get_ch_settings(config):
    params = {'alice': {}, 'bob': {}}
    for key in params:
        params[key]['radius'] = config[key]['coin_radius'] - 1
        params[key]['channels'] = config[key]['channelmap']
        params[key]['pkIdx'] = config[key]['channelmap']['pkIdx'] * 1.
        params[key]['heartbeat'] = config[key]['channelmap']['heartbeat']
    params['divider'] = config['DIVIDER'] * 1.
    params['measureViol'] = config['measureViol']
    params['findPk'] = config['analysis']['findPk']
    return (params)


def load_config_data(fname):
    config_fp = open(fname, 'r')
    config = yaml.safe_load(config_fp)
    config_fp.close()
    return config


def convert_str_to_bytes(strData):
    data = base64.b64decode(strData)
    return data


def convert_bytes_to_str(binData):
    strData = base64.b64encode(binData).decode('utf-8')
    return strData


'''
Loop through a directory and return a list of files for Alice and Bob for each
data run.
'''


def get_files_in_folder(path):
    files = [f for f in os.listdir(
        path) if os.path.isfile(os.path.join(path, f))]
    files.sort()
    files = [f for f in files if ".dat" in f]
    # files = [f for f in files if "2022_10_05_23_38_alice_suboptimal_test_run_two_1_60s.dat" in f]
    filesAlice = []
    filesBob = []
    filesConfig = []

    for i, s in enumerate(files):
        if 'alice' in s:
            s = path + '/' + s
            filesAlice.append(s)
            filesBob.append(s.replace('alice', 'bob'))
            # filesConfig.append(s.replace('alice', 'config').replace('.dat', '.yaml'))
            filesConfig.append('client.yaml')

    filesOut = {}
    filesOut['alice'] = filesAlice
    filesOut['bob'] = filesBob
    filesOut['config'] = filesConfig

    return filesOut


def main():
    path = '/Users/lks/Documents/BellData/2022/'
    date = '2022_10_06'

    files = get_files_in_folder(path + date)
    files['output'] = []

    for i in range(len(files['alice'])):
        fNameOut = files['bob'][i].replace('_bob', '').replace('.dat', '')
        fNameOut += '.bin.zip'
        fNameOut = fNameOut.split('/')[-1]
        fOut = path + 'processed/test2/' + date + '/' + fNameOut
        files['output'].append(fOut)
        # print(fOut)

    # fileA = ['2022_10_05_23_39_alice_suboptimal_test_run_two_2_60s.dat']
    # # fileA = ['2022_03_11_19_37_alice__1min_chunk_violation.dat',
    # #         '2022_03_11_19_38_alice__1min_chunk_violation_2.dat']

    # fileB = ['2022_10_05_23_39_bob_suboptimal_test_run_two_2_60s.dat']
    # fileB = ['2022_03_11_19_37_bob__1min_chunk_violation.dat',
    #         '2022_03_11_19_38_bob__1min_chunk_violation_2.dat']

    # fileA = ['2022_03_11_23_12_alice__80_percent_check_60s_.dat']
    # fileB = ['2022_03_11_23_12_bob__80_percent_check_60s_.dat']
    # fAlice = path+'/Alice/'+date+'/'+fileA
    # fBob = path+'/Bob/'+date+'/'+fileB

    # files ={}
    # files['alice'] = []
    # files['bob'] = []
    # files['output'] = []
    # for i in range(len(fileA)):
    #     fAlice = path+'/Alice/'+date+'/'+fileA[i]
    #     fBob = path+'/Bob/'+date+'/'+fileB[i]
    #     fNameOut = fileB[i].replace('_bob','').replace('.dat','')#.split['.'][0]
    #     fNameOut += '.bin.zip'
    #     fOut = path+'/processed/'+date+'/'+fNameOut
    #     print(fNameOut)
    #     files['alice'].append(fAlice)
    #     files['bob'].append(fBob)
    #     files['output'].append(fOut)

    # configFile = 'client.yaml'
    # config = load_config_data(configFile)

    filesSingle = {}
    indx = 11
    for key, fArray in files.items():
        try:
            filesSingle[key] = fArray[indx]
            # print(fArray[indx])
        except Exception:
            pass
    chStatsAll, compressedData, errors = process_single_run(
        filesSingle, aggregate=True)
    # chStatsAll = process_multiple_data_runs(files)

    # chStatsAll = np.array([[66074860,2384557,2309746,105999],
    #      [68059180,2381787,996158,761777],
    #      [68093218,1016778,2308621,750236],
    #      [69327491,1016925,997898,659596]])
    # chStatsAll[[1,2]] = chStatsAll[[2,1]]
    # chStatsAll[[0,3]] = chStatsAll[[3,0]]
    print(chStatsAll)
    print('')
    CH, CHn, ratio, pValue = cl.calc_violation(chStatsAll)


if __name__ == '__main__':
    main()
