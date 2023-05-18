import numpy as np
import zlib
from numba import jit
import json
import time

import scipy.signal
from scipy.stats import binom
from copy import deepcopy

TTAGERRESOLUTION = 78.125E-12


# @jit(nopython=True, cache=True)
def find_coincidences_cw(data, ch1, ch2, radius):
    '''
    The algorithm takses a sorted list of timetags and first separates out
    the timetags corresponding to the two channel numbers of interest. The
    radius sets the coincidence window in terms the number of the number
    timetag increments. A radius of x opens a coincidence window of 2x+1
    time bins. Each time bin is 78 ps in length.

    For each timetag in ch1, search in ch2 to see if a timetag has arrived in
    a given coincidence window. This alorithm is faster than the previous
    naieve alorithm as it steps through both lists at the same time. As soon as
    a timetag in ch2 is encountered that has a timestamp larger than the timetag
    in ch1, then we move to the next timetag in ch1 and keep searching. This
    keeps the number of iterations down. The algorithm should scale closer
    to linearly as opposed to quadratically?

    Returns the singles for each channel as well as the coincidences.
    '''
    firstTTAG = data['ttag'][0]

    ch1Bool = data['ch'] == ch1
    ch2Bool = data['ch'] == ch2

    ch1Data = data['ttag'][ch1Bool] - firstTTAG
    ch2Data = data['ttag'][ch2Bool] - firstTTAG

    results = find_coincidences(ch1Data, ch2Data, radius)

    return results


@jit(nopython=True, cache=True)
def find_coincidences(ch1Data, ch2Data, radius):
    '''
    The algorithm takses a sorted list of timetags and first separates out
    the timetags corresponding to the two channel numbers of interest. The
    radius sets the coincidence window in terms the number of the number
    timetag increments. A radius of x opens a coincidence window of 2x+1
    time bins. Each time bin is 78 ps in length.

    For each timetag in ch1, search in ch2 to see if a timetag has arrived in
    a given coincidence window. This alorithm is faster than the previous
    naieve alorithm as it steps through both lists at the same time. As soon as
    a timetag in ch2 is encountered that has a timestamp larger than the timetag
    in ch1, then we move to the next timetag in ch1 and keep searching. This
    keeps the number of iterations down. The algorithm should scale closer
    to linearly as opposed to quadratically?

    Returns the singles for each channel as well as the coincidences.
    '''
    # print(data)
    # firstTTAG = data['ttag'][0]

    # ch1Bool = data['ch'] == ch1
    # ch2Bool = data['ch'] == ch2

    # ch1Data = data['ttag'][ch1Bool] - firstTTAG
    # ch2Data = data['ttag'][ch2Bool] - firstTTAG

    singlesCh1 = len(ch1Data)
    singlesCh2 = len(ch2Data)

    coincidences = 0
    ch2Indx = 0
    for i in range(singlesCh1):
        ch1TTAG = ch1Data[i]
        upper = ch1TTAG + radius
        lower = ch1TTAG - radius
        search = True

        while search:
            if ch2Indx >= singlesCh2:
                search = False
                break

            ch2TTAG = ch2Data[ch2Indx]
            if ch2TTAG < lower:
                ch2Indx += 1
            if ch2TTAG > upper:
                search = False
            if (ch2TTAG >= lower) & (ch2TTAG <= upper):
                search = False
                coincidences += 1
                ch2Indx += 1

    return [singlesCh1, singlesCh2, coincidences]


def trim_data(data, ttagOffset, abDelay, syncTTagDiff, params, dt=None):
    err = False
    trimmedData = {'alice': {}, 'bob': {}}

    validData = True
    for key in data:
        if data[key] is None:
            validData = False

    if validData == False:
        for key in data:
            tdata = trim_data_single_party(data[key], dt=dt)
            trimmedData[key] = tdata
        err = True
        return trimmedData, err

    aData = data['alice']['ttag']
    # event number for all bob tags, corrected for delay
    bData = data['bob']['ttag'] + ttagOffset

    startTTag = max(aData[0], bData[0])
    # print('startTTag', startTTag)
    # abDelay = -1
    # print('abDelay', abDelay, 'START', startTTag, aData[0], bData[0],ttagOffset, '\n')

    if syncTTagDiff > 0:  # Bob's sync occurs before Alices
        bobSyncCh = params['bob']['channels']['sync']
        bobSyncs = data['bob']['ch'] == bobSyncCh
        bobSyncTTags = bData[bobSyncs]
        bobSyncMask = bobSyncTTags >= startTTag
        firstSyncTTag = bobSyncTTags[bobSyncMask][0]
    else:
        aliceSyncCh = params['alice']['channels']['sync']
        aliceSyncs = data['alice']['ch'] == aliceSyncCh
        aliceSyncTTags = aData[aliceSyncs]
        aliceSyncMask = aliceSyncTTags >= startTTag
        firstSyncTTag = aliceSyncTTags[aliceSyncMask][0]

    stopTTag = min(aData[-1], bData[-1])
    if dt is not None:
        endTime = dt/TTAGERRESOLUTION + startTTag
    else:
        endTime = stopTTag  # large number

    if stopTTag < endTime:
        err = True
        endTime = stopTTag

    if startTTag > endTime:
        err = True
        startTTag = 0
        endTime = max(aData[-1], bData[-1])

    aClicks = (aData >= firstSyncTTag) & (aData <= endTime)
    bClicks = (bData >= firstSyncTTag) & (bData <= endTime)


    trimmedData['alice'] = data['alice'][aClicks]
    trimmedData['bob'] = data['bob'][bClicks]
    trimmedData['alice']['ttag'] = trimmedData['alice']['ttag']-(startTTag)
    trimmedData['bob']['ttag'] = trimmedData['bob']['ttag']-(startTTag)+ttagOffset

    # Make sure each dataset has the same number of sync pulses / trials
    nSyncs = {}
    syncBool = {}
    for party in ['alice', 'bob']:
        ch = params[party]['channels']
        # syncArray, sBool = get_sync_array(trimmedData[party], ch['sync'])
        sBool, nSyncsParty = calc_n_syncs(trimmedData, party, params)
        # nSyncs[party] = sBool.sum()
        nSyncs[party] = nSyncsParty
        syncBool[party] = sBool

    # print('nSyncs before trim', nSyncs)
    if (nSyncs['alice'] == nSyncs['bob']):
        partyToTrim = None
    elif (nSyncs['alice'] > nSyncs['bob']):
        partyToTrim = 'alice'
    else:
        partyToTrim = 'bob'

    # print('party to trim', partyToTrim)

    if partyToTrim is not None:
        sync_diff = int(abs(nSyncs['alice'] - nSyncs['bob']))
        syncTTAGS = trimmedData[partyToTrim]['ttag'][syncBool[partyToTrim]]
        lastSyncTTAG = syncTTAGS[-1*sync_diff]
        mask = trimmedData[partyToTrim]['ttag'] < lastSyncTTAG

        trimmedData[partyToTrim] = trimmedData[partyToTrim][mask]
        sBoolParty, nSyncsParty = calc_n_syncs(trimmedData, partyToTrim, params)
        # print(sBool, nSyncsParty)
        nSyncs[partyToTrim] = nSyncsParty
        syncBool[partyToTrim] = sBoolParty
                # print(party, nSyncsParty)
    # print('finished trimming')

    return trimmedData, err

def calc_n_syncs(data, party, params):
    ch = params[party]['channels']
    syncArrayRef, sBoolRef = get_sync_array(data[party], ch['sync'])
    nSyncsRef = sBoolRef.sum()
    # print('inside', party, sBoolRef, nSyncsRef)
    return sBoolRef, nSyncsRef


def trim_data_single_party(data, dt=None):
    if dt is None:
        return data

    if data is None:
        return data

    startTTag = data['ttag'][0]
    endTime = dt/TTAGERRESOLUTION + startTTag
    lastTTag = data['ttag'][-1]
    if lastTTag < endTime:
        print('weird ending')
        endTime = lastTTag
    validTTags = data['ttag'] <= endTime
    return data[validTTags]


def trim_data_single(data, params):
    syncCh = params['channels']['sync']
    syncs = data['ch'] == syncCh
    syncTTags = data['ttag'][syncs]
    # print(syncTTags, syncTTags[0], data['ttag'])
    validTTags = (data['ttag'] >= syncTTags[0]) & (
        data['ttag'] <= syncTTags[-1])
    trimmedData = data[validTTags]
    return trimmedData


def get_sync_array(data, syncCh):
    '''
    Returns an array that is equal to the length of the number of timetags 
    recorded. Each element contains the sync number corresponding to that
    timetag. Useful for figuring out which synch pulse a given timetag 
    occurs in.

    data: the raw channel and ttag data aray.
    syncCh: the channel number that records the synch signal.
    '''

    syncBool = data['ch'] == syncCh
    syncs = data['ttag'][syncBool]
    syncArray = syncBool.cumsum() - 1
    return syncArray, syncBool

# @jit


def calc_settings(data, settingChannels, syncArray):
    '''
    Return an array the length of the time tag
    '''
    nSyncs = syncArray[-1] + 1
    settingsSync = np.zeros(nSyncs)
    j = 1
    for ch in settingChannels:
        setBool = data['ch'] == ch
        settingsSync[syncArray[setBool]] += j
        j += 1

    settings = settingsSync[syncArray]
    return settings, settingsSync

# @jit


def calc_period(det, divider=800, syncCh=6, maxPeriod=170):
    '''
    Need to determine the period of the laser. Cand do this by looking
    at how often the sync pulses come in and use the divider information
    to back out the laser period in terms of timetagging bins. Returns the 
    mean period as well as an array of the average period of the laser 
    between synch pulses.

    maxPeriod is an upper bound on the laser period. Any values computed
    larger than this are treated as an error, and their values are replaced
    with the previous valid value.
    '''
    syncBool = det['ch'] == syncCh
    syncs = det['ttag'][syncBool]
    laserPeriodArray = np.diff(syncs)/divider

    laserPeriodArray = np.append(laserPeriodArray, laserPeriodArray[-1])
    laserPeriod = np.mean(laserPeriodArray[laserPeriodArray < maxPeriod])
    errorLP = np.where(laserPeriodArray > maxPeriod)
    laserPeriodArray[errorLP] = laserPeriodArray[np.roll(errorLP, -1)]
    return(laserPeriod, laserPeriodArray)

# @jit

def check_for_timetagger_roll_over(rawData, params):
    rollover = {'err': False, 'party': [], 'position': []}

    for key in rawData:
        ttags = rawData[key]['ttag']
        diffTTags = np.diff(ttags)
        neg_indx = np.where(diffTTags<0)[0]
        if len(neg_indx)>0:
            rollover['err'] = True 
            rollover['party'].append(key)
            rollover['position'].append(neg_indx)
    return rollover 

def check_for_timetagger_jump(rawData, params):
    jump = {'skip': False, 'jumpInfo': {}, 'err': False}
    error = None
    for key in rawData:
        syncBool = rawData[key]['ch'] == params[key]['channelmap']['sync']
        sync = rawData[key]['ttag'][syncBool]

        diffTTags = np.abs(np.diff(sync))
        avgDiff = np.mean(diffTTags)
        scale = 1.01
        pos = np.where(diffTTags > avgDiff * scale)[0]
        ttags = sync[pos]

        if len(pos)>0:
            jump['skip'] = True
            jump['party'].append(key)
            # jump['position'].append(pos[0])
            jump['jumpInfo'][key] = {}
            jump['jumpInfo'][key]['position'] = pos
            jump['jumpInfo'][key]['ttag'] = ttags

    # print('jumps', jump)
    if jump['skip']:
        error = {'jointSkip':False, 'err': False, 'info':{}}
        error['info'] = jump['jumpInfo']
        # print('looking for errors')
        all_jump_pos = []
        ji = jump['jumpInfo']
        # Compute the overlap between jump events
        if len(jump['party']>1): # errors on both parties
            overlap, a_ind, b_ind = np.intersect1d(ji['alice']['position'], ji['bob']['position'], return_indices=True)
            
            if len(overlap)>0:
                error['jointSkip']=True
            '''
            Detect if only one timetagger jumped. This is an error.
            '''
            for party in jump['jumpInfo']:
                if party=='alice':
                    indx = a_ind
                else:
                    indx = b_ind
                mask = np.ones(len(ji[party]['position']), dtype=bool)
                mask[indx] = False 
                unique_jumps = ji[party]['position'][mask]
                if len(unique_jumps)>0:
                    error['err']=True

        else: #errors on one party
            error['jointSkip'] = False
            error['err'] = True
        '''
        Detect whether both timetaggers jump together. This tends not to introduce
        errors.
        '''
        



        # print(overlap, x_ind, y_ind)
        # print('')

        # for party in jump['jumpInfo']:
        #     n_jumps.append(len(jump['jumpInfo'][party]['position']))
        #     all_jump_pos+=jump['jumpInfo'][party]['position'].tolist()   
        # # print('n_jumps', n_jumps)

        # print('all jumps', all_jump_pos)
        # unique_jumps = np.unique(np.array(all_jump_pos),return_inverse=False)
        # duplicate_jumps, unique_jumps_indx = np.unique(np.array(all_jump_pos),return_inverse=True)
        # # duplicate_jumps_indx = unique_jumps_indx>0
        # # print('indicies:', duplicate_jumps_indx, unique_jumps_indx)
        # # duplicate_jumps = np.array(all_jump_pos)[duplicate_jumps_indx]
        # print('duplicate jumps', duplicate_jumps, 'unique_jumps', unique_jumps)
        # if len(unique_jumps)>0:
        #     # print('In jumps, ERROR')
        #     error['err']=True
        # if len(duplicate_jumps)>0:
        #     # both syncs skip together
        #     error['jointSkip']=True


        # if len(n_jumps)>1:
        #     n_jumps_diff = np.abs(np.diff(np.array(n_jumps)))
        #     # print('n_jumps_diff', n_jumps_diff)
        #     diff_sum = n_jumps_diff.sum()
        #     # print('diff_sum', diff_sum)
        #     if diff_sum>0:
        #         print('In jumps, ERROR')
        #         jump['err']=True

        # if len(pos)>0:
        #     print(pos)
        #     mask = np.array([True]*len(rawData[key]))
        #     # print('mask', mask)
        #     mask[pos]=False 
        #     syncBool[pos] = False
        #     goodSync = rawData[key]['ttag'][syncBool]
        #     diffTTagsGood = np.abs(np.diff(goodSync))
        #     avgDiffGood = np.mean(diffTTagsGood)
        #     # jump[key] = pos 

        #     # ttagDiff = sync - sync[0] - avgDiffGood
        #     diffTTagsGood = np.diff(sync) - avgDiffGood
        #     print('diffTTagsGood',diffTTagsGood)

        #     # for p in pos:
        #     #     syncBool = rawData[key]['ch'] == params[key]['channelmap']['sync']
        #     #     ttagJump = rawData[key]['ttag'][p]
        #     #     maskJump = rawData[key]['ttag'][syncBool]>= ttagJump
        #     #     jumpOffset = diffTTagsGood[p]
        #     #     print('jumpOffset', jumpOffset)
        #     #     syncs = rawData[key]['ttag'][syncBool]
        #     #     syncs[maskJump]=syncs[maskJump]-jumpOffset
        #     data = rawData[key]['ttag']
        #     for i,p in enumerate(pos):
        #         # ttagJump = rawData[key]['ttag'][p]
        #         # print(key, i, p)

        #         ttagJump = ttags[i]
        #         try:
        #             ttagEndJump = ttags[i+1]
        #         except:
        #             ttagEndJump = data[-1]
        #         maskJump = (rawData[key]['ttag']>= ttagJump) & (rawData[key]['ttag']<ttagEndJump)
        #         jumpOffset = diffTTagsGood[p]
        #         # print('jumpOffset', jumpOffset, maskJump, ttagJump)
                
        #         # rawData[key]= rawData[key][maskJump]

        #         # data=data[maskJump]
        #         data[maskJump]=data[maskJump]-jumpOffset 
        #         # syncBool = rawData[key]['ch'] == params[key]['channelmap']['sync']
        #         # syncs = rawData[key]['ttag'][syncBool]
        #         # sync[p] = sync[p]-jumpOffset 

    
    
    return error, rawData


def calc_phase_info(det, divider, ch):

    laserPeriod, laserPeriodArray = calc_period(det, divider, ch['sync'])

    syncArray, syncBool = get_sync_array(det, ch['sync'])
    syncTTAGs = det['ttag'][syncBool]
    # Create an array the same length as the det record of timetags,
    # but containing the sync timetag for that period. Useful for
    # computing timetags of events relative to the start of a sync pulse.
    syncAtEveryEvent = syncTTAGs[syncArray]
    ttagModSync = det['ttag'] - syncAtEveryEvent

    phase = ttagModSync % laserPeriodArray[syncArray]
    clickBool = det['ch'] == ch['detector']

    laserPulse = np.floor(ttagModSync*1./laserPeriodArray[syncArray])
    # print('Phase', laserPeriod, laserPeriodArray)
    return(phase, laserPulse, ttagModSync, laserPeriod)


def calc_phase_histogram(laserPeriod, phase, clickBool):
    bins = np.arange(0, np.floor(laserPeriod))
    n, x = np.histogram(phase[clickBool], bins=bins)
    phaseHist = {'x': (x[:-1]+x[1:])/2., 'y': n}
    pkIdx = np.argmax(phaseHist['y'])
    return phaseHist, pkIdx


def calc_coinc_window_mask(det, params, divider, pkIdx='auto', ):
    divider = divider*1.
    ch = params['channels']
    radius = params['radius']
    clickBool = det['ch'] == ch['detector']

    phase, laserPulse, ttagModSync, laserPeriod = calc_phase_info(
        det, divider, ch)
    phaseHist, pk = calc_phase_histogram(laserPeriod, phase, clickBool)
    # print(phaseHist)

    if (pkIdx is True) or (pkIdx is None):
        pkIdx = pk

    pkIdx = int(pkIdx)
    lowPhase = pkIdx - radius
    highPhase = pkIdx + radius
    inWindowMask = (phase > lowPhase) & (phase < highPhase)

    phaseHist['lowPhase'] = lowPhase
    phaseHist['highPhase'] = highPhase
    phaseHist['pkIdx'] = pkIdx
    phaseHist['phaseHist'] = phaseHist
    phaseHist['laserPeriod'] = laserPeriod
    return inWindowMask, laserPulse, phaseHist

# def calc_pockels_mask(data, laserPulse, pcStart, pcStop, abDelay):
#     # Now check for events that are inside the pockels cell Window
#     pcStart = params['analysis']['start']
#     pcStop = params['analysis']['stop']
#     inPCWindowMask = (laserPulse>pcStart) & (laserPulse<pcStop)
#     outPCWindowMask = np.logical_not(inPCWindowBool)
#     # results['laserPulsePC'] = laserPulse[inWindowMask&clickBool&inPCWindowBool]
#     # results['laserPulseNotPC'] = laserPulse[inWindowMask&clickBool&outPCWindowBool]
#     return inPCWindowMask, outPCWindowMask

# @jit


def calc_data_properties(data, params, divider, findPk=False):
    props = {}
    for party in data:
        det = data[party]
        p = params[party]
        # print('properties', p, det)
        props[party] = calc_data_properties_one_party(det, p, divider,
                                                      findPk=findPk, party=party)
    return props


def calc_data_properties_one_party(data, params, divider,
                                   findPk=False, party=''):

    props = {}
    ch = params['channels']
    det = data

    syncArray, syncBool = get_sync_array(det, ch['sync'])
    nSyncs = syncBool.sum()

    clickBool = det['ch'] == ch['detector']

    #  Settings calculations
    try:
        settingChannels = [ch['setting0'], ch['setting1']]
        settings, settingsSync = calc_settings(
            data, settingChannels, syncArray)
        props['settings'] = settings
        props['settingsSync'] = settingsSync
    except Exception:
        props['settings'] = None
        props['settingsSync'] = None

    # Compute the phase histogram and coincidence widow mask.
    # Also return the laserPulse information
    if findPk:
        pkIdx = True
    else:
        pkIdx = params['pkIdx']
    inWindowMask, laserPulse, phaseHist = calc_coinc_window_mask(
        det, params, divider, pkIdx=pkIdx)
    phaseHist['name'] = party

    # Compute the FWHM
    try:
        fwhm = find_fwhm(phaseHist['x'], phaseHist['y'])
    except Exception:
        fwhm = 0.

    laserPulseHist = laser_pulse_histogram(
        laserPulse, inWindowMask, clickBool, divider)

    props['divider'] = divider
    props['syncArray'] = syncArray
    props['syncBool'] = syncBool
    props['clickBool'] = clickBool
    props['nSyncs'] = nSyncs
    props['inWindowMask'] = inWindowMask
    props['phaseHist'] = phaseHist
    props['pkIdx'] = phaseHist['pkIdx']
    props['lowPhase'] = phaseHist['lowPhase']
    props['highPhase'] = phaseHist['highPhase']
    props['laserPeriod'] = phaseHist['laserPeriod']
    props['fwhm'] = fwhm
    props['laserPulseHist'] = laserPulseHist
    props['laserPulse'] = laserPulse
    props['params'] = params
    # print(party, phaseHist['laserPeriod'])
    # props['inPCWindowMask'] = inPCWindowMask
    # props['outPCWindowMask'] = outPCWindowMask

    return(props)


def get_processed_data(data, props, offset, pcStart, pcLength):
    settings = props['settingsSync'].astype(int)
    syncArray = props['syncArray']
    syncBool = props['syncBool']

    phaseMask = get_window_mask(data, props)
    pockelsMask, paramsPockels = get_pockels_mask(data, props,
                                                  offset, pcStart, pcLength)
    laserPulse = deepcopy(props['laserPulse'])
    # print('laser pulse', laserPulse)
    laserPulse = laserPulse - offset - pcStart  # index first lp to 0
    # print('laser pulse', laserPulse)
    validMask = phaseMask & pockelsMask
    # print('\n valid detections', np.sum(validMask), np.sum(phaseMask), np.sum(pockelsMask), '\n')

    # Remove detections not in the valid window
    laserPulse[np.logical_not(validMask)] = 0
    # print('laserPulses')
    # print(laserPulse, min(laserPulse), max(laserPulse))
    # print('')
    laserPulseVal = 2**(laserPulse)-1
    laserPulseCumSum = np.cumsum(laserPulseVal)
    laserPulseSyncVals = laserPulseCumSum[syncBool]
    detectionVals = np.diff(laserPulseSyncVals)
    # Need to correctly shift the encoded values back by one
    detectionVals[detectionVals > 0] += 1
    detectionVals = detectionVals.astype(int)

    params = props['params']
    syncCh = params['channels']['sync']
    syncs = data['ch'] == syncCh
    syncTTags = (data['ttag'][syncs]).astype(int)
    # print('\n finding lengths')
    # print(len(settings), len(syncTTags), len(detectionVals))

    results = {'Setting':  np.asarray([]).astype(int), 'Outcome':  np.asarray(
        []).astype(int), 'SyncTTag': np.asarray([]).astype(int)}

    nSyncs = min(len(settings), len(syncTTags), len(detectionVals))
    results['Setting'] = settings[0:nSyncs]
    results['Outcome'] = detectionVals[0:nSyncs]
    results['SyncTTag'] = syncTTags[0:nSyncs]
    # print('\n RESULTS')
    # print(detectionVals, max(detectionVals))
    # print('\n')

    return results


def trim_processed_data(data, ttagOffset, syncTTagDiff):
    for party in data.keys():
        data[party]['SyncTTag'] += ttagOffset[party]
    minTTag = []
    maxTTag = []
    for key in data.keys():
        minTTag.append(data[key]['SyncTTag'][0])
        maxTTag.append(data[key]['SyncTTag'][-1])

    startTTag = np.max(minTTag)
    stopTTag = np.min(maxTTag)

    if syncTTagDiff > 0:  # Bob's sync occurs before Alices
        party = 'bob'
    else:
        party = 'alice'

    syncMask = data[party]['SyncTTag'] >= startTTag
    firstSyncTTag = data[party]['SyncTTag'][syncMask][0]

    for key in data.keys():
        validRangeMask = (data[key]['SyncTTag'] >= firstSyncTTag) & (
            data[key]['SyncTTag'] <= stopTTag)
        for option in data[key].keys():
            data[key][option] = data[key][option][validRangeMask]
        del data[key]['SyncTTag']

    err = False
    return err, data


def compress_binary_data(data, aggregate=False):
    # f = open(fname, 'a+')
    '''
    data types:
    'u1' = 1-byte unsinged integer
    'u8' = 8-byte unsigned integers
    '''
    sA = data['alice']['Setting'].astype('u1')  # Alice Settings
    sB = data['bob']['Setting'].astype('u1')  # Bob settings
    eA = data['alice']['Outcome'].astype('u8')  # Alice outcome
    eB = data['bob']['Outcome'].astype('u8')  # Bob outcome
    if aggregate:
        print('Aggregating data')
        dataType = [('sA', 'u1'), ('sB', 'u1'), ('eA', 'u1'), ('eB', 'u1')]
        eA = (eA > 0).astype('u1')
        eB = (eB > 0).astype('u1')
        print(np.max(eA), np.max(eB))
    else:
        dataType = [('sA', 'u1'), ('sB', 'u1'), ('eA', 'u8'), ('eB', 'u8')]

    # Create a structured array. Each row represents the results from one trial.
    data = np.zeros(len(sA), dtype=dataType)

    data['sA'] = sA
    data['sB'] = sB
    data['eA'] = eA
    data['eB'] = eB

    # data.tofile(fname)
    binData = data.tobytes()
    compressedData = zlib.compress(binData, level=-1)

    return compressedData


def write_to_compressed_file(fname, data, aggregate=False):
    compressedData = compress_binary_data(data, aggregate=aggregate)

    with open(fname, mode="wb") as fout:
        fout.write(compressedData)

    return compressedData


def write_single_party_to_compressed_file(fname, data, save='bin'):
    '''
    data types:
    'u1' = 1-byte unsinged integer
    'u8' = 8-byte unsigned integers
    '''
    setting = data['Setting'].astype('u1')
    event = data['Outcome'].astype('u8')
    ttag = data['SyncTTag'].astype('u8')
    dataType = [('Setting', 'u1'), ('Outcome', 'u1'), ('SyncTTag', 'u8')]

    # Create a structured array. Each row represents the results from one trial.
    data = np.zeros(len(setting), dtype=dataType)

    data['Setting'] = setting
    data['Outcome'] = event
    data['SyncTTag'] = ttag

    if save == 'bin':
        f = open(fname, 'a+')
        data.tofile(f)
        f.close()
    else:
        np.savez_compressed(fname, data=data)
    return(fname)

########


def get_pockels_mask(data, props, offset, pcStart, pcLength):
    pcStop = pcStart + pcLength

    if data is None:
        pockelsMask = np.array([True]*len(data))
        params = None
    else:
        laserPulse = props['laserPulse']
        startLP = pcStart+offset
        stopLP = pcStop+offset
        pcMask = (laserPulse > startLP) & (laserPulse < stopLP)
        params = {'xbar1': startLP, 'xbar2': stopLP}
        pockelsMask = pcMask

    return pockelsMask, params


def get_window_mask(data, props):
    if data is None:
        mask = np.array([True]*len(data))
    else:
        coincWindowMask = props['inWindowMask']
        detMask = props['clickBool']
        mask = coincWindowMask & detMask
        # print('\n Phase Mask', np.sum(coincWindowMask), np.sum(detMask), np.sum(mask), '\n')
    return mask


def get_darks_mask(data, props):
    if data is None:
        mask = np.array([True]*len(data))
    else:
        darkWindowMask = np.logical_not(props['inWindowMask'])
        detMask = props['clickBool']
        mask = detMask & darkWindowMask
    return mask

#########

# @jit


def find_fwhm(X, Y):
    '''
    Compute the full-width-half-max of the detection events
    '''
    frac = 0.5
    d = Y - (max(Y) * frac)
    indexes = np.where(d > 0)[0]
    fwhm = np.abs(X[indexes[-1]] - X[indexes[0]]) + 1
    return(fwhm)

# @jit


def remove_duplicate_ttags(rawData):
    '''
    Find and remove any timetags that are duplicates.
    '''
    ttags = rawData['ttag']
    diffTtags = np.diff(ttags)
    diffTtags0 = diffTtags == 0
    diffTtags0Bool = np.append(np.invert(diffTtags0), [True])

    rawData['ch'] = rawData['ch'][diffTtags0Bool]
    rawData['ttag'] = rawData['ttag'][diffTtags0Bool]

    return(rawData)

# @jit


def get_reduced_data(rawData, params):
    '''
    Make sure that the first timetag in Alice and Bob's data
    corresponds to a sync pulse, and the last ttag is the last
    sync pulse.
    '''
    reducedData = {}
    for key in rawData:
        det = rawData[key]
        syncBool = det['ch'] == params[key]['channels']['sync']
        syncIdx = np.where(syncBool)[0]

        first = syncIdx[0]
        last = syncIdx[-1] + 1  # Include the last one...

        det = det[first:last]
        reducedData[key] = det
    return(reducedData)


def laser_pulse_histogram(laserPulse, inWindowMask, clickBool, divider):
    n, x = np.histogram(
        laserPulse[inWindowMask & clickBool], np.arange((divider+1)))
    laserPulseHist = {'x': x[:-1], 'y': n, 'name': ''}
    return laserPulseHist


def calc_offset_histograms(props):
    clickBool = props['clickBool']
    inWindowMask = props['inWindowMask']
    syncBool = props['syncBool']
    divider = props['divider']
    laserPulse = props['laserPulse']
    syncArray = props['syncArray']

    syncLaserPulse = np.diff(((clickBool & inWindowMask).cumsum())[syncBool])
    # effectively sub-trial number. Formerly labeled laserPulse2
    subTrial = laserPulse + (divider)*(syncArray)
    subTrial = subTrial*(inWindowMask & clickBool).astype(int)

    return syncLaserPulse, subTrial


def calc_convolution(y1, y2):
    y1 = y1 - np.average(y1)
    y2 = y2 - np.average(y2)
    convolution = scipy.signal.fftconvolve(y1, y2[::-1])
    convDelay = np.argmax(convolution) - (len(y2) - 1)
    convMax = np.max(convolution)
    return convolution, convDelay, convMax


def find_offset_coinc(props, abdelay, corr):
    if abdelay < 0:
        dA = props['alice']['laserPulse2']
        dB = props['bob']['laserPulse2'] + corr
    else:
        dA = props['alice']['laserPulse2'] - corr
        dB = props['bob']['laserPulse2']

    startSync = max(min(dA), min(dB))
    stopSync = min(max(dA), max(dB))
    # print("syncs", startSync, stopSync)

    dABool = (dA >= startSync) & (dA <= stopSync+1)
    dBBool = (dB >= startSync) & (dB <= stopSync+1)

    dA = dA[dABool]
    dB = dB[dBBool]

    coinc = np.intersect1d(dA, dB).astype(int)
    return(coinc)


def calc_offset(data, params, divider):
    DIVIDER = divider

    # print('calculating properties')
    props = calc_data_properties(data, params, divider=divider, findPk=True)

    # Save the peak positions for Alice and Bob
    peakIdx = {}
    for party in props:
        peakIdx[party] = props[party]['pkIdx']
    # print('finished properties')
    for key in data:
        # print('offset histogram', key)
        syncLaserPulse, laserPulse2 = calc_offset_histograms(props[key])
        props[key]['syncLaserPulse'] = syncLaserPulse
        props[key]['laserPulse2'] = laserPulse2

    # print('Starting first convolution')
    syncLaserPulseA = props['alice']['syncLaserPulse']
    syncLaserPulseB = props['bob']['syncLaserPulse']
    convSyncLaserPulse, abdelay, cMax = calc_convolution(
        syncLaserPulseA, syncLaserPulseB)
    # print('abdelay', abdelay)

    # # Now find out the offset in laser pulse number between Alice and Bob
    laserPulseA = props['alice']['laserPulseHist']['y']
    laserPulseB = props['bob']['laserPulseHist']['y']
    convLP, lpDelay, convLPMax = calc_convolution(laserPulseA, laserPulseB)

    offsetVals = np.where(convLP > convLPMax*0.5)[0]
    coincArray = np.zeros(len(offsetVals))
    corrArray = np.zeros(len(offsetVals))
    offsetLaserPulseArray = np.zeros(len(offsetVals))
    # print("offset between AB", offsetVals)

    '''
    Since there are multiple possible laser pulse offsets that show up
    in the convultion peak, iterate over all possible candidates and 
    calculate the number of coincidences each one predicts. 
    '''
    for i in range(len(offsetVals)):
        offset = offsetVals[i]
        offsetLaserPulseArray[i] = offset - (len(laserPulseB) - 1)
        corrArray[i] = (abdelay*DIVIDER + offsetLaserPulseArray[i])
        coinc = find_offset_coinc(props, abdelay, corrArray[i])
        coincArray[i] = len(coinc)

    # find which offset yielded the maximum coincidences
    ix = np.argmax(coincArray)
    offsetLaserPulse = offsetLaserPulseArray[ix]
    corr = corrArray[ix]
    offset = offsetVals[ix]
    coinc = find_offset_coinc(props, abdelay, corr)
    # coinc = coincArray[ix]
    print("coinc", coincArray[ix])
    print('')

    # Now compute the average timetag offset between coincidence pairs
    if len(coinc) > 0:
        if abdelay < 0:
            aIndx = np.in1d(props['alice']['laserPulse2'].astype(int), coinc)
            bIndx = np.in1d(
                props['bob']['laserPulse2'].astype(int), coinc - corr)
        else:
            aIndx = np.in1d(
                props['alice']['laserPulse2'].astype(int), coinc + corr)
            bIndx = np.in1d(props['bob']['laserPulse2'].astype(int), coinc)
        aTTag = data['alice']['ttag'][aIndx]
        bTTag = data['bob']['ttag'][bIndx]

        diffTTag = 1.*(aTTag*1. - 1.*bTTag)
        ttagOffset = np.mean(diffTTag*1.)
        # print( coinc, ttagOffset, ttagOffset - min(diffTTag), ttagOffset - (max(diffTTag)) )
    else:
        print("no Coincidences found during Offset search")
        ttagOffset = 0

    # print('offset results:',abdelay, offsetLaserPulse, ttagOffset)
    # print('Starting to find the sync order')
    # Check to see which sync comes first in time:
    syncBoolA = props['alice']['syncBool']
    syncBoolB = props['bob']['syncBool']
    # print('found sync bool')
    syncTTagsA = data['alice']['ttag'][syncBoolA]
    syncTTagsB = data['bob']['ttag'][syncBoolB] + ttagOffset
    # print(abdelay)
    if abdelay > 0:
        firstAliceSync = syncTTagsA[abdelay]
        firstBobSync = syncTTagsB[0]
    else:
        firstAliceSync = syncTTagsA[0]
        firstBobSync = syncTTagsB[-1*abdelay]

    syncTTagDiff = firstAliceSync-firstBobSync
    # print('Difference in sync ttags', syncTTagDiff)
    # print('')
    ret = {}
    ret['abDelay'] = abdelay
    ret['offsetLaserPulse'] = offsetLaserPulse
    ret['ttagOffset'] = ttagOffset
    ret['syncTTagDiff'] = syncTTagDiff
    ret['pkIdx'] = peakIdx

    return ret


def calc_violation(stats):
    #  J = P(++|ab)        - P(+0|a'b)           - P(0+|ab')         - P(++|a'b')

    pab11 = stats[3, 3]
    papb11 = stats[1, 3]
    pabp11 = stats[2, 3]
    papbp11 = stats[0, 3]
    papb10 = stats[2, 1]-pabp11
    pabp01 = stats[1, 2]-papb11
    chSingles = (stats[1, 2]+stats[3, 2]+stats[2, 1]+stats[3, 1])*1./2.

    chCoinc = pab11+papb11+pabp11-papbp11
    # Jparts = [stats['coin'][1,1], -stats['alice'][2,1]+stats['coin'][2,1], -stats['bob'][1,2]+stats['coin'][1,2], -stats['coin'][2,2]]
    # #print('Jparts: %r'%Jparts)
    # J = stats['coin'][1,1] - stats['alice'][2,1] - stats['bob'][1,2] - stats['coin'][2,2]
    # #print('J: %d'%sum(Jparts))

    # CH_single = (stats['alice'][1,1]*1. + stats['alice'][2,1] + stats['bob'][1,1] + stats['bob'][1,2] )/2.
    # CH_coin = stats['coin'][1,1] + stats['coin'][2,1] + stats['coin'][1,2] - stats['coin'][2,2]
    CH = chCoinc - chSingles
    CHn = chCoinc/chSingles
    print("CH violation:", CH)
    print("CH normalized:", CHn)

    # Pab11 = stats['coin'][1,1]
    # Papb10 = stats['alice'][2,1]-stats['coin'][2,1]
    # Pabp01 = stats['bob'][1,2]-stats['coin'][1,2]
    # Papbp11 = stats['coin'][2,2]

    Ntot = pab11 + papb10 + pabp01 + papbp11*1.
    ratio = pab11*1./Ntot

    pValue = calc_pvalue(Ntot, ratio)

    print("Ratio:", ratio, "pValue:", pValue)
    return(CH, CHn, ratio, pValue)


def calc_pvalue(N, prob):
    return(binom.cdf(N*(1-prob), N, 0.5))
