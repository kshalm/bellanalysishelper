import numpy as np
import logging
# import scipy.signal
from scipy.stats import binom
# from scipy.stats import mode
# from numba import jit
import scipy
# @jit


def calcSettings(det, key, syncArray, ch0=2, ch1=4):
    setting0Bool = det['ch'] == ch0
    setting1Bool = det['ch'] == ch1
    nSyncs = syncArray[-1] + 1
    # print(nSyncs)     #len(syncArray)# max(syncArray) +1
    settings = np.zeros(nSyncs)
    settings[syncArray[setting0Bool]] += 1
    settings[syncArray[setting1Bool]] += 2

    # , 'setting0Bool': setting0Bool, 'setting1Bool': setting1Bool}
    return({'settings': settings})

# @jit


def calc_period(det, divider=800, syncCh=6):
    # Need to calculate this in a loophole-free manner using only past data to change future data.
    # nBinsPerPulse = 13E-9 /78E-12

    nBinsPerPulse = 170
    syncBool = det['ch'] == syncCh
    syncs = det['ttag'][syncBool]
    laser_period2 = np.diff(syncs)/divider
    # print('calc period:', syncs, laser_period2)
    # print('syncs', np.sum(syncBool))
    # print('lp', laser_period2, 'lp')
    laser_period2 = np.append(laser_period2, laser_period2[-1])
    laser_period1 = np.mean(laser_period2[laser_period2 < nBinsPerPulse])
    errorLP = np.where(laser_period2 > nBinsPerPulse)
    # Replace the errant periods with the previous period.
    laser_period2[errorLP] = laser_period2[np.roll(errorLP, -1)]
    print("max", max(laser_period2), laser_period1)
    return(laser_period1, laser_period2)

# @jit


def check_for_timetagger_jump(rawData, paramsCh):
    jump = {'skip': False, 'party': [], 'position': []}

    for key in rawData:
        syncBool = rawData[key]['ch'] == paramsCh[key]['channels']['sync']
        sync = rawData[key]['ttag'][syncBool]

        diffTTags = np.abs(np.diff(sync))
        avgDiff = np.mean(diffTTags)
        scale = 2
        pos = np.where(diffTTags > avgDiff * scale)
        if (pos[0] > 0):
            jump['skip'] = True
            jump['party'].append(key)
            jump['position'].append(pos[0])
    return(jump)


def check_for_timetagger_jump_single_party(rawData, paramsCh, ch='sync'):
    jump = {'skip': False, 'position': []}

    chBool = rawData['ch'] == paramsCh['channels'][ch]
    chTTags = rawData['ttag'][chBool]

    diffTTags = np.abs(np.diff(chTTags))
    avgDiff = np.mean(diffTTags)
    scale = 2
    if ch == 'detector':
        scale = 500
    pos = np.where(diffTTags > avgDiff * scale)
    if (pos[0] > 0):
        jump['skip'] = True
        jump['position'].append(pos[0])
    return jump

# @jit


def calc_phase(det, divider, ch):
    clickBool = det['ch'] == ch['detector']
    syncBool = det['ch'] == ch['sync']
    # byncBool = det['ch'] == ch['bSync']
    laser_period1, laser_period2 = calc_period(det, divider, ch['sync'])
    # print("PERIOD: ", laser_period1, laser_period2)

    syncs = det['ttag'][syncBool]
    syncArray = syncBool.cumsum() - 1
    ttagModSync = det['ttag'] - syncs[syncArray]

    phase = ttagModSync % laser_period2[syncArray]
    bins = np.arange(0, np.floor(laser_period1))
    n, x = np.histogram(phase[clickBool], bins=bins)

    laserPulse = np.floor(ttagModSync/laser_period2[syncArray])
    phaseHist = {'x': (x[:-1]+x[1:])/2., 'y': n}
    pkIdx = np.argmax(phaseHist['y'])
    # print("PEAK INDX:", pkIdx)
    return(phase, phaseHist, pkIdx, laserPulse, ttagModSync, laser_period2)

# @jit


def calc_data_properties(data, paramsCh, findingOffset=False, findPk=False):

    results = {}
    reducedData = data
    # reducedData = get_reduced_data(data, paramsCh)
    for key in reducedData:
        # ch = {'sync': 6, 'detector': 0}
        # chKey = paramsCh[key]['channels']
        ch = paramsCh[key]['channels']
        # ch = {'sync': chKey['sync'], 'detector': chKey['detector'], 'setting0': chKey['setting0'], 'setting1': chKey['setting1']}

        det = reducedData[key]

        results[key] = {}
        syncBool = det['ch'] == ch['sync']
        results[key]['syncBool'] = syncBool
        nSyncs = syncBool.sum()

        # print("In parameters", nSyncs)

        # For each ttag, figure out what trial it is a part of. Cumsum is an easy way to do this.
        # This creates a syncArray where of what sync pulse corresponds to what ttag.
        syncArray = syncBool.cumsum() - 1  # This is effectively the trial number
        # print( 'len syncArray',len(syncArray), len(syncBool))
        syncs = det['ttag'][syncBool]
        clickBool = det['ch'] == ch['detector']

        results[key]['clickBool'] = clickBool
        results[key]['syncArray'] = syncArray

        #  Settings calculations
        try:
            res = calcSettings(det, key, syncArray,
                               ch['setting0'], ch['setting1'])
            results[key]['settings'] = res['settings']
        except Exception:
            results[key]['settings'] = None

        divider = paramsCh['divider']*1.
        results[key]['divder'] = divider

        (phase, phaseHist, pk, laserPulse,
         ttagModSync, laserPeriod) = calc_phase(det, divider, ch)
        results[key]['laserPeriod'] = laserPeriod
        # disregard this pk indx.
        phaseHist['name'] = key
        results[key]['phase'] = phase
        results[key]['phaseHist'] = phaseHist
        phaseRadius = paramsCh[key]['radius']
        try:
            fwhm = find_fwhm(phaseHist['x'], phaseHist['y'])
        except Exception:
            fwhm = 0.
        results[key]['fwhm'] = fwhm

        if findPk:
            pkIdx = pk
            # phaseRadius = 7
        else:
            pkIdx = paramsCh[key]['pkIdx']

        pkIdx = int(pkIdx)
        # phaseRadius = params[key]['radius']

        lowPhase = results[key]['phaseHist']['x'][pkIdx] - phaseRadius
        highPhase = results[key]['phaseHist']['x'][pkIdx] + phaseRadius

        inWindowBool = (phase > lowPhase) & (phase < highPhase)
        results[key]['pkIndx'] = pkIdx
        results[key]['lowPhase'] = lowPhase
        results[key]['highPhase'] = highPhase
        results[key]['inWindowBool'] = inWindowBool

        #  Use the hisogram below to determine the number of offset in number of syncs between
        #    the ttag's at Alice and Bob... Will use the convolution to find this number
        n, x = np.histogram(syncArray[inWindowBool], np.arange(1*(nSyncs+1)))
        results[key]['detSync'] = (n, x)

        n, x = np.histogram(
            laserPulse[inWindowBool & clickBool], np.arange((divider+1)))
        results[key]['laserPulse'] = laserPulse
        laserPulseHist = {'x': x[:-1], 'y': n, 'name': key}
        results[key]['laserPulseHist'] = laserPulseHist

        # Now check for events that are inside the pockels cell Window
        pcStart = paramsCh['analysis'][key]['start']
        pcStop = paramsCh['analysis'][key]['stop']
        inPCWindowBool = (laserPulse > pcStart) & (laserPulse < pcStop)
        #outPCWindowBool = (laserPulse>pcStop) or (laserPulse<pcStart)
        outPCWindowBool = np.logical_not(inPCWindowBool)
        # print(outPCWindowBool.sum(), inPCWindowBool.sum(), (laserPulse>0).sum() )
        results[key]['laserPulsePC'] = laserPulse[inWindowBool &
                                                  clickBool & inPCWindowBool]
        results[key]['laserPulseNotPC'] = laserPulse[inWindowBool &
                                                     clickBool & outPCWindowBool]
        # print(results[key]['laserPulseNotPC'].tolist())

        # laser_period1, laser_period2 = calc_period(det, divider, 6)
        # syncLaserPulse = np.floor(ttagModSync/laser_period2[syncArray])  # effectively sub-trial #

        syncLaserPulse = np.diff(
            ((clickBool & inWindowBool).cumsum())[syncBool])

        # n,x = np.histogram(syncLaserPulse[inWindowBool], np.arange((divider+1)))
        # results[key]['syncLaserPulse'] = syncLaserPulse
        # results[key]['syncLaserPulseHist']={'x': x[:-1], 'y': n, 'name':key}
        results[key]['syncLaserPulse'] = syncLaserPulse

        #
        # effectively sub-trial #
        laserPulse2 = laserPulse + (divider)*(syncArray)
        results[key]['laserPulse2'] = laserPulse2 * \
            (inWindowBool & clickBool).astype(int)
        # print("LASER PULSE", key, laserPulse)
        # print(key, "Nearly done")
        if findingOffset:
            # print(key, "findingOffset")
            n, x = np.histogram(
                laserPulse2[inWindowBool], np.arange((divider+1)*(nSyncs+1)))
            results[key]['laserPulseHist2'] = {
                'x': x[:-1], 'y': n, 'name': key}

    return(results, reducedData)

# @jit


def find_fwhm(X, Y):
    frac = 0.5
    d = Y - (max(Y) * frac)
    # print(max(Y), d)
    indexes = np.where(d > 0)[0]
    # print(X[indexes])
    fwhm = np.abs(X[indexes[-1]] - X[indexes[0]]) + 1
    return(fwhm)

# @jit


def remove_duplicate_ttags(rawData):
    ttags = rawData['ttag']
    diffTtags = np.diff(ttags)
    diffTtags0 = diffTtags == 0
    print("duplicates", np.cumsum(diffTtags0))
    print('')
    diffTtags0Bool = np.append(np.invert(diffTtags0), [True])

    rawData['ch'] = rawData['ch'][diffTtags0Bool]
    rawData['ttag'] = rawData['ttag'][diffTtags0Bool]

    return(rawData)

# @jit


def get_reduced_data(rawData, paramsCh):
    reducedData = {}
    for key in rawData:
        print(key)
        # print(key, rawData)
        # det = remove_duplicate_ttags(rawData[key])
        det = rawData[key]
        syncBool = det['ch'] == paramsCh[key]['channels']['sync']
        syncIdx = np.where(syncBool)[0]

        first = syncIdx[0]
        last = syncIdx[-1] + 1  # Include the last one...
        # print(key, first, last, len(syncBool))
        # print(len(syncIdx))
        det = det[first:last]
        # print(key, "Length DATA:", len(det), len(rawData[key]), len(syncBool), len(det['ch']==6))
        # print("Syncs INDX", len(syncIdx), first, last)
        reducedData[key] = det
    return(reducedData)


def calc_pc_on_off_slots(rawData, paramsCh):
    results, reducedData = calc_data_properties(rawData, paramsCh, findPk=True)
    offset = calc_offset_sync_bb(results, reducedData, paramsCh)
    # print(offset)
    pcStartStop = {'alice': [], 'bob': [],
                   'bbOffset': offset, 'bb': {'alice': [], 'bob': []}}

    for key in reducedData:
        # ch = paramsCh[key]['channels']
        lpHist = results[key]['laserPulseHist']['y']
        # print('raw', results[key]['laserPulseHist']['y'])
        # lpHistDiff = np.abs(np.diff(lpHist))
        # print('diff', lpHistDiff)
        lpHistAvg = lpHist.sum()*1./len(lpHist)*1
        # print('Avg', lpHistDiffAvg)
        thresh = (np.max(lpHist) + lpHistAvg)*0.5
        # thresh = np.max(lpHist) * .75
        # print('thresh', thresh)
        lpHistTh = np.where(lpHist > thresh)
        # strt = lpHist[lpHistTh][0]
        # stp = lpHist[lpHistTh][-1]
        # print(lpHistTh)
        strt = lpHistTh[0][0]
        stp = lpHistTh[0][-1]
        pcStartStop[key] = [strt, stp]
        # print(strt, stp, offset)
        bbStStp = [strt - offset[key], stp - offset[key]]
        pcStartStop['bb'][key] = bbStStp

    return(pcStartStop)


def calc_offset_sync_bb(results, reducedData, paramsCh):
    # results, reducedData = calc_data_properties(rawData, paramsCh, findPk = True)
    pulseOffset = {'alice': [], 'bob': []}

    for key in reducedData:
        ch = paramsCh[key]['channels']
        syncBool = results[key]['syncBool']
        syncTTags = reducedData[key]['ttag'][syncBool]
        # boolean index of all bSync locations
        bTrialBool = reducedData[key]['ch'] == ch['bSync']
        bTTags = reducedData[key]['ttag'][bTrialBool]
        bTTagsValidBool = bTTags > syncTTags[0]
        bTTags = bTTags[bTTagsValidBool]
        n = 100
        ttagDiff = bTTags[0:n] - syncTTags[0:n]
        # print('ttagDiff', ttagDiff)
        laserPeriod = np.mean(results[key]['laserPeriod'])
        offset = ttagDiff*1./laserPeriod
        # print('pulse offset', pulseOffset)
        pulseOffset[key] = np.round(np.mean(offset))
        # print('pulseoffset', key, pulseOffset)

    return(pulseOffset)

# @jit


def calc_pc_on_off_slots_bbone(rawData, paramsCh):
    results, reducedData = calc_data_properties(rawData, paramsCh, findPk=True)
    offset = 1
    offsetLaserPulse = {'alice': 0, 'bob': offset}
    abdelay = 0
    overlapWindow, eventNumber, numSyncs, startSync, stopSync, counts =\
        calc_windows(
            results, reducedData, offsetLaserPulse, paramsCh)

    divider = paramsCh['divider']
    pcStartStop = {'alice': [], 'bob': []}
    # bSyncOffset = paramsCh['bSyncOffset']
    # print(results)
    ar = ['bob', 'alice']
    for key in ar:
        ch = paramsCh[key]['channels']
        #
        # syncs = det['ttag'][syncBool]
        # syncArray = syncBool.cumsum() -1
        # ttagModSync = det['ttag'] - syncs[syncArray]

        # boolean index of all bSync locations
        bTrialBool = reducedData[key]['ch'] == ch['bSync']
        # bTrialBool = reducedData[key]['ch'] == ch['bSync'] # boolean index of all bsync locations
        firstBsync = np.where(bTrialBool)[0][0]
        bTrialCum = bTrialBool.cumsum() - 1
        print(key, 'bTrialCum', bTrialCum)
        # only the time tags associated with the bSync
        bSyncTT = reducedData[key]['ttag'][bTrialBool]
        # bSyncTTags = bSyncTT - bSyncTT[0]
        laserPeriod = np.mean(results[key]['laserPeriod'])
        # Divide by the laser period (in ttags) to find the pulse number
        # bPulseNumber = (np.round(bSyncTTags*1./laserPeriod)).astype(int)

        # Now look at the detector time tags and figure out which trial they belong to.
        # All the locations where a detection event occured within the coinc radius.
        detBool = (overlapWindow[key]['eventBool'] >= firstBsync)
        detAllBool = reducedData[key]['ch'] == ch['detector']
        # all of the time tags of valid detection events
        detTTags = reducedData[key]['ttag']
        detTTagsModbSync = (detTTags - bSyncTT[bTrialCum])[detBool]
        detPulseNumber = (
            np.floor(detTTagsModbSync*1./laserPeriod)).astype(int)
        # laserPulse = np.floor(ttagModSync/laser_period2[syncArray])
        print(key, 'detPulseNumber', detPulseNumber)
        lpHist, x = np.histogram(detPulseNumber, np.arange((divider+1)))
        print('lpHist', lpHist)

        # lpHist = results[key]['laserPulseHist']['y']
        # print('raw', results[key]['laserPulseHist']['y'])
        # lpHistDiff = np.abs(np.diff(lpHist))
        # print('diff', lpHistDiff)
        lpHistAvg = lpHist.sum()*1./len(lpHist)*1
        # print('Avg', lpHistDiffAvg)
        thresh = (np.max(lpHist) + lpHistAvg)*0.5
        # thresh = np.max(lpHist) * .75
        # print('thresh', thresh)
        lpHistTh = np.where(lpHist > thresh)
        # strt = lpHist[lpHistTh][0]
        # stp = lpHist[lpHistTh][-1]
        print(lpHistTh)
        strt = lpHistTh[0][0]
        stp = lpHistTh[0][-1]
        pcStartStop[key] = [strt, stp]
        # print(key, "pc start stop pulse: ", strt, stp, lpHist[strt], lpHist[stp], np.max(lpHist))
        # print(key, 'optimal', lpHist[129])
    return(pcStartStop)


# @jit
def calc_offset(rawData, paramsCh):

    # print('OFFSET--------')
    DIVIDER = paramsCh['divider']

    results, reducedData = calc_data_properties(
        rawData, paramsCh, findingOffset=True, findPk=True)
    # print('Starting first convolution')
    yalice = results['alice']['syncLaserPulse']
    yalice = yalice - np.average(yalice)
    ybob = results['bob']['syncLaserPulse']
    ybob = ybob - np.average(ybob)
    convolution = scipy.signal.fftconvolve(yalice, ybob[::-1])
    abdelay = np.argmax(convolution) - (len(ybob) - 1)
    print('abdelay', abdelay)

    # # Now find out the offset in laser pulse number between Alice and Bob
    yalice2 = results['alice']['laserPulseHist']['y']
    yalice2 = yalice2 - np.average(yalice2)
    ybob2 = results['bob']['laserPulseHist']['y']
    ybob2 = ybob2 - np.average(ybob2)
    convolution2 = scipy.signal.fftconvolve(yalice2, ybob2[::-1])
    conv2MaxVal = np.max(convolution2)
    offset2Vals = np.where(convolution2 > conv2MaxVal*0.5)[0]
    coincArray = np.zeros(len(offset2Vals))
    corrArray = np.zeros(len(offset2Vals))
    offsetLaserPulseArray = np.zeros(len(offset2Vals))
    print("offset between AB", offset2Vals)

    for i in range(len(offset2Vals)):
        offset2 = offset2Vals[i]
        #
        offsetLaserPulseArray[i] = offset2 - (len(ybob2) - 1)

        corrArray[i] = (abdelay*DIVIDER + offsetLaserPulseArray[i])

        def find_coinc(results, abdelay, corr):
            if abdelay < 0:
                dA = results['alice']['laserPulse2']
                dB = results['bob']['laserPulse2'] + corr
            else:
                dA = results['alice']['laserPulse2'] - corr
                dB = results['bob']['laserPulse2']

            startSync = max(min(dA), min(dB))
            stopSync = min(max(dA), max(dB))
            # print("syncs", startSync, stopSync)

            dABool = (dA >= startSync) & (dA <= stopSync+1)
            dBBool = (dB >= startSync) & (dB <= stopSync+1)

            dA = dA[dABool]
            dB = dB[dBBool]

            coinc = np.intersect1d(dA, dB).astype(int)
            return(coinc)

        coinc = find_coinc(results, abdelay, corrArray[i])
        coincArray[i] = len(coinc)
    print("Coinc Array", coincArray)
    print('')

    ix = np.argmax(coincArray)
    offsetLaserPulse = offsetLaserPulseArray[ix]
    corr = corrArray[ix]
    offset2 = offset2Vals[ix]
    coinc = find_coinc(results, abdelay, corr)

    print("coinc", len(coinc))
    if len(coinc) > 0:
        if abdelay < 0:
            aIndx = np.in1d(results['alice']['laserPulse2'].astype(int), coinc)
            bIndx = np.in1d(
                results['bob']['laserPulse2'].astype(int), coinc - corr)
        else:
            aIndx = np.in1d(
                results['alice']['laserPulse2'].astype(int), coinc + corr)
            bIndx = np.in1d(results['bob']['laserPulse2'].astype(int), coinc)
        aTTag = reducedData['alice']['ttag'][aIndx]
        bTTag = reducedData['bob']['ttag'][bIndx]

        diffTTag = 1.*(aTTag*1. - 1.*bTTag)
        # ttagOffset = mode(diffTTag)[0][0]
        ttagOffset = np.mean(diffTTag*1.)
        print("Coinc", len(coinc), ttagOffset)
        print(len(coinc), ttagOffset, ttagOffset -
              min(diffTTag), ttagOffset - (max(diffTTag)))
    else:
        print("no Coincidences found during Offset search")
        ttagOffset = 0

    print(abdelay, offsetLaserPulse, ttagOffset)
    print('')

    return(abdelay, offsetLaserPulse, ttagOffset)

# @jit


def trim_data(data, ttagOffset, paramsCh, nSyncs=None):
    # if (not data['alice']) or (not data['bob']):
    #     # if there is no data in either dictionary, just return the data
    #     print('Empty detected')
    #     return(data)
    # print("Before Reduced: ", len(data['alice']), len(data['bob']))
    err = False
    syncChAlice = paramsCh['alice']['channels']['sync']
    syncChBob = paramsCh['bob']['channels']['sync']
    # data = get_reduced_data(data, paramsCh)
    # print("After Reduced: ", len(data['alice']), len(data['bob']))
    aData = data['alice']['ttag']*1.  # event number for all alice tags

    # data['bob']['ttag'] = (data['bob']['ttag'])#*1. +ttagOffset*1.)
    bData = data['bob']['ttag']*1. + ttagOffset * \
        1.  # event number for all bob tags, corrected for delay

    aSyncBool = data['alice']['ch'] == syncChAlice
    bSyncBool = data['bob']['ch'] == syncChBob

    aSyncs = aData[aSyncBool]
    bSyncs = bData[bSyncBool]

    # # print(aSyncs, bSyncs)
    # print(aSyncBool, bSyncBool)
    # print('')
    startTTag = max(min(aSyncs), min(bSyncs))
    # print('start ttag', startTTag)
    startTTagA = aSyncs[aSyncs >= startTTag]
    startTTagB = bSyncs[bSyncs >= startTTag]

    indxA = 0
    indxB = 0

    if startTTagA[0] < startTTagB[0]:
        indxB = 0
        if (np.abs(startTTagA[0] - startTTagB[0]) <
                np.abs(startTTagA[1] - startTTagB[0])):
            indxA = 0
        else:
            indxA = 1
    if startTTagA[0] > startTTagB[0]:
        indxA = 0
        if (np.abs(startTTagA[0] - startTTagB[0]) <
                np.abs(startTTagA[0] - startTTagB[1])):
            indxB = 0
        else:
            indxB = 1

    # stopTTag = min (max(aSyncs), max(bSyncs))
    # startTTagA is just a front truncated/trimmed list of syncs for Alice
    stopTTag = min(max(startTTagA), max(startTTagB))
    if nSyncs != None:
        remainingSyncs = min(len(startTTagA[indxA:]), len(startTTagB[indxB:]))
        stopIndx = min(remainingSyncs, nSyncs)
        # print(stopTTag)
        if stopIndx != nSyncs:
            # print('not exact',calc_pc_on_off_slots stopTTag)
            err = True
        else:
            stopTTag = max(startTTagA[nSyncs-1 + indxA],
                           startTTagB[nSyncs-1 + indxB])

    # startTTagA = aSyncs[aSyncs>= startTTag]
    # relevantSyncTTags = startTTagA[startTTagA<=stopTTag]

    # now take the frist element to find the start ttag.
    startTTagA = startTTagA[indxA]
    startTTagB = startTTagB[indxB]

    stopTTagAIndx = np.cumsum(aSyncs <= stopTTag) - 1
    stopTTagA = aSyncs[stopTTagAIndx[-1] - indxA]
    stopTTagBIndx = np.cumsum(bSyncs <= stopTTag) - 1
    stopTTagB = bSyncs[stopTTagBIndx[-1] - indxB]

    aClicks = (aData >= startTTagA) & (aData <= stopTTagA)
    bClicks = (bData >= startTTagB) & (bData <= stopTTagB)
    trimmedData = {'alice': {}, 'bob': {}}

    trimmedData['alice'] = data['alice'][aClicks]
    trimmedData['alice']['ttag'] = trimmedData['alice']['ttag'] - \
        trimmedData['alice']['ttag'][0]  # .astype('float')
    trimmedData['bob'] = data['bob'][bClicks]
    trimmedData['bob']['ttag'] = trimmedData['bob']['ttag'] - \
        trimmedData['bob']['ttag'][0]  # rescaledB#.astype('float')

    # print(trimmedData['alice'])
    # print(trimmedData['bob'])

    return(trimmedData, err)

# @jit


def calc_windows(results, reducedData,
                 offsetLaserPulse, paramsCh,
                 notPC=False):

    logging.basicConfig(level=logging.CRITICAL)
    overlapWindow = {'alice': {}, 'bob': {}}
    aSync = results['alice']['syncArray']  # event number for all alice tags
    bSync = results['bob']['syncArray']

    startSync = max(min(aSync), min(bSync))
    stopSync = min(max(aSync), max(bSync))
    # print('startSync', startSync)
    # print('stopSync', stopSync)
    # startSync = 0
    # print("MINMAX", min(aSync), min(bSync), max(aSync), max(bSync), offsetLaserPulse)
    overlapWindow['alice']['bool'] = (aSync >= startSync) & (aSync <= stopSync)
    overlapWindow['bob']['bool'] = (bSync >= startSync) & (bSync <= (stopSync))

    numSyncs = stopSync - startSync + 1
    # print("in Calc_Windows", numSyncs)

    overlapWindow['nSyncs'] = numSyncs
    overlapWindow['alice']['event'] = np.zeros(numSyncs)
    overlapWindow['bob']['event'] = np.zeros(numSyncs)
    overlapWindow['alice']['laserPulse'] = -np.ones(numSyncs)
    overlapWindow['bob']['laserPulse'] = -np.ones(numSyncs)

    eventCalc = {'alice': [], 'bob': []}

    eventNumber = {}  # Dict w ith list of eventnumber for each detector
    for key in results:
        firstOverlapSync = reducedData[key][overlapWindow[key]['bool']][0]
        results[key]['firstOverlapSync'] = firstOverlapSync
        # print(key, "FIRST OVERLAP SYNC", firstOverlapSync)
        syncBool = reducedData[key]['ch'] == paramsCh[key]['channels']['sync']
        # print(key, syncBool)
        overlapWindow[key]['syncBool'] = overlapWindow[key]['bool'] & syncBool
        overlapWindow[key]['syncTags'] = reducedData[key][overlapWindow[key]
                                                          ['bool'] & syncBool]
        eventBool = (overlapWindow[key]['bool'] &
                     results[key]['clickBool'] &
                     results[key]['inWindowBool'])
        if notPC:  # only look for events outside the PC Window
            eventCalc[key] = eventBool & overlapWindow[key]['bool']
        else:
            eventCalc[key] = eventBool
        overlapWindow[key]['eventBool'] = eventBool
        singles = eventBool.sum()
        delay = 0

        # corr = (abdelay*DIVIDER + offsetLaserPulse)
        # print()
        # Need to subtract startSync if >0
        eventSyncNumber = results[key]['syncArray']+delay-startSync
        if notPC:
            eventNumber[key] = (results[key]['laserPulseNotPC'] +
                                1*(eventSyncNumber-1) * int(
                                    results[key]['divder']) +
                                1 * offsetLaserPulse[key]).astype(int)
        else:
            eventNumber[key] = (results[key]['laserPulse']
                                + 1*(eventSyncNumber-1) * int(
                                    results[key]['divder']) +
                                1 * offsetLaserPulse[key]).astype(int)
        overlapWindow[key]['eventBool'] = eventBool
        overlapWindow[key]['eventSyncNumber'] = eventSyncNumber

        #  Try to map the settings from the raw data to the data that overlaps
        #    if abdelay is negative, than bob timetags are earlier... Need to start bob settings later
        #    if abdelay is positive, alice is earlier... need to start alice settings later
        start = 0
        # abdelay = 0
        # if abdelay < 0:
        #     if key=='bob':
        #         start = -abdelay
        #         # start -= 1  #  I think the setting is offset by one sync cycle
        # else:
        #     if key=='alice':
        #         start = abdelay
        stop = start + numSyncs
        overlapWindow[key]['settings'] = results[key]['settings'][start:stop]

    # print('A: ' , eventNumber['alice'][overlapWindow['alice']['eventBool']][0:20])
    # print('B: ' , eventNumber['bob'][overlapWindow['bob']['eventBool']][0:20])
    # print('')
    coinEventNum = np.intersect1d(
        eventNumber['alice'][overlapWindow['alice']['eventBool']],
        eventNumber['bob'][overlapWindow['bob']['eventBool']])
    # print("COINC: ", len(coinEventNum))
    counts = {'sAlice': eventCalc['alice'].sum(),
              'sBob': eventCalc['bob'].sum(),
              'coinc': len(coinEventNum)}

    # counts = {'sAlice': len(reducedData['alice'][reducedData['alice']['ch']==0]),
    #             'sBob': len(reducedData['bob'][reducedData['bob']['ch']==0]),
    #             'coinc': len(coinEventNum)}
    # print("counts", counts)
    return(overlapWindow, eventNumber, numSyncs, startSync, stopSync, counts)

# @jit


def get_stats(results, reducedData, offsetLaserPulse,
              abdelay, pockelProp, paramsCh, overlapWindow=None,
              numSyncs=None, PC=False):
    # reset definitions of event and laserPulse in the overlapping timetags
    #   include will now include boolean for pockelCell timing
    if (overlapWindow is None) | (numSyncs is None):
        (overlapWindow, eventNumber, numSyncs,
         startSync, stopSync, counts) = calc_windows(results,
                                                     reducedData,
                                                     offsetLaserPulse,
                                                     paramsCh)
    pcOverlapWindow = {'alice': {}, 'bob': {}}
    pcOverlapWindow['alice']['eventCode'] = np.zeros(numSyncs)
    pcOverlapWindow['bob']['eventCode'] = np.zeros(numSyncs)
    # pcOverlapWindow['counts'] = counts
    pcSlots = np.asarray(pockelProp['slots'])
    mask = (2**pcSlots).sum().astype('int')
    pcEventNumber = {}
    pcEvents = {}

    if PC:
        for key in results:

            pcStart = pockelProp['start'] - offsetLaserPulse[key]

            # [overlapWindow[key]['eventBool']]
            laserPulse = results[key]['laserPulse']

            pockelBool = (laserPulse >= pcStart) & (
                laserPulse <= (pcStart+pockelProp['length']))
            PCeventBool = overlapWindow[key]['eventBool'] & pockelBool

            valTest = 2**((laserPulse * PCeventBool.astype(int)
                           ) - pcStart * PCeventBool.astype(int))
            valTest[np.invert(PCeventBool)] = 0

            laserPulseSum = np.cumsum(valTest)
            trialSum = laserPulseSum[overlapWindow[key]['syncBool']]
            trialVal = np.diff(trialSum)

            # Shift pulse 1 so that the labeling now starts a 1, not 0. Eventcode for first pulse is 2.
            # For pulse 2 it is 4, and so on.
            # pcOverlapWindow[key]['eventCode'] = (1<<(trialVal.astype(int) - pcStart)) & mask
            pcOverlapWindow[key]['eventCode'] = trialVal.astype(int) & mask
            pcOverlapWindow[key]['settings'] = np.delete(
                overlapWindow[key]['settings'], -1)

            pcEventNumber[key] = pcOverlapWindow[key]['eventCode'] > 0

        coinEventNumPC =\
            np.intersect1d(np.nonzero(pcEventNumber['alice'])[0],
                           np.nonzero(pcEventNumber['bob'])[0])
        totalSettings = pcOverlapWindow['alice']['settings'] + \
            4*pcOverlapWindow['bob']['settings']
        pcEventNumber['coin'] = coinEventNumPC
        bins = np.arange(-0.5, 16)
        for key in pcEventNumber:
            idx = pcEventNumber[key]
            n, x = np.histogram(totalSettings[idx], bins)
            n.shape = (4, 4)
            # print(key, n)
            pcEvents[key] = n
        pcEvents['numSyncs'] = numSyncs
    # print(pcEvents)
    return(pcEvents, pcOverlapWindow)

# @jit


def calc_violation(stats):
    #  J = P(++|ab)        - P(+0|a'b)           - P(0+|ab')         - P(++|a'b')
    Jparts = [stats['coin'][1, 1], -stats['alice'][2, 1]+stats['coin']
              [2, 1], -stats['bob'][1, 2]+stats['coin'][1, 2], -stats['coin'][2, 2]]
    #print('Jparts: %r'%Jparts)
    J = stats['coin'][1, 1] - stats['alice'][2, 1] - \
        stats['bob'][1, 2] - stats['coin'][2, 2]
    #print('J: %d'%sum(Jparts))

    CH_single = (stats['alice'][1, 1]*1. + stats['alice']
                 [2, 1] + stats['bob'][1, 1] + stats['bob'][1, 2])/2.
    CH_coin = stats['coin'][1, 1] + stats['coin'][2, 1] + \
        stats['coin'][1, 2] - stats['coin'][2, 2]
    CH = CH_coin - CH_single
    CHn = CH_coin/CH_single
    # print("CH violation:", CH)
    # print("CH normalized:", CHn)

    Pab11 = stats['coin'][1, 1]
    Papb10 = stats['alice'][2, 1]-stats['coin'][2, 1]
    Pabp01 = stats['bob'][1, 2]-stats['coin'][1, 2]
    Papbp11 = stats['coin'][2, 2]

    Ntot = Pab11 + Papb10 + Pabp01 + Papbp11*1.
    ratio = Pab11*1./Ntot

    pValue = calc_pvalue(Ntot, ratio)

    # print("Ratio:", ratio, "pValue:", pValue)
    return(CH, CHn, ratio, pValue)

# @jit


def calc_pvalue(N, prob):
    return(binom.cdf(N*(1-prob), N, 0.5))
