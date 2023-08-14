from dataclasses import dataclass
import numpy as np


@dataclass
class TimeTags:
    #     """Class for keeping track of an item in inventory."""
    name: str
    sync_time_tags: np.ndarray 
    detector_ttags: list 
    setting_ttags: list 

    quantity_on_hand: int = 0

def time_tag_factory(name: str, data, detector_channels: list, setting_channels: list, sync_channel: int, int, divider:int) -> TimeTags
    props = calc_data_properties_one_party(data, params, divider,
                                   findPk=False, party='')

    
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


'''
Needs:
A data structure that, for a single timetagger, holds the data + props information. 
This should be immutable? 
data: ((ch,timetag)) data structure
masks: (settings, click, laserpulse, sync)
trials: (Syncarray, settings_array,) // should be another function? Pipe data into this function to return a 
trial object?

1. Get data for each timetagger
2. Trim data 
3. Calculate properties for each channel/each time tagger
4. Create masks
5. Apply masks and get singles/coincidences

Need another dataclass for reduced data? This would be a list of what happened in each trial.
The reduced data would only apply to a single party.
Setting, outcome_det_1, outcome_det_2, ...

Timetaggers -> fetch_data_from_two_timetaggers
Each timetagger returns raw data in the expected numpy structure
Trim raw data
For each piece of rawdata from each timetagger create datatype that holds data & mask information
Pipe data along with masks into coincidience calculation functions
pipe data along with pc mask into function to return reduced data
compress data
compute stats

May need to refactor/simplfiy the timetaggers class.
Just have it return the trimmed raw data from each timetagger 

Then have a function that processes this data by piping it as needed. 
timetaggers -trimmed data-> create_data_structure_with_masks -ttag_structure->
compute_joint_statistics (coincs, histograms, etc..)
or
->fetch_reduced_trial_data -reduced_data_structure-> compress_data / compute_freq_table

This would allow the same code to be used to saving/reading.

class TTagChannel(name:str, ch:int):
    pass

class CoincidenceWindow:
    coincidence_window_radius: int
    coincidence_window_peak: int
    find_coincidence_window_peak: bool

class PockelsCellParameters:
    start_pulse: int
    number_of_pulses: int
    offset_laser_pulses: int = 0

class TTagParameters:
    sync: TTagChannel
    detectors: list[TTagChannel]
    settings: list[TTagChannel]
    coincidence_window: CoincidenceWindow
    divider: int

class Mask:
    name: str
    value: int
    mask: np.ndarray

class SyncTTagData:
    sync_channel: TTagChannel
    data: nd.array
    sync_mask: nd.array 
    sync_index_for_all_trials: nd.array
    laser_period: float
    number_trials: int


class SettingData:
    settings_channel: TTagChannel
    sync_data: SyncTTagData
    settings_mask: Mask
    settings_value_for_all_trials: Mask


class DetectorTTagData:
    detector_channel: TTagChannel
    sync_data: SyncTTagData
    data: nd.array
    # Compute post initialization
    detection_mask: Mask
    detection_in_coincidence_window_mask: Mask
    laser_pulse_mask: Mask 
    detections_labeled_by_laser_pulse_number: nd.array

class TTagData:
    data: np.ndarray
    ttager_parameters: ttagParameters
    sync_data: SyncTTagData
    settings: list[SettingData]
    detectors: list[DetectorTTagData]

    # # Compute post initialization
    # sync_mask: nd.array 
    # sync_index_for_all_trials: nd.array
    # number_trials: int
    # settings_mask: list[mask]
    # settings_value_for_all_trials: list[mask]
    # detector_mask: list[mask]
    # detector_in_coincidence_window_mask: list[mask]
    # laser_pulse_mask: list[mask] 
    # detections_labeled_by_laser_pulse_number: list[nd.array]
    # laser_period: int

class TTagHistogram:
    low_value: int
    high_value: int
    histogram_x: np.ndarray
    histogram_y: np.ndarray 

class TTagCount:
    name: str
    counts: int

def factory_TTagParameters(parameters:dict): ->TTagParameters
    pass


def factory_TTagData(data: np.ndarray, ttager_parameters: ttagParameters) -> TTagData
    1. Compute sync properties
    2. Compute settings properties
    3. Compute detector properties

###################################
Example using initVar from DataClass. Could solve the data initialization problem.
from dataclasses import dataclass, field, InitVar
from typing import List
https://www.infoworld.com/article/3563878/how-to-use-python-dataclasses.html

@dataclass
class Book:
    name: str     
    condition: InitVar[str] = "Good"
    weight: float = field(default=0.0, repr=False)
    shelf_id: int = field(init=False)
    chapters: List[str] = field(default_factory=list)

    def __post_init__(self, condition):
        if condition == "Unacceptable":
            self.shelf_id = None
        else:
            self.shelf_id = 0
'''