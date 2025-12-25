from ...binaries import np
from numba import njit

def filter_and_downsample(pmts: np.ndarray, random_phase=0) -> np.ndarray:

    if len(pmts) == 2048:
        return filter_and_downsample_pmt(pmts)
    else:
        filtered_and_downsampled = np.empty(shape=(len(pmts), 683))
        for i, pmt in enumerate(pmts):
            filtered_and_downsampled[i, ] = filter_and_downsample_pmt(pmt)

        return filtered_and_downsampled

@njit
def filter_and_downsample_pmt(pmt: np.ndarray, random_phase=0) -> np.ndarray:
    """Convert UUB trace to UB compatibility trace"""

    BINS = 2048

    kFirCoefficients = np.array([
        5, 0, 12, 22, 0, -61, -96,
        0, 256, 551, 681, 551, 256, 0,
        -96, -61, 0, 22, 12, 0, 5,
    ], dtype=np.int32)

    kFirLen = kFirCoefficients.size
    buffer_length = kFirLen // 2
    kFirNormalizationBitShift = 11
    kADCSaturation = 4095

    temp = np.zeros(BINS + kFirLen, dtype=np.uint16)

    pmt_rev = pmt[::-1].copy()

    for i in range(buffer_length):
        temp[i] = pmt_rev[buffer_length - i]

    for i in range(buffer_length):
        temp[BINS + kFirLen - 1 - buffer_length + i] = pmt_rev[i]

    for i in range(BINS):
        temp[buffer_length + i] = pmt[i]

    temp_shifted = np.zeros((kFirLen, BINS), dtype=np.int32)

    for k in range(kFirLen):
        for j in range(BINS):
            temp_shifted[k, j] = temp[k + j]

    for k in range(kFirLen):
        coef = kFirCoefficients[k]
        for j in range(BINS):
            temp_shifted[k, j] *= coef
            
    acc = np.zeros(BINS, dtype=np.int32)
    for k in range(kFirLen):
        for j in range(BINS):
            acc[j] += temp_shifted[k, j]

    out = np.zeros(BINS, dtype=np.uint16)
    for j in range(BINS):
        val = acc[j] >> kFirNormalizationBitShift
        if val < 0:
            val = 0
        elif val > kADCSaturation:
            val = kADCSaturation
        out[j] = val

    return out[random_phase::3]

@njit
def T2_th(wcd_traces: np.ndarray, threshold: float = 3.2) -> bool:
    """T2 threshold trigger for the WCD pmts"""
    return (wcd_traces > threshold).all(axis=1).any()

@njit
def T2_tot(wcd_traces: np.ndarray, 
           threshold: float = 0.2, 
           occupancy: int = 12,
           multiplicity: int = 2,
           window_size: int = 120) -> bool:
    """T2 time over threshold trigger for the WCD pmts"""

    counts = np.cumsum(wcd_traces > threshold, axis=1)
    running_counts = counts[:, window_size-1:] - counts[:, :-120+1]
    window_occupancy = np.sum(running_counts > multiplicity, axis=0)

    return (window_occupancy > occupancy).any()

