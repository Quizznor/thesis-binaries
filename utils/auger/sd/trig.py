from ...binaries import np

def filter_and_downsample(*traces: np.ndarray, random_phase: int = 1) -> np.array:
    """convert UUB trace to UB equivalent, for compatibility mode"""

    filtered_and_downsampled_traces = []

    # see Framework/SDetector/UUBDownsampleFilter.h in Offline main branch for more information
    kFirCoefficients = np.array(
        [
            5,
            0,
            12,
            22,
            0,
            -61,
            -96,
            0,
            256,
            551,
            681,
            551,
            256,
            0,
            -96,
            -61,
            0,
            22,
            12,
            0,
            5,
        ]
    )
    buffer_length = int(0.5 * len(kFirCoefficients))
    kFirNormalizationBitShift = 11
    kADCSaturation = 4095
    kFirLen = len(kFirCoefficients)

    for pmt in traces:
        temp = np.zeros(len(pmt) + len(kFirCoefficients), dtype=int)
        temp[0:buffer_length] = pmt[::-1][-buffer_length - 1 : -1]
        temp[-buffer_length - 1 : -1] = pmt[::-1][0:buffer_length]
        temp[buffer_length : -buffer_length - 1] = pmt

        temp_shifted = np.array([temp[k : k + len(pmt)] for k in range(kFirLen)])
        outer_product = temp_shifted * kFirCoefficients[:, np.newaxis]

        trace = np.sum(outer_product, axis=0)
        trace = np.clip(
            np.right_shift(trace, kFirNormalizationBitShift), 0, kADCSaturation
        )

        trace = trace[random_phase::3]
        filtered_and_downsampled_traces.append(np.array(trace, dtype="u2"))

    return np.array(filtered_and_downsampled_traces)

def T2_th(wcd_traces: np.ndarray, threshold: float = 3.2) -> bool:
    """T2 threshold trigger for the WCD pmts"""
    return (wcd_traces > threshold).all(axis=1).any()

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

