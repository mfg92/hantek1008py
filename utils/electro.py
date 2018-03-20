from typing import List, Union, Tuple, Optional
from math import sqrt, log
import numpy as np

"""
EeMt: Einführung in die elektrische Messtechnik, 3. Auflage
"""


def rms(data: Union[List[float], np.ndarray]) -> float:
    """
    Calculates the 'Root Mean Square' (Effektivwert)
    EeMt 5.2.3 P. 109
    :param data:
    :return:
    """
    return np.sqrt(np.mean(np.array(data)**2))
    # return sqrt(sum(v*v for v in data)/len(data))


def interpolate(samples: Union[List[float], np.ndarray], index: int, mode: str) -> float:
    """
    This is no normal interpolation (find a value between two sample points) instead this finds the
    position of a peak
    :param samples:
    :param index:
    :param mode:
    :return:
    """
    assert mode in {"none", "parabolic", "gaussian"}
    if mode == "parabolic":
        return parabolic_interpolation(samples, index)
    elif mode == "gaussian":
        return gaussian_interpolation(samples, index)
    return index


def parabolic_interpolation(data: List[float], local_max_index: int) -> float:
    assert 0 <= local_max_index <= len(data)-1
    i_max = local_max_index
    if i_max == 0 or i_max == len(data)-1:
        return data[i_max]
    v_max = data[i_max]
    v_left = data[local_max_index - 1]
    v_right = data[local_max_index + 1]
    return i_max + (v_right - v_left)/(2*(2*v_max - v_right - v_left))


def gaussian_interpolation(data: List[float], local_max_index: int) -> float:
    """
    Like parabolic_interpolation but the natural logarithm of the values is used for interpolation
    :param data:
    :param local_max_index:
    :return:
    """
    neigbours_ln = [log(v) for v in data[local_max_index-1:local_max_index+2]]
    return local_max_index + parabolic_interpolation(neigbours_ln, 1)


def measure_main_frequency_zero_crossing(data: List[float], sampling_rate: float, calc_offset: bool = False) \
        -> Optional[float]:
    assert sampling_rate > 0
    assert not calc_offset, "Offset calculation is not supported yet"

    data_dc = np.mean(data)
    data_ac = [x - data_dc for x in data]  # remove dc part of signal

    # data_ac[i - 1] * data_ac[i] < 0 only if one sample point is negative and the other one is positive
    zero_crossings = [i for i in range(1, len(data_ac)) if data_ac[i - 1] * data_ac[i] < 0.0]

    if len(zero_crossings) < 2:
        return None

    def abs_interpolate(a, b):
        return abs(a) / (abs(a) + abs(b))

    start = zero_crossings[0]
    end = zero_crossings[-1]
    # interpolate at what position the signal crosses the zero/avg
    start = start - abs_interpolate(data_ac[start], data_ac[start - 1])
    end   = end   - abs_interpolate(data_ac[end], data_ac[end - 1])

    half_wave_length_avg = (end - start) / (len(zero_crossings) - 1) / sampling_rate
    frequency = 1.0 / (2 * half_wave_length_avg)

    # delta = (end - start) * (1.0 / sampling_rate)
    # periods = (len(zero_crossings) - 1) / 2
    # frequency = 1.0 / (delta / periods)
    return frequency


def measure_main_frequency_fft(samples: List[float], sampling_rate: float, mode: str = "parabolic") -> float:
    fourier = np.fft.rfft(samples * np.blackman(len(samples)))
    # blackman is better for main frequency estimation using parabolic or Gaussian interpolation
    # according to FFT_resol_note.pdf (IMPROVING FFT FREQUENCY MEASUREMENT RESOLUTION BY PARABOLIC
    # AND GAUSSIAN INTERPOLATION)

    fourier_amplitude = np.absolute(fourier)
    fourier_frequency = np.fft.rfftfreq(n=len(samples), d=1.0 / sampling_rate)
    fourier_frequency_step_width = fourier_frequency[1]
    # get the highest value (+ index of that)
    max_index, _ = max(enumerate(fourier_amplitude), key=lambda v: v[1])
    return interpolate(fourier_amplitude, max_index, mode) * fourier_frequency_step_width


# https://de.wikipedia.org/wiki/Autokorrelation#Finden_von_Signalperioden
# https://stackoverflow.com/questions/13439718/how-to-interpret-numpy-correlate-and-numpy-corrcoef-values/37886856#37886856
def measure_main_frequency_autocorrelate(samples: List[float], sampling_rate: float,  mode: str = "parabolic") -> float:
    assert mode in {"max", "parabolic", "gaussian"}
    auto = np.correlate(samples, samples, mode="full")
    auto = auto[round(len(auto) / 2):]

    def find_local_peak(data: List[float], from_pos: int):
        for i in range(from_pos+1, len(data)-1):
            l, m, r = data[i-1:i+2]
            if l < m > r or l > m < r:
                return i
        return None

    first_min_index = find_local_peak(auto, 0)
    if first_min_index is None:
        return -1
    second_max_index = find_local_peak(auto, first_min_index)
    if second_max_index is None:
        return -1
    return sampling_rate / interpolate(auto, second_max_index, mode)


# for phase angles between 0 and 180 degrees
# https://dsp.stackexchange.com/questions/8673/best-method-to-extract-phase-shift-between-2-sinosoids-from-data-provided/26012#26012
def measure_offset_signum(samples_a: List[float], samples_b: List[float]) -> float:
    signum_a = np.sign(samples_a - np.mean(samples_a))
    signum_b = np.sign(samples_b - np.mean(samples_b))
    return 90 - 90*(np.mean(signum_a*signum_b))  # offset in degrees [0-180]


def measure_offset_correlate(samples_a: List[float], samples_b: List[float],
                             sampling_rate: float,  mode: str = "parabolic") -> float:
    auto = np.correlate(samples_a, samples_b, mode="full")
    best_offset = np.argmax(auto)
    return interpolate(auto, best_offset, mode) * (1/sampling_rate)  # offset in sec


# https://stackoverflow.com/questions/27545171/identifying-phase-shift-between-signals/27546385#27546385
def measure_offset_fft(samples_a: List[float],
                       samples_b: List[float],
                       mode: str = "parabolic") -> float:
    assert mode in {"max", "parabolic", "gaussian"}
    fourier_a, fourier_b = np.fft.rfft(samples_a), np.fft.rfft(samples_b)
    max_index, _ = max(enumerate(np.absolute(fourier_a)), key=lambda v: v[1])
    return np.angle(fourier_a[max_index] / fourier_b[max_index], deg=True)  # offset in degrees [-180, 180]


def calc_power(voltage_data: List[float],
               ampere_data: List[float])\
        -> Tuple[float, float, float]:
    assert len(voltage_data) == len(ampere_data) > 0
    # P = 0  # active power/real power      (Wirkleistung)
    # Q = 0  # reactive power               (Blindleistung)
    # S = 0  # complex power/apparent power (Scheinleistung)

    # https://electronics.stackexchange.com/questions/199395/how-to-calculate-instantaneous-active-power-from-sampled-values-of-voltage-and-c/199401#199401
    instantaneous_power = [v*a for v, a in zip(voltage_data, ampere_data)]

    P = np.mean(instantaneous_power)
    S = rms(voltage_data) * rms(ampere_data)
    Q = sqrt(S**2 - P**2)
    # power_factor = P / S
    return P, Q, S


# TODO name is not adequate
# WARNING: Experimental
def measure_main_frequencies_fft(samples: List[float],
                                 sampling_rate: float,
                                 freqeuency_search_count: int = 9,
                                 frqeuency_distinction_range: float = 2,  # in Hz
                                 mode: str = "parabolic")\
        -> List[Tuple[float, float]]:
    fourier = np.fft.rfft(samples * np.blackman(len(samples)))
    # blackman is better for main frequency estimation using parabolic or Gaussian interpolation
    # according to FFT_resol_note.pdf (IMPROVING FFT FREQUENCY MEASUREMENT RESOLUTION BY PARABOLIC
    # AND GAUSSIAN INTERPOLATION)

    fourier_amplitude = np.absolute(fourier)
    fourier_frequency = np.fft.rfftfreq(n=len(samples), d=1.0 / sampling_rate)
    fourier_frequency_step_width = fourier_frequency[1]

    # import matplotlib.pyplot as plt
    # plt.plot(fourier_frequency, fourier_amplitude)
    # plt.grid()
    # plt.show()

    frequencies = []
    for harmonic_count in range(freqeuency_search_count):
        # get the highest value (+ index of that)
        max_index, max_value = max(enumerate(fourier_amplitude), key=lambda v: v[1])
        max_index_interpolated = interpolate(fourier_amplitude, max_index, mode)
        max_value_interpolated = max_value  # TODO interpolate
        frequencies.append((max_index_interpolated * fourier_frequency_step_width, max_value_interpolated))

        # "remove" that frequency from the FFT
        half_fdr_as_index_size = (frqeuency_distinction_range/fourier_frequency_step_width)/2
        frequency_range_left = max(0, int(max_index_interpolated - half_fdr_as_index_size))
        frequency_range_right = min(len(fourier_amplitude)-1, int(max_index_interpolated + half_fdr_as_index_size))
        fourier_amplitude[frequency_range_left:frequency_range_right] = [0] * (frequency_range_right - frequency_range_left)
        print("fft step width", fourier_frequency_step_width)
        print("frequency_range_left", frequency_range_left)
        print("frequency_range_right", frequency_range_right)

    return frequencies

# TODO name is not adequate
# WARNING: Experimental
# def measure_harmonics_fft(samples: List[float], sampling_rate: float,
#                           harmonics_search_count: int = 9, mode: str = "parabolic")\
#         -> List[Tuple[float, float]]:
#     fourier = np.fft.rfft(samples * np.blackman(len(samples)))
#     # blackman is better for main frequency estimation using parabolic or Gaussian interpolation
#     # according to FFT_resol_note.pdf (IMPROVING FFT FREQUENCY MEASUREMENT RESOLUTION BY PARABOLIC
#     # AND GAUSSIAN INTERPOLATION)
#
#     fourier_amplitude = np.absolute(fourier)
#     fourier_frequency = np.fft.rfftfreq(n=len(samples), d=1.0 / sampling_rate)
#     fourier_frequency_step_width = fourier_frequency[1]
#
#     import matplotlib.pyplot as plt
#     plt.plot(fourier_frequency, fourier_amplitude)
#     plt.grid()
#     plt.show()
#
#     harmonics = []
#     index_of_harmonic_0 = 0
#     previous_max_index = 0
#     for harmonic_count in range(harmonics_search_count):
#         # get the highest value (+ index of that)
#         search_start_index = index_of_harmonic_0 * harmonic_count
#         max_index, max_value = max(enumerate(fourier_amplitude[search_start_index:]), key=lambda v: v[1])
#         max_index_interpolated = interpolate(fourier_amplitude, max_index, mode)
#         max_value_interpolated = max_value  # TODO interpolate
#         harmonics.append(max_index_interpolated * fourier_frequency_step_width, max_value_interpolated)
#         previous_max_index = max_index
#         if index_of_harmonic_0 == 0:
#             index_of_harmonic_0 = max_index_interpolated
#
#     return harmonics

