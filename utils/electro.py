from typing import List, Union
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
        -> float or (float, float):
    assert sampling_rate > 0

    signal_avg = sum(data) / len(data)

    # data[i - 1] * data[i] < 0 only if one sample point is negative and the other one is positive
    zero_crossings = [i for i in range(1, len(data)) if (data[i - 1] - signal_avg) * (data[i] - signal_avg) < 0.0]
    if len(zero_crossings) < 2:
        return None

    def abs_interpolate(a, b):
        return abs(a) / (abs(a) + abs(b))

    start = zero_crossings[0]
    end = zero_crossings[-1]
    # interpolate at what position the signal crosses the zero/avg
    start = start - abs_interpolate(data[start] - signal_avg, data[start - 1] - signal_avg)
    end = end - abs_interpolate(data[end] - signal_avg, data[end - 1] - signal_avg)

    half_wave_length_avg = (end - start) / (len(zero_crossings) - 1) / sampling_rate
    frequency = 1.0 / (2 * half_wave_length_avg)
    if not calc_offset:
        return frequency
    offset = 0
    return frequency, offset


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

    # auto_change = [auto[i+1] - auto[i] for i in range(0, len(auto)-1)]

    # import matplotlib.pyplot as plt
    # f, axarr = plt.subplots(2, sharex=True)
    # axarr[0].plot(samples)
    # axarr[1].plot(auto)
    # axarr[1].grid(True)
    # plt.show()

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

    # funktioniert nur wenn hauptfrequenz ausreichend domninat ist
    # sonst werden unter umständen andere periodische signale gemessen:
    # first_min_index = np.argmin(auto)
    # second_max_index = first_min_index + np.argmax(auto[first_min_index:])
    # return sampling_rate / interpolate(auto, second_max_index, mode)


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
               ampere_data: List[float],
               sampling_rate: float):
    assert len(voltage_data) == len(ampere_data) > 0

    # P = 0  # active power/real power (Wirkleistung)
    # Q = 0  # reactive power (blindleistung)
    # S = 0  # complex power/apparent power (Scheinleistung)
    # S = math.sqrt(Q**2 + P**2)  # apparent power
    # return S, P, Q

    # https://electronics.stackexchange.com/questions/199395/how-to-calculate-instantaneous-active-power-from-sampled-values-of-voltage-and-c/199401#199401
    instantaneous_power = [v*a for v, a in zip(voltage_data, ampere_data)]

    P = abs(np.mean(instantaneous_power))
    S = rms(voltage_data) * rms(ampere_data)

    Q = sqrt(S**2 - P**2)
    # power_factor = P / S
    return P, Q, S

# voltage_frequency = measure_main_frequency_autocorrelate(voltage_data, sampling_rate)
# offset = measure_offset_correlate(voltage_data, ampere_data, sampling_rate)
# offset %= 1/voltage_frequency
#
# print(f"offset(xcor): {offset*1000:6.2f} ms / {360 * offset/(1/voltage_frequency):8.3f} °")
# print(f"offset(sign): {'-':^6} ms / {measure_offset_signum(voltage_data, ampere_data):8.3f} °")
# print(f"offset(fft2): {'-':^6} ms / {measure_offset_fft(voltage_data, ampere_data):8.3f}°")
# # S_avg = sum(u * (i * ampere_factor) for u, i in zip(voltage_data, ampere_data)) / len(voltage_data)
# # return S_avg
