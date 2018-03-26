import usb.core
import usb.util
import usb.backend
import time
from time import sleep
from typing import Union, Optional, List, Dict, Tuple, Callable, Generator
import logging as log
import math
from threading import Thread
import copy
import sys

# marking a child class method with overrides makes sure the method overrides a parent class method.
# this check is only needed during development so its no problem if this package is not installed.
# to avoid errors, we need to define a dummy decorator.
try:
    from overrides import overrides
except ImportError:
    def overrides(method: Callable) -> Callable:
        return method

assert sys.version_info >= (3, 6)

"""
 To get access to the USB Device:

 1. create file "/etc/udev/rules.d/99-hantek1008.rules" with content:
    ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="0783", ATTR{idProduct}=="5725", MODE="0666"
 2. sudo udevadm control -R
 3. Replug the device
"""


class Hantek1008Raw:
    """
    This class communicates to a Hantek1008 device via USB.
    It supports configuring the device (set vertical scale, sampling frequency, waveform generator,..)
    and measuring samples with it. Either in continuous (rolling) mode or in windows (normal/burst) mode.
    """
    # channel_id/channel_index are zero based
    # channel names are one based

    __MAX_PACKAGE_SIZE: int = 64
    __VSCALE_FACTORS: List[float] = [0.02, 0.125, 1.0]
    __roll_mode_sampling_rate_to_id_dic: Dict[float, int] = \
        {440: 0x18, 220: 0x19, 88: 0x1a, 44: 0x1b,
         22: 0x1c, 11: 0x1d, 5: 0x1e, 2: 0x1f,
         1: 0x20, 0.5: 0x21, 0.25: 0x22, 0.125: 0x23,
         1.0/16: 0x24}
    # ids for all valid nanoseconds per div. These ns_per_divs have following pattern:  (1|2|3){0}
    # eg. 10, 2000 or 5. Maximum is 200_000_000
    # a div contains around 25 samples
    __burst_mode_ns_per_div_to_id_dic = {({0: 1, 1: 2, 2: 5}[id % 3] * 10 ** (id // 3)): id for id in range(26)}

    def __init__(self, ns_per_div: int = 500_000,
                 vertical_scale_factor: Union[float, List[float]] = 1.0,
                 active_channels: Optional[List[int]] = None,
                 trigger_channel: int = 0,
                 trigger_slope: str = "rising",
                 trigger_level: int = 2048
                 ) -> None:
        """
        :param ns_per_div:
        :param vertical_scale_factor: must be an array of length 8 with a float scale value for each channel
               or a single float scale factor applied to all channels. The float must be either 1.0, 0.2 or 0.02.
        :param active_channels: a list of channel that will be used
        """

        assert isinstance(vertical_scale_factor, float) \
               or len(vertical_scale_factor) == Hantek1008Raw.channel_count()

        self.__ns_per_div: int = ns_per_div  # one value for all channels

        self.__active_channels: List[int] = copy.deepcopy(active_channels) if active_channels is not None\
            else Hantek1008Raw.valid_channel_ids()
        self.__active_channels = sorted(self.__active_channels)  # some methods depend of ascending order of this

        # one vertical scale factor (float) per channel
        self.__vertical_scale_factors: List[float] = [vertical_scale_factor] * Hantek1008Raw.channel_count() \
            if isinstance(vertical_scale_factor, float) \
            else copy.deepcopy(vertical_scale_factor)  # scale factor per channel

        self.__trigger_channel: int = trigger_channel
        self.__trigger_slope: str = trigger_slope
        self.__trigger_level: int = trigger_level

        # dict of list of floats, outer dict is of size 3 and contains values
        # for every vertical scale factor, inner list contains an zero offset per channel
        self._zero_offsets: Optional[Dict[float, List[float]]] = None

        self.__out: usb.core.Endpoint = None  # the usb out endpoint
        self.__in: usb.core.Endpoint = None  # the usb in endpoint
        self._dev: usb.core.Device = None  # the usb device
        self._cfg: usb.core.Configuration = None  # the used usb configuration
        self._intf: usb.core.Interface = None  # the used usb interface

        self.__pause_thread: Optional[Thread] = None
        self.__cancel_pause_thread: bool = False

    def connect(self) -> None:
        """Find a plugged in hantek 1008c device and set up the connection to it"""

        self._dev = usb.core.find(idVendor=0x0783, idProduct=0x5725)

        # was it found?
        if self._dev is None:
            raise RuntimeError('No Hantek 1008 device found')

        # set the active configuration. With no arguments, the first
        # configuration will be the active one
        self._dev.set_configuration()

        self._cfg = self._dev.get_active_configuration()
        self._intf = self._cfg[(0, 0)]

        # get an output endpoint instance
        self.__out = usb.util.find_descriptor(
            self._intf,
            # match the first OUT endpoint
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT)

        # get an input endpoint instance
        self.__in = usb.util.find_descriptor(
            self._intf,
            # match the first IN endpoint
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN)

        assert self.__out is not None
        assert self.__in is not None

    def __write_and_receive(self, message: bytes, response_length: int,
                            sec_till_response_request: float = 0.002, sec_till_start: float = 0.002) -> bytes:
        """write to and read from the device"""
        start_time = time.time()

        assert isinstance(message, bytes)
        log.debug(f">[{len(message):2}] {bytes.hex(message)}")

        sleep(sec_till_start)

        self.__out.write(message)

        sleep(sec_till_response_request)

        response = bytes(self.__in.read(response_length))

        log.debug(f"<[{len(response):2}] {bytes.hex(response)}")
        log.debug(f"delta: {time.time()-start_time:02.4f} sec")
        assert len(response) == response_length

        return response

    def __send_cmd(self, cmd_id: int, parameter: Union[bytes, List[int], str] = b'',
                   response_length: int = 0, echo_expected: bool = True,
                   sec_till_response_request: float = 0, sec_till_start: float = 0.002) -> bytes:
        """sends a command to the device and checks if the device echos the command id"""
        if isinstance(parameter, str):
            parameter = bytes.fromhex(parameter)
        elif isinstance(parameter, list):
            parameter = bytes(parameter)
        assert isinstance(parameter, bytes)
        assert 0 <= cmd_id <= 255

        msg = bytes([cmd_id]) + parameter
        response = self.__write_and_receive(msg, response_length + (1 if echo_expected else 0),
                                            sec_till_response_request=sec_till_response_request,
                                            sec_till_start=sec_till_start)
        if echo_expected:
            assert response[0] == cmd_id
            return response[1:]
        else:
            return response

    def __send_c6_a6_command(self, parameter: int) -> bytes:
        """send the c602 or c603 command, then parse the response as sample_length. then CEIL(sample_length/64)
        a602 or a603 requests follow. The responses are concatenated and finally returned trimmed to fit the sample_length.
        """
        assert parameter in [2, 3]
        response = self.__send_cmd(0xc6, parameter=[parameter], response_length=2, echo_expected=False)
        sample_length = int.from_bytes(response, byteorder="big", signed=False)
        sample_packages_count = int(math.ceil(sample_length / self.__MAX_PACKAGE_SIZE))
        # print("sample_length: {} -> {} packages".format(sample_length, sample_packages_count))
        samples = b''
        for _ in range(sample_packages_count):
            response = self.__send_cmd(0xa6, parameter=[parameter], response_length=64, echo_expected=False)
            samples += response
        return samples[0:sample_length]

    def __send_a55a_command(self, attempts: int=20) -> None:
        for _ in range(attempts):
            response = self.__send_cmd(0xa5, parameter=[0x5a], response_length=1)
            assert response[0] in [0, 1, 2, 3]
            if response[0] in [2, 3]:
                return
            sleep(0.02)
            self.__send_ping()
        raise RuntimeError(f"a55a command failed, all {attempts} attempts were answered with 0 or 1.")

    def __send_set_time_div(self, ns_per_div: int = 500000) -> None:
        """send the a3 command to set the sample rate.
        only allows values that follow this pattern: (1|2|3){0}. eg. 10, 2000 or 5.
        Maximum is 200_000_000"""
        # assert isinstance(ns_per_div, int)
        # assert 0 < ns_per_div <= 200 * 1000 * 1000  # when the value is higher than 200ms/div, the scan mode must be used
        # assert int(str(ns_per_div)[1:]) == 0, "only first digit is allowed to be != 0"
        # assert int(str(ns_per_div)[0]) in [1, 2, 5], "first digit must be 1, 2 or 5"
        # time_per_div_id = {1: 0, 2: 1, 5: 2}[int(str(ns_per_div)[0])] + int(math.log10(ns_per_div)) * 3
        assert ns_per_div in self.__burst_mode_ns_per_div_to_id_dic, "The given ns_per_div is invalid"

        time_per_div_id = self.__burst_mode_ns_per_div_to_id_dic[ns_per_div]
        self.__send_cmd(0xa3, parameter=[time_per_div_id])

    @staticmethod
    def _vertical_scale_id_to_factor(vs_id: int) -> float:
        assert 1 <= vs_id <= len(Hantek1008Raw.__VSCALE_FACTORS)
        return Hantek1008Raw.__VSCALE_FACTORS[vs_id - 1]

    @staticmethod
    def _vertical_scale_factor_to_id(vs_factor: float) -> int:
        assert vs_factor in Hantek1008Raw.__VSCALE_FACTORS
        return Hantek1008Raw.__VSCALE_FACTORS.index(vs_factor) + 1

    def __send_set_vertical_scale(self, scale_factors: List[float]) -> None:
        """send the a2 command to set the vertical sample scale factor per channel.
        Only following values are allowed: 1.0, 0.125, 0.02 [TODO: check] Volt/Div.
        scale_factor must be an array of length 8 with a float scale value for each channel.
        Or a single float, than all channel will have that scale factor"""
        assert all(x in Hantek1008Raw.__VSCALE_FACTORS for x in scale_factors)
        scale_factor_id: List[int] = [Hantek1008Raw._vertical_scale_factor_to_id(sf) for sf in scale_factors]
        self.__send_cmd(0xa2, parameter=scale_factor_id, sec_till_response_request=0.2132)

    def __send_set_active_channels(self, active_channels: List[int]) -> None:
        """
        Activates only the channels thar are in the list
        :param active_channels: a list of the channels that should be active
        :return:
        """
        assert active_channels is not None
        assert len(active_channels) > 0
        assert all(c in self.valid_channel_ids() for c in active_channels)
        assert len(set(active_channels)) == len(active_channels), "One channel must nut be more than once in the list"

        # set the count of active channels
        self.__send_cmd(0xa0, parameter=[len(active_channels)])

        active_channels_byte_map = [(0x01 if i in active_channels else 0x00)
                                    for i in range(0, 8)]
        # what channels should be active?
        self.__send_cmd(0xaa, parameter=active_channels_byte_map)

    def __send_set_trigger(self, source_channel: int, slope: str) -> None:
        slope_map = {"rising": 0, "falling": 1}
        assert source_channel in self.valid_channel_ids()
        assert slope in slope_map, f"Only following slope types are allowed: {list(slope_map.keys())}"

        self.__send_cmd(0xc1, parameter=[source_channel, slope_map[slope]])

    def __send_set_trigger_level(self, level: int) -> None:
        assert 0 <= level <= 2**12
        self.__send_cmd(0xab, parameter=int.to_bytes(level, length=2, byteorder="big", signed=False))

    def __send_ping(self, sec_till_start: float=0) -> None:
        self.__send_cmd(0xf3, sec_till_start=sec_till_start)

    def init(self) -> None:
        self._init1()
        self._init2()
        self._init3()

    def _init1(self) -> None:
        """Initialize the device like the windows software does it"""
        self.__send_cmd(0xb0)
        sleep(0.7)  # not sure if needed
        self.__send_cmd(0xb0)
        self.__send_ping()


        #self.__send_cmd(0xb9, parameter=bytes.fromhex("01 b0 04 00 00"))  # 185
        #self.__send_cmd(0xb7, parameter=bytes.fromhex("00"))  # 183
        #self.__send_cmd(0xbb, parameter=bytes.fromhex("08 00"))  # 187
        self.set_generator_speed(300_000)
        self.set_generator_on(False)

        response = self.__send_cmd(0xb5, response_length=64, echo_expected=False,
                                   sec_till_response_request=0.0193)  # 181
        # assert response == bytes.fromhex("00080008000800080008000800080008d407c907ef07cd07df07eb07c707d707"
        #                                 "e107d207f007d807e607ed07d507e207f607e007f007e907f007ef07ea07f207")

        response = self.__send_cmd(0xb6, response_length=64, echo_expected=False)  # 182
        # assert response == bytes.fromhex("04040404040404040404040404040404d200d500d800d400d400d500d200d200"
        #                                 "9c009f009f009d009d009d009e009d00fd01fc01fc01fc01fb01fa01fd01fc01")

        response = self.__send_cmd(0xe5, response_length=2, echo_expected=False)
        # assert response == bytes.fromhex("d6 06")

        response = self.__send_cmd(0xf7, response_length=64, echo_expected=False)
        # assert response == bytes.fromhex("2cfd8ffb54fa2ef878007a007b00780079007a0079007800b801bf01c301ba01"
        #                                 "bb01be01b701b801f90203030803fb02fc020003f502f80294ff92ff8fff93ff")

        response = self.__send_cmd(0xf8, response_length=64, echo_expected=False)
        # assert response == bytes.fromhex("92ff91ff96ff94ffc9fec4febdfec8fec7fec2fecffec9fe4cfe45fe3afe4afe"
        #                                 "48fe42fe54fe4dfe70ff70ff71ff70ff71ff71ff72ff71ff7efe7bfe7afe7efe")

        response = self.__send_cmd(0xfa, response_length=56, echo_expected=False)
        # assert response == bytes.fromhex("7dfe7efe80fe7ffe90019401930192018f01900191018f0195029b0299029802"
        #                                 "930294029702940290fd89fd84fd90fd8dfd8cfd94fd91fd")

        self.__send_cmd(0xf5, sec_till_response_request=0.2132)

        # self.__send_cmd(0xa0, parameter=bytes.fromhex("08"))
        # self.__send_cmd(0xaa, parameter=bytes.fromhex("0101010101010101"))
        # activate all 8 channels
        self.__send_set_active_channels(Hantek1008Raw.valid_channel_ids())

        self.__send_set_time_div(500 * 1000)  # 500us, the default value in the windows software

        self.__send_set_trigger(0, "rising")

        response = self.__send_cmd(0xa7, parameter=bytes.fromhex("0000"), response_length=1)
        assert response == bytes.fromhex("00")

        self.__send_cmd(0xac, parameter=bytes.fromhex("01f40009c50009c5"))

    def _init2(self) -> None:
        """get zero offsets for all channels and vscales"""
        self._zero_offsets = {}
        for vscale_id in range(1, 4):
            vscale = Hantek1008Raw._vertical_scale_id_to_factor(vscale_id)

            self.__send_ping()

            self.__send_set_vertical_scale([vscale] * Hantek1008Raw.channel_count())

            self.__send_cmd(0xa4, parameter=[0x01])

            self.__send_cmd(0xc0)

            sleep(0.0124)
            self.__send_cmd(0xc2)

            self.__send_a55a_command()

            samples2 = self.__send_c6_a6_command(0x02)
            samples3 = self.__send_c6_a6_command(0x03)
            samples = samples2 + samples3
            shorts = Hantek1008Raw.__from_bytes_to_shorts(samples)
            per_channel_data = Hantek1008Raw.__to_per_channel_lists(shorts, Hantek1008Raw.valid_channel_ids())
            zero_offset_per_channel = [sum(per_channel_data[ch]) / float(len(per_channel_data[ch]))
                                       for ch in Hantek1008Raw.valid_channel_ids()]
            self._zero_offsets[vscale] = zero_offset_per_channel

    def _init3(self) -> None:
        self.__send_cmd(0xf6, sec_till_response_request=0.2132)

        response = self.__send_cmd(0xe5, echo_expected=False, response_length=2)
        assert response == bytes.fromhex("d606")

        response = self.__send_cmd(0xf7, echo_expected=False, response_length=64)
        assert response == bytes.fromhex("2cfd8ffb54fa2ef878007a007b00780079007a0079007800b801bf01c301ba01"
                                         "bb01be01b701b801f90203030803fb02fc020003f502f80294ff92ff8fff93ff")

        response = self.__send_cmd(0xf8, echo_expected=False, response_length=64)
        assert response == bytes.fromhex("92ff91ff96ff94ffc9fec4febdfec8fec7fec2fecffec9fe4cfe45fe3afe4afe"
                                         "48fe42fe54fe4dfe70ff70ff71ff70ff71ff71ff72ff71ff7efe7bfe7afe7efe")

        response = self.__send_cmd(0xfa, echo_expected=False, response_length=56)
        assert response == bytes.fromhex("7dfe7efe80fe7ffe90019401930192018f01900191018f0195029b0299029802"
                                         "930294029702940290fd89fd84fd90fd8dfd8cfd94fd91fd")

        self.__send_set_time_div(self.__ns_per_div)

        self.__send_cmd(0xac, parameter=bytes.fromhex("00c80002bd0002bd"))

        self.__send_cmd(0xe4, parameter=[0x01])

        self.__send_cmd(0xe6, parameter=[0x01], echo_expected=False, response_length=10)
        # assert response == bytes.fromhex("eb06e606e606e706e706")

        self.__send_ping()

        self.__send_set_active_channels(self.__active_channels)

        self.__send_set_vertical_scale(self.__vertical_scale_factors)

        self.__send_set_time_div(self.__ns_per_div)

        self.__send_set_trigger(self.__trigger_channel, self.__trigger_slope)

        response = self.__send_cmd(0xa7, parameter=[0x00, 0x00], response_length=1)
        assert response == bytes.fromhex("00")

        self.__send_cmd(0xac, parameter=bytes.fromhex("0000000001000579"))

        self.__send_set_trigger_level(self.__trigger_level)

        response = self.__send_cmd(0xe9, echo_expected=False, response_length=2)
        assert response == bytes.fromhex("0109")

    def request_samples_burst_mode(self) -> Dict[int, List[int]]:
        """get the data"""

        self.__send_ping()

        # these two commands are not necessarily required
        self.__send_cmd(0xe4, parameter=[0x01])
        self.__send_cmd(0xe6, parameter=[0x01], echo_expected=False, response_length=10)
        # response ~ e906e506e406e406e506

        self.__send_cmd(0xa4, parameter=[0x01], sec_till_response_request=0.015)

        self.__send_cmd(0xc0)

        self.__send_cmd(0xc2)

        self.__send_a55a_command()

        sample_response = self.__send_c6_a6_command(0x02)
        sample_response += self.__send_c6_a6_command(0x03)

        # these two commands are not necessarily required
        self.__send_cmd(0xe4, parameter=[0x01])
        self.__send_cmd(0xe6, parameter=[0x01], echo_expected=False, response_length=10)
        # response ~ e806e406e506e406e406

        sample_shorts = Hantek1008Raw.__from_bytes_to_shorts(sample_response)

        per_channel_data = Hantek1008Raw.__to_per_channel_lists(sample_shorts, self.__active_channels)
        return per_channel_data

    @staticmethod
    def channel_count() -> int:
        return 8

    @staticmethod
    def valid_channel_ids() -> List[int]:
        return list(range(0, Hantek1008Raw.channel_count()))

    @staticmethod
    def valid_roll_mode_sampling_rates() -> List[float]:
        return copy.deepcopy(list(Hantek1008Raw.__roll_mode_sampling_rate_to_id_dic.keys()))

    @staticmethod
    def valid_burst_mode_ns_per_divs() -> List[float]:
        return copy.deepcopy(list(Hantek1008Raw.__burst_mode_ns_per_div_to_id_dic.keys()))

    @staticmethod
    def valid_vscale_factors() -> List[float]:
        return copy.deepcopy(Hantek1008Raw.__VSCALE_FACTORS)

    @staticmethod
    def actual_sampling_rate_factor(active_channel_count: int) -> float:
        """
        If not all channels are used the actual sampling rate is higher than the
        given sampling rate. The factor describe how much higher it is, depending on the amount
        of active channels.
        :return:
        """
        assert 1 <= active_channel_count <= Hantek1008Raw.channel_count()
        return [4.56, 3.03, 2.27, 1.82, 1.51, 1.3, 1.14, 1.00][active_channel_count-1]

    def request_samples_roll_mode_single_row(self, **argv) \
            -> Generator[Dict[int, int], None, None]:
        for per_channel_data in self.request_samples_roll_mode(**argv):
            for row in list(zip(*per_channel_data.values())):
                yield dict(zip(per_channel_data.keys(), row))

    def request_samples_roll_mode(self, sampling_rate: int = 440) \
            -> Generator[Dict[int, List[int]], None, None]:

        assert sampling_rate in Hantek1008Raw.__roll_mode_sampling_rate_to_id_dic, \
            f"sample_rate must be in {Hantek1008Raw.__roll_mode_sampling_rate_to_id_dic.keys()}"

        try:
            # sets the sample rate: 18 -> 440 samples/sec/channel
            sample_rate_id = Hantek1008Raw.__roll_mode_sampling_rate_to_id_dic[sampling_rate]
            self.__send_cmd(0xa3, parameter=[sample_rate_id])

            self.__send_ping(sec_till_start=0.0100)

            self.__send_cmd(0xa4, parameter=[0x02])

            # pipe error if a3 cmd/__send_set_time_div was not with parameter 1a/
            self.__send_cmd(0xc0)

            self.__send_cmd(0xc2)

            while True:
                ready_data_length = 0
                while ready_data_length == 0:
                    self.__send_ping()

                    response = self.__send_cmd(0xc7, response_length=2, echo_expected=False)
                    ready_data_length = int.from_bytes(response, byteorder="big", signed=False)
                    # ready_data_length =
                    #  (active_channels + ONE_MYSTIC_EXTRA_CHANNEL) * TWO_BYTES_PER_SAMPLE * row_count
                    assert ready_data_length % ((len(self.__active_channels) + 1)*2) == 0

                sample_response = b''
                while ready_data_length > 0:
                    sample_response_part = self.__send_cmd(0xc8, response_length=64, echo_expected=False)

                    if ready_data_length < 64:
                        #  remove zeros at the end
                        sample_response_part = sample_response_part[0:ready_data_length]

                    ready_data_length -= 64
                    sample_response += sample_response_part

                sample_shorts = Hantek1008Raw.__from_bytes_to_shorts(sample_response)
                # in rolling mode there is an additional 9th channel, with values around 1742
                # this channel will not be past to the caller
                per_channel_data = self.__to_per_channel_lists(sample_shorts, self.__active_channels,
                                                               expect_ninth_channel=True)
                yield per_channel_data
        except GeneratorExit:
            # TODO: auto start pause tread?
            pass

    def get_zero_offsets(self) -> Optional[Dict[float, List[float]]]:
        return copy.deepcopy(self._zero_offsets)

    def get_zero_offset(self, channel_id: int, vscale: Optional[float] = None) -> Optional[float]:
        assert channel_id in Hantek1008Raw.valid_channel_ids()
        assert vscale is None or vscale in Hantek1008Raw.valid_vscale_factors()

        # if this methode is called before init/connect zero_offset will be null
        if self._zero_offsets is None:
            return None

        if vscale is None:
            vscale = self.get_vscale(channel_id)

        return self._zero_offsets[vscale][channel_id]

    @staticmethod
    def get_generator_waveform_max_length() -> int:
        return 1440

    def set_generator_on(self, turn_on: bool) -> None:
        # TODO not tested
        self.__send_cmd(0xb7, parameter=[0x00])

        self.__send_cmd(0xbb, parameter=[0x08, 0x01 if turn_on else 0x00])

    def set_generator_speed(self, speed_in_rpm: int) -> None:
        # TODO speed_in_rpm must be round to valid values, dont know how
        def compute_pulse_length(speed_in_rpm: int, bits_per_wave: int = 8) -> int:
            assert 1 <= speed_in_rpm <= 750_000
            assert 1 <= bits_per_wave <= Hantek1008Raw.get_generator_waveform_max_length()
            # TODO values great then 750_000 are possible too, but then the decoding changes (firt paramter gets a 02)
            # and this other decoding is not completely understood
            return int(((8 * 360_000_000) / bits_per_wave) / speed_in_rpm)

        assert compute_pulse_length(300_000) == 1200

        pulse_length = compute_pulse_length(speed_in_rpm)
        parameter = bytes.fromhex("01") + pulse_length.to_bytes(length=4, byteorder='little', signed=False)
        assert len(parameter) == 1 + 4
        self.__send_cmd(0xb9, parameter=parameter)

    def set_generator_waveform(self, waveform: List[int]) -> None:
        """
        Every Byte in the waveform list contains information for every of the 8 digital ouputs to be on or of.
        The bit number i in one of those bytes tells if output i should be on or off in that part of the wave.
        :param waveform:
        :return:
        """
        # TODO not tested
        # example for waveform: F0 0F F0 0F
        # -> switches the output of every channel at every pulse
        # ch1 to ch4 start with down, ch5 to ch8 start up
        assert len(waveform) <= Hantek1008Raw.get_generator_waveform_max_length()
        assert len(waveform) <= 62, "Currently not supported"
        assert all(b <= 0b1111_1111 for b in waveform)

        self.__send_cmd(0xb7, parameter=[0x00])

        # send the length of the waveform in bytes
        self.__send_cmd(0xbf, parameter=int.to_bytes(len(waveform), length=2, byteorder="little", signed=False))

        zeros = [0] * (62 - len(waveform))
        self.__send_cmd(0xb8, parameter=[0x01] + waveform + zeros)

    def __loop_f3(self) -> None:
        log.debug("start pause thread")
        while not self.__cancel_pause_thread:
            self.__send_ping()
            sleep(0.01)
        log.debug("stop pause thread")

    def pause(self) -> None:
        if self.is_paused():
            raise RuntimeError("Can't pause because device is already pausing")
        self.__cancel_pause_thread = False
        self.__pause_thread = Thread(target=self.__loop_f3)
        self.__pause_thread.start()

    def cancel_pause(self) -> None:
        if not self.is_paused():
            raise RuntimeError("Can't cancel pause because device is not paused")
        assert self.__pause_thread is not None
        self.__cancel_pause_thread = True
        self.__pause_thread.join()
        self.__pause_thread = None

    def is_paused(self) -> bool:
        return self.__pause_thread is not None

    def close(self) -> None:
        if self.is_paused():
            self.cancel_pause()

        # read maybe leftover data
        self.__clear_leftover()
        self.__send_ping()
        self.__send_cmd(0xf4)
        self._dev.reset()

    def __clear_leftover(self) -> None:
        """
        If a __send_cmd was canceled after the write but before the read, the Hantek device
        still wants to send the answer. This method will try to read such a leftover answer
        if there is there is one
        :return:
        """
        try:
            response = bytes(self.__in.read(64, timeout=100))
        except usb.core.USBError:
            log.debug("no left over data")
            pass
        else:
            log.debug(f"left over data: {response.hex()}")

    def get_vscales(self) -> List[float]:
        return copy.deepcopy(list(self.__vertical_scale_factors))

    def get_vscale(self, channel_id: int) -> float:
        assert channel_id in Hantek1008Raw.valid_channel_ids()
        return self.__vertical_scale_factors[channel_id]

    def get_active_channels(self) -> List[int]:
        return copy.deepcopy(self.__active_channels)

    @staticmethod
    def __from_bytes_to_shorts(data: bytes) -> List[int]:
        """Take two following bytes to build a integer (using little endianess) """
        assert len(data) % 2 == 0
        return [data[i] + data[i + 1] * 256 for i in range(0, len(data), 2)]

    @staticmethod
    def __to_per_channel_lists(shorts: List[int], active_channels: List[int], expect_ninth_channel: bool = False
                               ) -> Dict[int, List[int]]:
        """Create a dictionary (of the size of 'channel_count') of lists,
        where the dictionary at key x contains the data for channel x+1 of the hantek device.
        In rolling mode there is an additional 9th channel, with values around 1742 this
        channel will not be past to the caller.
        """
        active_channels = sorted(active_channels)
        active_channel_count = len(active_channels)
        real_channel_count = active_channel_count
        if expect_ninth_channel:
            real_channel_count += 1
        return {active_channels[i]: shorts[i::real_channel_count]
                for i in range(0, active_channel_count)}


"""
Below goes stuff that is needed for more advanced features
"""

# list of dicts of lists of dicts
# usecase: __correction_data[channel_id][vscale][..] = {"units":..., "factor": ...}
CorrectionDataType = List[Dict[float, Dict[float, float]]]

# a function that awaits an channel id [0,7], vscale and a deltatime (time in sec since creation of this class)
# it computes a correction factor that can be applied (added) to the normal zero_offset
ZeroOffsetShiftCompensationFunctionType = Callable[[int, float, float], float]


class Hantek1008(Hantek1008Raw):
    """
    A more advanced version of Hantek1008Raw. It features raw values to voltage conversion
    , usage of external generated calibration data and zero offset shift calibration compensation.
    """

    def __init__(self, ns_per_div: int = 500_000,
                 vertical_scale_factor: Union[float, List[float]] = 1.0,
                 active_channels: Optional[List[int]] = None,
                 correction_data: Optional[CorrectionDataType] = None,
                 zero_offset_shift_compensation_channel: Optional[int] = None,
                 zero_offset_shift_compensation_function: Optional[ZeroOffsetShiftCompensationFunctionType] = None,
                 zero_offset_shift_compensation_function_time_offset_sec: int = 0) -> None:

        if active_channels is None:
            active_channels = Hantek1008Raw.valid_channel_ids()
        if correction_data is None:
            correction_data = [{} for _ in range(Hantek1008Raw.channel_count())]

        assert len(correction_data) == Hantek1008Raw.channel_count()
        assert all(isinstance(x, dict) for x in correction_data)

        assert zero_offset_shift_compensation_channel is None or zero_offset_shift_compensation_function is None
        if zero_offset_shift_compensation_channel is not None:
            assert zero_offset_shift_compensation_channel not in active_channels
            assert zero_offset_shift_compensation_channel in Hantek1008Raw.valid_channel_ids()
            assert zero_offset_shift_compensation_channel not in active_channels
            active_channels = active_channels + [zero_offset_shift_compensation_channel]

        Hantek1008Raw.__init__(self, ns_per_div, vertical_scale_factor, active_channels)

        self.__correction_data: CorrectionDataType = copy.deepcopy(correction_data)

        self.__zero_offset_shift_compensation_channel: Optional[int] = zero_offset_shift_compensation_channel
        self.__zero_offset_shift_compensation_value: float = 0.0

        self.__zero_offset_shift_compensation_function: Optional[ZeroOffsetShiftCompensationFunctionType] \
            = zero_offset_shift_compensation_function
        self.__start_monotonic_time = time.monotonic() - zero_offset_shift_compensation_function_time_offset_sec

    def get_used_zero_offsets_shift_compensation_method(self)-> Optional[str]:
        assert not (self.__zero_offset_shift_compensation_channel
                    and self.__zero_offset_shift_compensation_function)
        if self.__zero_offset_shift_compensation_channel:
            return f"channel {self.__zero_offset_shift_compensation_channel}"
        if self.__zero_offset_shift_compensation_function:
            return f"function {self.__zero_offset_shift_compensation_function}"
        return None

    def __update_zero_offset_compensation_value(self, zero_readings: List[int]) -> None:
        # TODO problem zero offset different on different vscales?
        assert self.__zero_offset_shift_compensation_channel is not None
        assert self._zero_offsets is not None
        zoscc_vscale = Hantek1008Raw.get_vscale(self, self.__zero_offset_shift_compensation_channel)
        assert zoscc_vscale == 1.0  # is this really necessary?
        zoscc_zero_offset = self._zero_offsets[zoscc_vscale][self.__zero_offset_shift_compensation_channel]

        adaption_factor = 0.00002  # [0,1]
        for v in zero_readings:
            # print("v", v, "zo", zoscc_zero_offset)
            delta = v - zoscc_zero_offset
            self.__zero_offset_shift_compensation_value = \
                (1.0 - adaption_factor) * self.__zero_offset_shift_compensation_value \
                + adaption_factor * delta
        log.debug("zosc-value", self.__zero_offset_shift_compensation_value)

    @overrides
    def get_zero_offset(self, channel_id: int, vscale: Optional[float] = None) -> float:
        if vscale is None:
            vscale = Hantek1008Raw.get_vscale(self, channel_id)

        zero_offset = Hantek1008Raw.get_zero_offset(self, channel_id, vscale)
        assert zero_offset is not None
        if self.__zero_offset_shift_compensation_channel is not None:
            zero_offset += self.__zero_offset_shift_compensation_value
        if self.__zero_offset_shift_compensation_function is not None:
            delta_sec = time.monotonic() - self.__start_monotonic_time
            zero_offset += self.__zero_offset_shift_compensation_function(channel_id, vscale, delta_sec)
        return zero_offset

    @overrides
    def request_samples_roll_mode_single_row(self, **argv)\
            -> Generator[Dict[int, float], None, None]:
        for per_channel_data in self.request_samples_roll_mode(**argv):
            for row in list(zip(*per_channel_data.values())):
                yield dict(zip(per_channel_data.keys(), row))

    @overrides
    def request_samples_roll_mode(self, sampling_rate: int = 440, mode: str = "volt") \
            -> Generator[Dict[int, Union[List[float], List[int]]], None, None]:

        assert mode in ["volt", "raw", "volt+raw"]
        active_channel_count = len(Hantek1008Raw.get_active_channels(self))

        for raw_per_channel_data in Hantek1008Raw.request_samples_roll_mode(self, sampling_rate):
            assert len(raw_per_channel_data) == active_channel_count
            yield self.__process_raw_per_channel_data(raw_per_channel_data, mode)

    def __remove_zosc_channel_data(self, per_channel_data: Dict[int, Union[List[int], List[float]]]) -> None:
        if self.__zero_offset_shift_compensation_channel is not None:
            if self.__zero_offset_shift_compensation_channel in per_channel_data:
                del per_channel_data[self.__zero_offset_shift_compensation_channel]
            if self.__zero_offset_shift_compensation_channel + Hantek1008Raw.channel_count() in per_channel_data:
                del per_channel_data[self.__zero_offset_shift_compensation_channel]

    def __extract_channel_volts(self, per_channel_data: Dict[int, List[int]]) -> Dict[int, List[float]]:
        """Extract the voltage values from the raw byte array that came from the device"""
        if self.__zero_offset_shift_compensation_channel is not None:
            self.__update_zero_offset_compensation_value(
                per_channel_data[self.__zero_offset_shift_compensation_channel])
        return {ch: self.__raw_to_volt(channel_data, ch) for ch, channel_data in per_channel_data.items()}

    def __raw_to_volt(self, raw_values: List[int], channel_id: int) -> List[float]:
        """Convert the raw shorts to useful volt values"""
        vscale = 1.0
        zero_offset = 2048

        if channel_id < Hantek1008Raw.channel_count():
            vscale = Hantek1008Raw.get_vscale(self, channel_id)
            # get right zero offset for that channel and the used vertical scale factor (vscale)
            zero_offset = self.get_zero_offset(channel_id, vscale)

        scale = 0.01 * vscale

        # accuracy = -int(math.log10(scale)) + 2  # amount of digits after the dot that is not nearly random
        accuracy = [3, 4, 5][Hantek1008Raw._vertical_scale_factor_to_id(vscale) - 1]
        return [round(
            self.__calc_correction_factor(v - zero_offset, channel_id, vscale) * (v - zero_offset) * scale
            , ndigits=accuracy)
            for v in raw_values]

    def __calc_correction_factor(self, delta_to_zero: float, channel_id: int, vscale: float) -> float:
        """
        Compute a correction factor based on the given calibration data.
        Always returns 1.0 if no calibration data for the requested channel or at all is available.
        :param delta_to_zero:
        :param channel_id:
        :param vscale:
        :return:
        """
        if channel_id not in Hantek1008Raw.valid_channel_ids() \
                or vscale not in self.__correction_data[channel_id]:
            return 1.0

        channel_cd = self.__correction_data[channel_id][vscale]

        if len(channel_cd) == 0:
            return 1.0

        if len(channel_cd) == 1:
            return channel_cd[0]

        units_less, cfactor_less = max(((key, value)
                                        for key, value
                                        in channel_cd.items()
                                        if key <= delta_to_zero), default=(None, None))
        units_greater, cfactor_greater = min(((key, value)
                                              for key, value
                                              in channel_cd.items()
                                              if key >= delta_to_zero), default=(None, None))
        assert units_less is not None or units_greater is not None
        if units_less is None:
            return cfactor_less
        if units_greater is None:
            return cfactor_greater

        alpha = (delta_to_zero - units_less) / (units_greater - units_less)
        return (1.0 - alpha) * cfactor_less + alpha * cfactor_greater

    def __process_raw_per_channel_data(self, raw_per_channel_data: Dict[int, List[int]], mode: str
                                       ) -> Dict[int, Union[List[int], List[float]]]:
        assert mode in ["raw", "volt", "volt+raw"]
        result: Dict[int, Union[List[float], List[int]]] = {}
        if "volt" in mode:
            result.update(self.__extract_channel_volts(raw_per_channel_data))
        if "raw" in mode:
            raw_channel_offset = Hantek1008Raw.channel_count() if mode == "volt+raw" else 0
            result.update({ch + raw_channel_offset: values
                           for ch, values in raw_per_channel_data.items()})
        self.__remove_zosc_channel_data(result)
        return result

    @overrides
    def request_samples_burst_mode(self, mode: str = "volt"
                                   ) -> Dict[int, Union[List[int], List[float]]]:
        assert self.__zero_offset_shift_compensation_channel is None, \
            "zero offset shift compensation is not implemented for burst mode"
        raw_per_channel_data = Hantek1008Raw.request_samples_burst_mode(self)
        return self.__process_raw_per_channel_data(raw_per_channel_data, mode)
