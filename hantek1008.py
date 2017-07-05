import usb.core
import usb.util
import usb.backend
import time
from typing import List, Dict, Tuple
import logging as log
import math
from threading import Thread
import copy

"""
 To get acces to the USB Device:
 
 1. create file "/etc/udev/rules.d/99-hantek1008.rules" with content:
    ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="0783", ATTR{idProduct}=="5725", MODE="0666"
 2. sudo udevadm control -R
 3. Replug the device
"""


def to_hex_array(hex_string: str) -> bytes:
    return bytes.fromhex(hex_string)


def to_hex_string(hex_array: bytes) -> str:
    return bytes.hex(hex_array)


CorrectionDataType = List[Dict[float, Dict[float, float]]]


class Hantek1008:
    """
    
    """
    # channel_id/channel_index are zero based
    # channel names are one based

    __MAX_PACKAGE_SIZE: int = 64
    __VSCALE_FACTORS: List[int] = [0.02, 0.125, 1.0]
    __roll_mode_sampling_rate_to_id_dic: Dict[int, int] = {440: 0x18, 220: 0x19, 88: 0x1a, 44: 0x1b, 22: 0x1c}

    def __init__(self, ns_per_div: int = 500_000,
                 vertical_scale_factor: float or List[float] = 1.0,
                 correction_data: CorrectionDataType or None = None,
                 zero_offset_shift_compensation_channel: int or None = None):
        """
        :param ns_per_div: 
        :param vertical_scale_factor: must be an array of length 8 with a float scale value for each channel. 
        Or a single float, than all channel will have that scale factor. The float must be 1.0, 0.2 or 0.02.
        """
        if correction_data is None:
            correction_data = [{} for _ in range(8)]
        assert isinstance(vertical_scale_factor, float) or len(vertical_scale_factor) == 8
        assert len(correction_data) == 8
        assert all(isinstance(x, dict) for x in correction_data)
        assert zero_offset_shift_compensation_channel is None or zero_offset_shift_compensation_channel in range(0, 8)

        self.__ns_per_div: int = ns_per_div  # on value for all channels

        # on vertical scale factor (float) per channel
        self.__vertical_scale_factors: List[float] = [vertical_scale_factor] * 8 if isinstance(vertical_scale_factor,
                                                                                               float) \
            else copy.deepcopy(vertical_scale_factor)  # scale factor per channel

        # list of dicts of lists of dicts
        # usecase: __correction_data[channel_id][vscale][..] = {"units":..., "factor": ...}
        self.__correction_data: CorrectionDataType = copy.deepcopy(correction_data)

        self.__zero_offset_shift_compensation_channel: int or None = zero_offset_shift_compensation_channel
        self.__zero_offset_shift_compensation_value: float = 0.0

        # dict of list of shorts, outer dict is of size 3 and contains values
        # for every vertical scale factor, inner list contains an zero offset per channel
        self.__zero_offsets: Dict[float, List[int]] = None

        self.__out = None  # the usb out endpoint
        self.__in = None  # the usb in endpoint
        self._dev = None  # the usb device
        self._cfg = None  # the used usb configuration
        self._intf = None  # the used usb interface

    def connect(self):
        """Looks for a plugged hantek 1008c device and set ups the connection to it"""
        # find our device
        self._dev = usb.core.find(idVendor=0x0783, idProduct=0x5725)

        # was it found?
        if self._dev is None:
            raise RuntimeError('No Hantek 1008 device found')

        # set the active configuration. With no arguments, the first
        # configuration will be the active one
        self._dev.set_configuration()

        # get an endpoint instance
        self._cfg = self._dev.get_active_configuration()
        self._intf = self._cfg[(0, 0)]

        self.__out = usb.util.find_descriptor(
            self._intf,
            # match the first OUT endpoint
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT)

        self.__in = usb.util.find_descriptor(
            self._intf,
            # match the first IN endpoint
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN)

        assert self.__out is not None
        assert self.__in is not None

    def __sleep(self, sleep_time=0.002):
        """
        Sleeps sleep_time seconds
        defaults to: delay between the commands send in windows software (is about 2 ms)
        """
        if sleep_time <= 0.0:
            return
        time.sleep(sleep_time)
        # start = time.time()
        # while time.time() - start < sleep_time: # loop to guarantee delay is at least past
        #    time.sleep(sleep_time)

    def __write_and_receive(self, message: bytes, response_length: int,
                            sec_till_response_request: float = 0.002, sec_till_start: float = 0.002) -> bytes:
        """write and read from the device"""
        start_time = time.time()

        assert isinstance(message, bytes)
        log.debug(f">[{len(message):2}] {to_hex_string(message)}")

        self.__sleep(sec_till_start)

        self.__out.write(message)

        self.__sleep(sec_till_response_request)

        response = bytes(self.__in.read(response_length))

        log.debug(f"<[{len(response):2}] {to_hex_string(response)}")
        log.debug(f"delta: {time.time()-start_time:02.4f} sec")
        assert len(response) == response_length

        return response

    def __send_cmd(self, cmd_id: int, parameter: bytes or List[int] or str = b'',
                   response_length: int = 0, echo_expected: bool = True,
                   sec_till_response_request: float = 0, sec_till_start: float = 0.002) -> bytes:
        """sends a command to the device and checks if the device echos the command id"""
        if isinstance(parameter, str):
            parameter = to_hex_array(parameter)
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

    def __send_c6_a6_command(self, parameter: int):
        """send the c602 or c603 command, then parse the response as sample_length. then follow CEIL(sample_length/64) 
        a602 or a603 request. The responses are concatenated and finally returned trimmed to the fit the sample_length.
        """
        assert parameter in [2, 3]
        response = self.__send_cmd(0xc6, parameter=[parameter], response_length=2, echo_expected=False)
        sample_length = int(response.hex(), 16)
        sample_packages_count = math.ceil(sample_length / self.__MAX_PACKAGE_SIZE)
        # print("sample_length: {} -> {} packages".format(sample_length, sample_packages_count))
        samples = b''
        for _ in range(sample_packages_count):
            response = self.__send_cmd(0xa6, parameter=[parameter], response_length=64, echo_expected=False)
            # sec_till_start=0, sec_till_response_request=0)
            samples += response
        return samples[0:sample_length]

    def __send_a55a_command(self, attempts=20):
        for _ in range(attempts):
            response = self.__send_cmd(0xa5, parameter=[0x5a], response_length=1)
            assert response[0] in [0, 1, 2, 3]
            if response[0] in [2, 3]:
                return
            self.__sleep(0.02)
            self.__send_cmd(0xf3)
        raise RuntimeError(f"a55a command failed, all {attempts} attempts were answered with 0 or 1.")

    def __send_set_time_div(self, ns_per_div: int = 500000):
        """send the a3 command to set the sample rate.
        only allows values that follow this pattern: (1|2|3){0}. eg. 10, 2000 or 5.
        Maximum is 200_000_000"""
        assert isinstance(ns_per_div, int)
        assert ns_per_div > 0
        assert ns_per_div <= 200 * 1000 * 1000  # when the value is higher than 200ms/div, the scan mode must be used
        assert int(str(ns_per_div)[1:]) == 0, "only first digit is allowed to be != 0"
        assert int(str(ns_per_div)[0]) in [1, 2, 5], "first digit must be 1, 2 or 5"
        time_per_div_id = {1: 0, 2: 1, 5: 2}[int(str(ns_per_div)[0])] + int(math.log10(ns_per_div)) * 3
        self.__send_cmd(0xa3, parameter=[time_per_div_id])

    def __vertical_scale_id_to_factor(self, vs_id: int):
        assert 1 <= vs_id <= len(Hantek1008.__VSCALE_FACTORS)
        return Hantek1008.__VSCALE_FACTORS[vs_id - 1]

    def __vertical_scale_factor_to_id(self, vs_factor: float):
        assert vs_factor in Hantek1008.__VSCALE_FACTORS
        return Hantek1008.__VSCALE_FACTORS.index(vs_factor) + 1

    def __send_set_vertical_scale(self, scale_factors: List[float] = 1.0):
        """send the a2 command to set the vertical sample scale factor per channel. 
        Only following values are allowed: 1.0, 0.125, 0.02 [TODO: check] Volt/Div.
        scale_factor must be an array of length 8 with a float scale value for each channel. 
        Or a single float, than all channel will have that scale factor"""
        assert all(x in Hantek1008.__VSCALE_FACTORS for x in scale_factors)
        scale_factors = [self.__vertical_scale_factor_to_id(sf) for sf in scale_factors]
        self.__send_cmd(0xa2, parameter=scale_factors, sec_till_response_request=0.2132)

    def init(self):
        self._init1()
        self._init2()
        self._init3()

    def _init1(self):
        """Initialize the device like the windows software does it"""
        self.__send_cmd(0xb0)  # 176
        time.sleep(0.7)  # not sure if needed
        self.__send_cmd(0xb0)  # 176
        self.__send_cmd(0xf3)  # 243
        self.__send_cmd(0xb9, parameter=to_hex_array("01 b0 04 00 00"))  # 185
        self.__send_cmd(0xb7, parameter=to_hex_array("00"))  # 183
        self.__send_cmd(0xbb, parameter=to_hex_array("08 00"))  # 187

        response = self.__send_cmd(0xb5, response_length=64, echo_expected=False,
                                   sec_till_response_request=0.0193)  # 181
        assert response == to_hex_array("00080008000800080008000800080008d407c907ef07cd07df07eb07c707d707"
                                        "e107d207f007d807e607ed07d507e207f607e007f007e907f007ef07ea07f207")

        response = self.__send_cmd(0xb6, response_length=64, echo_expected=False)  # 182
        assert response == to_hex_array("04040404040404040404040404040404d200d500d800d400d400d500d200d200"
                                        "9c009f009f009d009d009d009e009d00fd01fc01fc01fc01fb01fa01fd01fc01")

        response = self.__send_cmd(0xe5, response_length=2, echo_expected=False)
        assert response == to_hex_array("d6 06")

        response = self.__send_cmd(0xf7, response_length=64, echo_expected=False)
        assert response == to_hex_array("2cfd8ffb54fa2ef878007a007b00780079007a0079007800b801bf01c301ba01"
                                        "bb01be01b701b801f90203030803fb02fc020003f502f80294ff92ff8fff93ff")

        response = self.__send_cmd(0xf8, response_length=64, echo_expected=False)
        assert response == to_hex_array("92ff91ff96ff94ffc9fec4febdfec8fec7fec2fecffec9fe4cfe45fe3afe4afe"
                                        "48fe42fe54fe4dfe70ff70ff71ff70ff71ff71ff72ff71ff7efe7bfe7afe7efe")

        response = self.__send_cmd(0xfa, response_length=56, echo_expected=False)
        assert response == to_hex_array("7dfe7efe80fe7ffe90019401930192018f01900191018f0195029b0299029802"
                                        "930294029702940290fd89fd84fd90fd8dfd8cfd94fd91fd")

        self.__send_cmd(0xf5, sec_till_response_request=0.2132)

        self.__send_cmd(0xa0, parameter=to_hex_array("08"))

        self.__send_cmd(0xaa, parameter=to_hex_array("0101010101010101"))

        # self.send_cmd(0xa3, parameter=to_hex_array("11"))
        self.__send_set_time_div(500 * 1000)  # 500us, the default value in the windows software

        self.__send_cmd(0xc1, parameter=to_hex_array("0000"))

        response = self.__send_cmd(0xa7, parameter=to_hex_array("0000"), response_length=1)
        assert response == to_hex_array("00")

        self.__send_cmd(0xac, parameter=to_hex_array("01f40009c50009c5"))

    def _init2(self):
        """calibrate"""
        self.__zero_offsets = {}
        for vscale_id in range(1, 4):
            vscale = self.__vertical_scale_id_to_factor(vscale_id)

            self.__send_cmd(0xf3)

            # self.send_cmd(0xa2, parameter=to_hex_array("0101010101010101"))
            # self.send_cmd(0xa2, parameter=[i+1]*8, sec_till_response_request=0.2132)
            self.__send_set_vertical_scale([vscale] * 8)

            self.__send_cmd(0xa4, parameter=[0x01])

            self.__send_cmd(0xc0)

            self.__sleep(0.0124)
            self.__send_cmd(0xc2)

            self.__send_a55a_command()

            samples2 = self.__send_c6_a6_command(0x02)
            samples3 = self.__send_c6_a6_command(0x03)
            samples = samples2 + samples3
            shorts = self.__from_bytes_to_shorts(samples)
            zero_offset_per_channel = [sum(channel_data) / float(len(channel_data))
                                       for channel_data in self.__to_per_channel_lists(shorts)]
            self.__zero_offsets[vscale] = zero_offset_per_channel

    def _init3(self):
        self.__send_cmd(0xf6, sec_till_response_request=0.2132)

        response = self.__send_cmd(0xe5, echo_expected=False, response_length=2)
        assert response == to_hex_array("d606")

        response = self.__send_cmd(0xf7, echo_expected=False, response_length=64)
        assert response == to_hex_array("2cfd8ffb54fa2ef878007a007b00780079007a0079007800b801bf01c301ba01"
                                        "bb01be01b701b801f90203030803fb02fc020003f502f80294ff92ff8fff93ff")

        response = self.__send_cmd(0xf8, echo_expected=False, response_length=64)
        assert response == to_hex_array("92ff91ff96ff94ffc9fec4febdfec8fec7fec2fecffec9fe4cfe45fe3afe4afe"
                                        "48fe42fe54fe4dfe70ff70ff71ff70ff71ff71ff72ff71ff7efe7bfe7afe7efe")

        response = self.__send_cmd(0xfa, echo_expected=False, response_length=56)
        assert response == to_hex_array("7dfe7efe80fe7ffe90019401930192018f01900191018f0195029b0299029802"
                                        "930294029702940290fd89fd84fd90fd8dfd8cfd94fd91fd")

        # self.send_cmd(0xa3, parameter=[0x10])
        self.__send_set_time_div(self.__ns_per_div)

        self.__send_cmd(0xac, parameter=to_hex_array("00c80002bd0002bd"))

        self.__send_cmd(0xe4, parameter=[0x01])

        response = self.__send_cmd(0xe6, parameter=[0x01], echo_expected=False, response_length=10)
        # assert response == to_hex_array("eb06e606e606e706e706")

        self.__send_cmd(0xf3)

        self.__send_cmd(0xa0, parameter=[0x08])

        self.__send_cmd(0xaa, parameter=[0x01] * 8)

        # self.send_cmd(0xa2, parameter=to_hex_array("0303030303030303"), sec_till_response_request=0.2132)
        self.__send_set_vertical_scale(self.__vertical_scale_factors)

        # self.send_cmd(0xa3, parameter=[0x10])        # auch schon mal 0x10         oder 0x12
        self.__send_set_time_div(self.__ns_per_div)

        self.__send_cmd(0xc1, parameter=[0x07, 0x00])  # auch schon mal [0x07, 0x00] oder [0x00, 0x01]

        response = self.__send_cmd(0xa7, parameter=[0x00, 0x00], response_length=1)
        assert response == to_hex_array("00")

        self.__send_cmd(0xac, parameter=to_hex_array("0000000001000579"))

        self.__send_cmd(0xab, parameter=to_hex_array("080e"))  # auch schon mal 080e oder 0811
        # oder 080f oder 07fd oder 07e9

        response = self.__send_cmd(0xe9, echo_expected=False, response_length=2)
        assert response == to_hex_array("0109")

    def get_calibration_data(self) -> Dict[float, List[float]]:
        # copy dict and return it
        return copy.deepcopy(self.__zero_offsets)

    def get_zero_offset(self, channel_id: int, vscale: float = None) -> float:
        if vscale is None:
            vscale = self.__vertical_scale_factors[channel_id]
        return self.__zero_offsets[vscale][channel_id]

    def request_samples_normal_mode(self) -> Tuple[List[List[float]]]:
        """get the data"""
        assert self.__zero_offset_shift_compensation_channel is None, \
            "zero offset shift compensation is not implemented for normal mode"

        self.__send_cmd(0xf3)

        self.__send_cmd(0xe4, parameter=[0x01])

        response = self.__send_cmd(0xe6, parameter=[0x01], echo_expected=False, response_length=10)
        # response ~ e906e506e406e406e506

        self.__send_cmd(0xa4, parameter=[0x01], sec_till_response_request=0.015)

        self.__send_cmd(0xc0)

        self.__send_a55a_command()

        samples2 = self.__send_c6_a6_command(0x02)
        samples3 = self.__send_c6_a6_command(0x03)

        self.__send_cmd(0xe4, parameter=[0x01])

        response = self.__send_cmd(0xe6, parameter=[0x01], echo_expected=False, response_length=10)
        # response ~ e806e406e506e406e406

        # now process the data
        channel_volts2 = self.__extract_channel_volts(samples2)
        channel_volts3 = self.__extract_channel_volts(samples3)

        return channel_volts2, channel_volts3

    @staticmethod
    def valid_roll_sampling_rates() -> List[int]:
        return copy.deepcopy(list(Hantek1008.__roll_mode_sampling_rate_to_id_dic.keys()))

    @staticmethod
    def valid_vscale_factors() -> List[int]:
        return copy.deepcopy(Hantek1008.__VSCALE_FACTORS)

    def request_samples_roll_mode_single_row(self, sampling_rate: int = 440, raw: bool = False) -> List[List[float]]:
        for channel_data in self.request_samples_roll_mode(sampling_rate=sampling_rate, raw=raw):
            for v in zip(*channel_data):
                yield v

    def request_samples_roll_mode(self, sampling_rate: int = 440, raw: bool = False) -> List[List[float]]:
        assert sampling_rate in Hantek1008.valid_roll_sampling_rates(), \
            f"sample_rate must be in {Hantek1008.valid_roll_sampling_rates()}"

        try:
            # sets the sample rate: 18 -> 440 samples/sec/channel
            sample_rate_id = Hantek1008.__roll_mode_sampling_rate_to_id_dic[sampling_rate]
            self.__send_cmd(0xa3, parameter=[sample_rate_id])

            self.__send_cmd(0xf3, sec_till_start=0.100)

            self.__send_cmd(0xa4, parameter=[0x02])

            # pipe error if a3 cmd/__send_set_time_div was not with parameter 1a/
            self.__send_cmd(0xc0)

            self.__send_cmd(0xc2)

            while True:
                ready_data_length = 0
                while ready_data_length == 0:
                    self.__send_cmd(0xf3)

                    response = self.__send_cmd(0xc7, response_length=2, echo_expected=False)
                    ready_data_length = response[0] * 256 + response[1]
                    assert ready_data_length % 9 == 0

                sample_response = b''
                while ready_data_length > 0:
                    sample_response_part = self.__send_cmd(0xc8, response_length=64, echo_expected=False)

                    if ready_data_length < 64:
                        #  remove zeros at the end
                        sample_response_part = sample_response_part[0:ready_data_length]

                    ready_data_length -= 64
                    sample_response += sample_response_part

                if raw:
                    yield self.__to_per_channel_lists(self.__from_bytes_to_shorts(sample_response), 9)[
                          0:8]  # remove strange 9th channel
                else:
                    # in rolling mode there is an additional 9th channel, with values around 1742
                    channel_volts = self.__extract_channel_volts(sample_response, channel_count=9)
                    yield channel_volts[0:8]  # remove strange 9th channel
        except GeneratorExit:
            # TODO: auto start pause tread?
            pass

    def __update_zero_offset_compensation_value(self, zero_readings: List[float]) -> None:
        # TODO problem zero offset different on different vscales?
        zoscc_vscale = self.__vertical_scale_factors[self.__zero_offset_shift_compensation_channel]
        assert zoscc_vscale == 1.0  # is this really necessary?
        zoscc_zero_offset = self.__zero_offsets[zoscc_vscale][self.__zero_offset_shift_compensation_channel]

        adaption_factor = 0.00002  # [0,1]
        for v in zero_readings:
            # print("v", v, "zo", zoscc_zero_offset)
            delta = v - zoscc_zero_offset
            self.__zero_offset_shift_compensation_value = \
                (1.0 - adaption_factor) * self.__zero_offset_shift_compensation_value \
                + adaption_factor * delta
        print("zosc-value", self.__zero_offset_shift_compensation_value)

    def __get_zero_offset(self, channel_id: int, vscale: float) -> float:
        zero_offset = self.__zero_offsets[vscale][channel_id]
        if self.__zero_offset_shift_compensation_channel is not None:
            zero_offset += self.__zero_offset_shift_compensation_value
        return zero_offset

    def set_generator_on(self, turn_on: bool):
        # TODO not tested
        if turn_on:
            self.__send_cmd(0xb9, parameter=to_hex_array("01b0040000"))

        self.__send_cmd(0xb7, parameter=[0x00])

        self.__send_cmd(0xbb, parameter=[0x08, 0x01 if turn_on else 0x00])

    def set_generator_waveform(self, waveform: List[int]) -> None:
        # TODO not tested
        # example for waveform: F0 0F F0 0F
        # -> switches the output of every channel at every pulse
        # ch1 to ch4 start with down, ch5 to ch8 start up
        assert len(waveform) <= 0xFFFF
        assert len(waveform) <= 62, "Currently not supported"
        assert all(b <= 0xFF for b in waveform)

        self.__send_cmd(0xb7, parameter=[len(waveform) % 256, len(waveform) >> 8])

        zeros = [0] * (62 - len(waveform))
        self.__send_cmd(0xbf, parameter=[0x01] + waveform + zeros)

    __pause_thread = None
    __cancel_pause_thread = None

    def __loop_f3(self) -> None:
        log.debug("start pause thread")
        while not self.__cancel_pause_thread:
            self.__send_cmd(0xf3)
            time.sleep(0.01)
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
        self.__send_cmd(0xf3)
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

    def __extract_channel_volts(self, data: bytes, channel_count: int = 8) -> List[List[float]]:
        """Extract the voltage values from the raw byte array that came from the device"""
        shorts = self.__from_bytes_to_shorts(data)
        per_channel_lists = self.__to_per_channel_lists(shorts, channel_count)
        if self.__zero_offset_shift_compensation_channel is not None:
            self.__update_zero_offset_compensation_value(
                per_channel_lists[self.__zero_offset_shift_compensation_channel])
        return [self.__shorts_to_volt(channel_data, ch) for ch, channel_data in enumerate(per_channel_lists)]

    def __from_bytes_to_shorts(self, data: bytes) -> List[int]:
        """Take two following bytes to build a integer (using little endianess) """
        assert len(data) % 2 == 0
        return [data[i] + data[i + 1] * 256 for i in range(0, len(data), 2)]

    def __to_per_channel_lists(self, shorts: List[int], channel_count: int = 8) -> List[List[int]]:
        """Create a list (of the size of 'channel_count') of lists, 
        where the list at position x contains the data for channel x+1 of the hantek device """
        return [shorts[i::channel_count] for i in range(0, channel_count)]

    def __shorts_to_volt(self, shorts: List[int], channel_id: int) -> List[float]:
        """Convert the raw shorts to usefull volt values"""
        vscale = 1.0
        zero_offset = 2048

        if channel_id < len(self.__vertical_scale_factors):
            vscale = self.__vertical_scale_factors[channel_id]
            # get right zero offset for that channel and the used vertical scale factor (vscale)
            # zero_offset = self.__zero_offsets[vscale][channel_id]
            zero_offset = self.__get_zero_offset(channel_id, vscale)

        scale = 0.01 * vscale

        # math.log10(4096/2)
        accuracy = -int(math.log10(scale)) + 2  # amount of digits after the dot that is not nearly random
        return [round(
            self.__calc_correction_factor(v - zero_offset, channel_id, vscale)
            * (v - zero_offset) * scale
            , ndigits=accuracy)
            for v in shorts]

    def __calc_correction_factor(self, delta_to_zero: float, channel_id: float, vscale: float) -> float:
        if channel_id >= 8 or vscale not in self.__correction_data[channel_id]:
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
        # log.info((1.0 - alpha) * cfactor_less + alpha * cfactor_greater)
        return (1.0 - alpha) * cfactor_less + alpha * cfactor_greater