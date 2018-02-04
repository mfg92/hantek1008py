import re
from typing import List, Callable
import lzma
import time
import math
from abc import ABCMeta, abstractmethod
import argparse

measured_sampling_rate_regex = re.compile(r"(# measured samplingrate:)\s+(\d*\.?\d+)\s*(hz)",
                                          re.IGNORECASE)
sampling_rate_regex = re.compile(r"(# samplingrate:)\s+(\d*\.?\d+)\s*(hz)", re.IGNORECASE)
unix_time_regex = re.compile(r"(# unix-time:)\s+(\d*\.?\d+)\s*", re.IGNORECASE)


def csv_file_type(file_path: str):
    """for use in argparse as type="""
    try:
        return open_csv_file(file_path)
    except:
        raise argparse.ArgumentTypeError(f"There is no file '{file_path}' or can not open it.")


def parse_csv_lines(lines: List[str]) -> (List[float], List[float], List[float], List[List[float]]):
    measured_sampling_rate = [float(measured_sampling_rate_regex.search(line).group(2))
                              for line in lines
                              if line[0] == "#" and measured_sampling_rate_regex.search(line)]

    sampling_rate = [float(sampling_rate_regex.search(line).group(2)) for line in lines
                     if line[0] == "#" and sampling_rate_regex.search(line)]

    unix_time = [float(unix_time_regex.search(line).group(2)) for line in lines
                 if line[0] == "#" and unix_time_regex.search(line)]

    values = [[float(v) for v in line.split(",")]
              for line in lines
              if line[0] != "#"]

    per_channel_data = list(zip(*values))

    return sampling_rate, measured_sampling_rate, unix_time, per_channel_data


def open_csv_file(file_name: str):
    open_function = lzma.open if file_name.endswith(".xz") else open
    return open_function(file_name, mode="rt")


def read_csv_file(file_name: str) -> List[str]:
    with open_csv_file(file_name) as f:
        return f.readlines()


def parse_csv_file(file_name: str) ->[List[float], List[float], List[float], List[List[float]]]:
    """
    Parse a file chunk for chunk: reading up to chunk_size bytes, parse them
    and finally merge all data of all pared chunks together
    :param file_name:
    :return:
    """
    sampling_rate, measured_sampling_rate, unix_time, per_channel_data = [], [], [], None

    def on_parse_func(sampling_rate_part, measured_sampling_rate_part, unix_time_part, per_channel_data_part):
        nonlocal sampling_rate, measured_sampling_rate, unix_time, per_channel_data

        only_comments = False if per_channel_data_part else True

        if per_channel_data is None and not only_comments:
            per_channel_data = [[] for _ in range(len(per_channel_data_part))]

        sampling_rate.extend(sampling_rate_part)
        measured_sampling_rate.extend(measured_sampling_rate_part)
        unix_time.extend(unix_time_part)
        if not only_comments:
            assert len(per_channel_data) == len(per_channel_data_part)
            for index in range(len(per_channel_data_part)):
                per_channel_data[index].extend(per_channel_data_part[index])

    parse_csv_file_chunked(file_name, on_parse_func)
    return sampling_rate, measured_sampling_rate, unix_time, per_channel_data


def parse_csv_file_chunked(file_name: str, on_parse_func: Callable[[List[float], List[float], List[float], List[List[float]]], None],
                           chunk_size: int = 2**10):
    with open_csv_file(file_name) as f:
        while True:
            lines_part = f.readlines(chunk_size)

            if len(lines_part) == 0:
                break

            sampling_rate_part, measured_sampling_rate_part, unix_time_part, per_channel_data_part = parse_csv_lines(lines_part)

            on_parse_func(sampling_rate_part, measured_sampling_rate_part, unix_time_part, per_channel_data_part)


class FileChangeReader:
    def __init__(self, file_path: str, ignore_existing_file_content: bool = True):
        self.__file_path: str = file_path
        self.__stream_position: int = 0

        if ignore_existing_file_content:
            with self.__open()(self.__file_path, "r") as file:
                file.seek(0, 2)  # jump to the end
                self.__stream_position = file.tell()

    def __open(self):
        return open if not self.__file_path.endswith(".xz") else lzma.open

    def read_changed_lines(self) -> List[str]:
        with self.__open()(self.__file_path, "r") as file:
            file.seek(0, 2)
            if file.tell() == self.__stream_position:
                return []  # file size did not change

            file.seek(self.__stream_position, 0)
            lines = file.readlines()
            self.__stream_position = file.tell()
            return lines


class ChannelDataUpdater(metaclass=ABCMeta):
    @abstractmethod
    def get_channel_data(self, channel_id: int) -> List[float]:
        return []

    @abstractmethod
    def update(self):
        pass


class CsvChannelDataUpdater(ChannelDataUpdater):
    def __init__(self, file: FileChangeReader, buffer_size: int):
        self.__file: FileChangeReader = file
        self.__buffer_size: int = buffer_size
        self.__channel_data: List[List[float]] = [[] for _ in range(8)]
        self.__sampling_rate: float = None

    def get_channel_data(self, channel_id: int) -> List[float]:
        return self.__channel_data[channel_id]

    def update(self):
        lines = self.__file.read_changed_lines()
        if not lines:
            return

        sampling_rate, measured_sampling_rate, per_channel_data = parse_csv_lines(lines)

        self.__sampling_rate = [self.__sampling_rate, *sampling_rate, *measured_sampling_rate][-1]

        for channel_id in range(8):
            if channel_id < len(per_channel_data):
                self.__channel_data[channel_id].extend(per_channel_data[channel_id])
                # trim to self.__channel_data_max_len
                del self.__channel_data[channel_id][:-self.__buffer_size]


class DemoChannelDataUpdater(ChannelDataUpdater):
    def __init__(self, sampling_rate: int, buffer_size: int):
        self.__sampling_rate: int = sampling_rate
        self.__buffer_size: int = buffer_size
        self.__channel_data: List[List[float]] = [[] for _ in range(8)]
        self.__time_of_last_update: float = time.time()

    def get_channel_data(self, channel_id: int) -> List[float]:
        return self.__channel_data[channel_id]

    def update(self):
        now = time.time()
        delta = now - self.__time_of_last_update  # time passed since last demo calculations
        for channel_id in range(8):
            for i in range(round(delta * self.__sampling_rate)):
                t = self.__time_of_last_update + (i / self.__sampling_rate)
                amplitude = 0.2 + 1.8 * (channel_id / 8)
                # x_scale = 3 + 12.987 * (1 / 8) * channel_id
                x_scale = 10
                # x_offset = channel_id * 42.123
                x_offset = 0
                if channel_id == 7:
                    x_offset = 0.5/8
                y = amplitude * math.sin((x_offset + t*x_scale) * 2*math.pi)
                if channel_id == 0:
                    # y += amplitude*0.5 * math.sin(x_offset + t * math.pi * x_scale*3)
                    # y += amplitude*0.5 * math.sin((x_offset+0.5 + t*x_scale*2) * math.pi)
                    y += amplitude*0.5 * math.sin((x_offset+0 + t*x_scale*2) * 2*math.pi)

                self.__channel_data[channel_id].append(y)
            # trim to self.__channel_data_max_len
            del self.__channel_data[channel_id][:-self.__buffer_size]
        self.__time_of_last_update = now


class EmptyChannelDataUpdater(ChannelDataUpdater):
    def get_channel_data(self, channel_id: int) -> List[float]:
        return []

    def update(self):
        pass
