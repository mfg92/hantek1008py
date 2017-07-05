#!/bin/python3

from hantek1008 import Hantek1008
from hantek1008 import CorrectionDataType
from typing import Optional, List
import logging as log
import argparse
import time
import os
import lzma
import sys
import csv


def main(csv_file_path: str,
         selected_channels: Optional[List[int]]=None,
         vertical_scale_factor: Optional[List[float]]=1.0,
         roll_mode: bool=True,
         calibrate_output_file_path: Optional[str]=None,
         calibration_file_path: Optional[str]=None,
         zero_offset_shift_compensation_channel: Optional[int]=None,
         raw: bool=False,
         sampling_rate: int=440) -> None:

    if selected_channels is None or len(selected_channels) == 0:
        selected_channels = list(range(1, 9))

    assert len(set(selected_channels)) == len(selected_channels)
    assert all(1 <= c <= 8 for c in selected_channels)
    selected_channels = [i-1 for i in selected_channels]

    assert zero_offset_shift_compensation_channel is None or 1 <= zero_offset_shift_compensation_channel <= 8
    if zero_offset_shift_compensation_channel is not None:
        zero_offset_shift_compensation_channel -= 1

    assert vertical_scale_factor is None or isinstance(vertical_scale_factor, List)
    if vertical_scale_factor is None:
        vertical_scale_factor = [1.0] * 8
    elif len(vertical_scale_factor) == 1:
        vertical_scale_factor = [1.0 if i not in selected_channels
                                 else vertical_scale_factor[0]
                                 for i in range(8)]
    else:
        assert len(vertical_scale_factor) == len(selected_channels)
        # the vscale value of a channel is the value in vertical_scale_factor
        # on the same index as the channel in selected channel
        # or 1.0 if the channel is not in selected_channels
        vertical_scale_factor = [1.0 if i not in selected_channels
                                 else vertical_scale_factor[selected_channels.index(i)]
                                 for i in range(8)]

    correction_data: CorrectionDataType = [{} for _ in range(8)]  # list of dicts of dicts
    # usecase: correction_data[channel_id][vscale][units] = correction_factor

    if calibration_file_path:
        if not os.path.exists(calibration_file_path):
            log.error(f"There is no file '{calibration_file_path}'.")
            sys.exit(1)
        if os.path.isdir(calibration_file_path):
            log.error(f"'{calibration_file_path}' is a directory.")
            sys.exit(1)
        with open(calibration_file_path) as f:
            import json
            calibration_data = json.load(f)
        # print(calibration_data)

        log.info(f"Using calibration data from file '{calibration_file_path}' to correct measured values")

        for channel_id, channel_cdata in sorted(calibration_data.items()):
            channel_id = int(channel_id)
            if len(channel_cdata) == 0:
                continue
            log.info(f"  Channel {channel_id+1}:")
            for test in channel_cdata:
                vscale = test["vscale"]
                test_voltage = test["test_voltage"]
                units = test["measured_value"] - test["zero_offset"]
                correction_factor = test_voltage / (units * 0.01 * vscale)

                log.info(f"    {test} -> {correction_factor}")

                if vscale not in correction_data[channel_id]:
                    correction_data[channel_id][vscale] = {}

                correction_data[channel_id][vscale][units] = correction_factor

        # log.info("\n".join(str(x) for x in correction_data))
        channels_without_cd = [i + 1 for i, x in enumerate(correction_data) if len(x) == 0]
        if len(channels_without_cd) > 0:
            log.warning(f"There is no calibration data for channel(s): {channels_without_cd}")

    device = Hantek1008(ns_per_div=1_000_000,
                        vertical_scale_factor=vertical_scale_factor,
                        correction_data=correction_data,
                        zero_offset_shift_compensation_channel=zero_offset_shift_compensation_channel)

    try:
        log.info("Connecting...")
        try:
            device.connect()
        except RuntimeError as e:
            log.error(e)
            sys.exit(1)
        log.info("Connection established")

        log.info("Initialising...")
        try:
            device.init()
        except RuntimeError as e:
            log.error(e)
            sys.exit(1)
        log.info("Initialisation completed")
    except KeyboardInterrupt:
        device.close()
        sys.exit(1)
        pass

    if calibrate_output_file_path:
        calibration_routine(device, calibrate_output_file_path)
        device.close()
        quit()

    log.info(f"Processing data of channel{'' if len(selected_channels) == 1 else 's'}:"
             f" {' '.join([str(i+1) for i in selected_channels])}")

    """
    # measure sample rate
    start_time = time.time()
    counter = 0
    for data in device.request_samples_roll_mode():
        #print(data[-1+1])
        counter += len(data[0])
        if counter >= 1000:
            break

    duration = time.time()-start_time
    print("duration: {}, samples: {}".format(duration, counter))
    print("-> {}".format(counter/duration))
    quit()
    """

    try:
        # output_csv_filename = "channel_data.csv"
        if csv_file_path == '-':
            log.info("Exporting data to stdout...")
            csv_file = sys.stdout
        elif csv_file_path.endswith(".xz"):
            log.info(f"Exporting data lzma-compressed to file '{csv_file_path}'...")
            csv_file = lzma.open(csv_file_path, 'at', newline='')
        else:
            log.info(f"Exporting data to file '{csv_file_path}'...")
            csv_file = open(csv_file_path, 'at', newline='')
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_file.write(f"# {', '.join([ f'ch_{i + 1}' for i in selected_channels])}\n")
        csv_file.write("# Calibration data:\n")
        for vscale, zero_offset in sorted(device.get_calibration_data().items()):
            csv_file.write(f"# zero_offset [{vscale:<4}]: {' '.join([str(round(v, 1)) for v in zero_offset])}\n")

        if roll_mode:
            for channel_data in device.request_samples_roll_mode(raw=raw, sampling_rate=sampling_rate):
                channel_data = [d for i, d in enumerate(channel_data) if i in selected_channels]
                milli_volt_int_representation = False
                if milli_volt_int_representation:
                    channel_data = [[f"{round(value*1000)}" for value in single_channel]
                                    for single_channel in channel_data]
                csv_writer.writerows(zip(*channel_data))
                csv_file.write(f"# UNIX-Time: {time.time()}\n")
        else:
            while True:
                channel_data2, channel_data3 = device.request_samples_normal_mode()

                print(len(channel_data2[0]), len(channel_data3[0]))
                # channel_data = [cd2 + cd3 for cd2, cd3 in zip(channel_data2, channel_data3)]
                # channel_data = [cd[70:] + cd[:70] for cd in channel_data]

                csv_writer.writerows(zip(*channel_data2))
                csv_writer.writerows(zip(*channel_data3))
                csv_file.write(f"# UNIX-Time: {time.time()}\n")
    except KeyboardInterrupt:
        log.info("Sample collection was canceled by user")
        pass

    csv_file.close()
    log.info("Exporting data finished")

    device.close()


def calibration_routine(device: Hantek1008, calibrate_file_path: str):
    print("This interactive routine will generate a calibration that can later be used "
          "to get more precise results. It works by connecting different well known "
          "voltages one after another to a channel. Once all calibration voltages are "
          "measured, the same is done for every other channel.")

    import json
    required_calibration_samples_nun = 512
    calibration_data = {}  # dictionary of lists
    device.pause()

    test_voltages = None
    while test_voltages is None:
        try:
            in_str = input("Calibration voltages (x, y, z, ...): ")
            test_voltages = [float(v) for v in in_str.split(',')]
            if len(test_voltages) < 1:
                print("Input must contain at least one voltage")
        except ValueError:
            print("Input must be comma separated floats")

    print(f"Calibration voltages are: {' '.join([ f'{v}V' for v in test_voltages])}")

    for channel_id in range(8):

        calibration_data[channel_id] = []

        for test_voltage in test_voltages:
            cmd = input(f"Do {test_voltage}V measurement on channel {channel_id+1} (Enter),"
                        f" skip voltage (s), skip channel (ss) or quit (q): ")
            if cmd == 'q':
                return
            elif cmd == 'ss':
                break
            elif cmd == 's':
                continue

            device.cancel_pause()

            print(f"Measure {required_calibration_samples_nun} values for {test_voltage}V...")
            data = []
            for _, row in zip(
                    range(required_calibration_samples_nun),
                    device.request_samples_roll_mode_single_row(raw=True)):
                data.append(row)
                pass

            device.pause()

            channel_data = list(zip(*data))
            cd = channel_data[channel_id]
            avg = sum(cd) / len(cd)

            calibration_data[channel_id].append({
                "test_voltage": test_voltage,
                "measured_value": round(avg, 2),
                "vscale": device.get_vscales()[channel_id],
                "zero_offset": round(device.get_zero_offset(channel_id=channel_id), 2)
            })
            #    "channel_data": channel_data})

    with open(calibrate_file_path, 'w') as calibration_file:
        calibration_file.write(json.dumps(calibration_data))


if __name__ == "__main__":

    def channel_type(value):
        ivalue = int(value)
        if 1 <= ivalue <= 8:
            return ivalue
        raise argparse.ArgumentTypeError(f"There is no channel {value}")

    str_to_log_level = {log.getLevelName(ll).lower(): ll for ll in [log.DEBUG, log.INFO, log.WARN]}

    parser = argparse.ArgumentParser(description="Collect data from device 'Hantek 1008'")
    parser.add_argument(metavar='csv_path', dest='csv_path',
                        type=str, default=None,
                        help='Exports measured data to the given file in CSV format.'
                             " If filename ends with '.xz' the content is compress using lzma/xz."
                             " This reduces the filesize to ~ 1/12 compered to the uncompressed format."
                             " Those files can be decompressed using 'xz -dk <filename>")
    parser.add_argument('-s', '--channels', metavar='channel', nargs='+',
                        type=channel_type,
                        help="Select channels that are of interest")
    parser.add_argument('-l', '--loglevel', dest='log_level', nargs='?',
                        type=str, default="info", choices=str_to_log_level.keys(),
                        help='Set the loglevel to debug')
    parser.add_argument('-v', '--vscale', metavar='scale', nargs="+",
                        type=float, default=1.0, choices=Hantek1008.valid_vscale_factors(),
                        help='Set the pre scale in the hardware, must be 1, 0.125, or 0.02. If one value is given, all '
                             'channels will use that vscale, otherwise there must be one value per selected channel')
    parser.add_argument('--calibrate', metavar='calibrationfile_path',
                        type=str, default=None,
                        help='If set calibrates the device and write calibration values to given file.'
                             ' Ignores all other args')
    parser.add_argument('-c', '--calibrationfile', dest="calibration_file_path", metavar='calibrationfile_path',
                        type=str, default=None,
                        help="Use the content of the given calibration file to correct the measured samples")
    parser.add_argument('-r', '--raw', dest="raw",
                        action='store_const', default=False, const=True,
                        help='Outputs the sample values return from the device, without transformation'
                             ' to volts or any calibration data used')
    parser.add_argument('-z', '--zoscompensation', dest="zos_compensation", metavar='channel',
                        type=channel_type, default=None, const=8, nargs='?',
                        help='Compensate the zero offset shift that obscures over longer timescales. Needs at least'
                             ' one unused channel, make sure that no voltage is applied to the given channel. '
                             'Defaults to no compensation, if used without an argument channel 8 is used')
    parser.add_argument('-a', '--samplingrate', dest='sampling_rate',
                        type=int, default=440, choices=Hantek1008.valid_roll_sampling_rates(),
                        help='Set the sampling rate of the device in Hz (default:440)')

    args = parser.parse_args()

    args.log_level = str_to_log_level[args.log_level]

    def arg_assert(ok, fail_message):
        if not ok:
            parser.print_usage()
            print(fail_message, file=sys.stderr)
            sys.exit(1)

    arg_assert(len(args.vscale) == 1 or len(args.vscale) == len(args.channels),
               "There must be one vscale factor or as many as selected channels")
    arg_assert(len(set(args.channels)) == len(args.channels),
               "Selected channels list is not a set (multiple occurrences of the same channel id")
    arg_assert(args.calibration_file_path is None or not args.raw,
               "--calibrationfile can not be used together with the --raw flag")
    arg_assert(args.zos_compensation is None or not args.raw,
               "--zoscompensation can not be used together with the --raw flag")
    arg_assert(args.zos_compensation is None or len(args.channels) < 8,
               "Zero-offset-shift-compensation is only possible if there is at least one unused channel")
    arg_assert(args.zos_compensation is None or args.zos_compensation not in args.channels,
               f"The channel {args.zos_compensation} is used for Zero-offset-shift-compensation,"
               f" but it is also a selected channel")

    log.basicConfig(level=args.log_level, format='%(levelname)-7s: %(message)s')

    main(selected_channels=args.channels,
         vertical_scale_factor=args.vscale,
         csv_file_path=args.csv_path,
         calibrate_output_file_path=args.calibrate,
         calibration_file_path=args.calibration_file_path,
         raw=args.raw,
         zero_offset_shift_compensation_channel=args.zos_compensation,
         sampling_rate=args.sampling_rate)
