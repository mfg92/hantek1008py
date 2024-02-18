# Hantek1008Driver

This project provides tooling for using Hantek 1008 USB-oscilloscopes
without proprietary software on Linux and Windows (not tested).
You can include the Hantek1008 class from 'hantek1008.py' in your project
to get access to the features of the device.
Alternatively use the csvexport.py Python application to gain data and save it to a file.

This project is based on careful reverse engineering of the device's USB protocol.
The reverse engineering was only done to the extent necessary to obtain data for my master's thesis and
does not cover all the features and configuration options of the device.
Therefore, no guarantees can be made as to the reliability or accuracy of the data collected.

### Usageexample of csvexport.py:
`python3 csvexport.py mydata.csv -s 1 2`
This will write the measured data of channel 1 and 2 to 'mydata.csv' until you press CTRL+C to stop the measurement.

### Help Options:
`python3 csvexport.py --help`
This will show you all the available options/parameters and explains them in-depth.

### Notes:
* Requires Python >= 3.6
* Requires *pyusb* and *overrides* (install it using pip: `pip3 install pyusb overrides`)
* If the software can not access the usb device because of lacking accessright, do the following (tested on linux/fedora):
  1. Create file "/etc/udev/rules.d/99-hantek1008.rules" with content:
     ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="0783", ATTR{idProduct}=="5725", MODE="0666"
  2. Then `sudo udevadm control -R`
  3. Replug the device
* The code contains many assert statements.
  They exist because I noticed at the time that the respective responses on my device were always the same.
  I was not able (nor was there any need) to find out what these responses meant,
  but I wanted to be notified if the response changed for any reason, hence the assert statements.
  With a different copy of the device, you might get different answers. So some asserts may fail.
  Therefore, it might be necessary to remove or adapt these assert statements.
