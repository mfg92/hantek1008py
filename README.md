# Hantek1008Driver

This project provides the ability to use Hantek 1008 USB-oscilloscopes without the proprietary software on Linux and Windows (not tested). You can include the Hantek1008 from 'hantek1008.py' class in your project to get access to the features of the device or use the csvexport.py Python application to gain data and save it to a file.

### Usageexample of csvexport.py:
`python3 csvexport.py mydata.csv -s 1 2`
This will write the measured data of channel 1 and 2 to 'mydata.csv' until you press CTRL+C to stop the measurement.


### Notes:
* Requires Python >= 3.6
* Requires *pyusb* and *overrides* (install it using pip: `pip3 install pyusb overrides`)
* If the software can not access the usb device because of lacking accessright, do the following (tested on linux/fedora):
  1. Create file "/etc/udev/rules.d/99-hantek1008.rules" with content:
     ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="0783", ATTR{idProduct}=="5725", MODE="0666"
  2. Then `sudo udevadm control -R`
  3. Replug the device

### TODO:
* Usage example of the Hantek1008 class
* Better comments in source
