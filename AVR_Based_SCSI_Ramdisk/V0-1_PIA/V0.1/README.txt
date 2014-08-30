
SCSI RAM-Disk PIA firmware V0.1
===============================
This is the firmware for the PIA (Parallel Interface Agent) on the selfmade
SCSI RAM-Disk. It implements the "PIA" module which (physically) communicate
with the initiator via the PHY module. Normally the initiator is a hostadapter.


Hardware requirements
---------------------
- RAM-Disk hardware V1.0.1 compatible PHY module
  Atmel ATmega165-16 or ATmega169-16 processor


Build requirements
------------------
- UNIX compatible host system (I have used GNU/Linux 2.2.26)
- Assembler tavrasm V1.22
- GENtoHEX V0.4 file format converter for Intel HEX format (optional)

Type 'make all' (or 'make gen' if 'GENtoHEX' is not installed) to build the
firmware.

The tavrasm list file can be found in the './bin' directory.


Compatibility
-------------
This firmware was developed to comply with the SCSI3 standard ANSI X3.301:1997.

It should also be compatible with all initiators that comply to the SCSI2
standard ANSI X3.131:1994 and the SCSI1 standard ANSI X3.131:1986 (Parity must
be supported by the initiator).


Notes
-----
SCSI3 specify a selection abort time of 200us (Maximum time for the target to
take over bus control), the PIA defaults to 192us (after this time, the ACCEPT
SELECTION command reports an error). This time may be too short for the target
module if the debug system is enabled. For historically reasons, the selection
timeout delay (the time the initiator should wait for the target to take over
bus control) is recommended to be 250ms.  Most initiators wait at least 64ms so
this value can be increased in 'init.s' up to 16ms.


More info can be found in the './doc' directory.


2005-07-22  Michael Baeuerle <micha@hilfe-fuer-linux.de>


EOF
