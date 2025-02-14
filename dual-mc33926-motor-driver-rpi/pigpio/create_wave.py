#!/usr/bin/env python

# create_wave.py
# 2019-11-18
# Public Domain

usage = """
A wave file and a bit time must be specified.  The wave file has one or
more lines.  Blank lines and lines starting with a # are ignored.  Other
lines define a GPIO and the level changes required on that GPIO.  The bit
time is the number of microseconds between level changes.

# GPIO  levels
23      11000001
11      01110000
12      00011100
4       00000111

You can also specify one of py, c, or pdif: the script output will then
be a complete program to generate the wave (py for Python script, c for
a C program, pdif for a C program using the pigpio daemon I/F).

If none of py, c, or pdif are chosen the waveform will be generated for
30 seconds.

To generate a pdif program with a bit time of 100 microseconds
./create_wave.py wave_file 100 pdif >wave_pdif.c

To just transmit the wave with a bit time of 50 microseconds
./create_wave.py wave_file 50
"""

import sys

argc = len(sys.argv)

if argc < 3:
   print(usage)
   exit()

gpios=[]
pulses=[]

with open(sys.argv[1], 'r') as f:
   edges=[]
   num_bits = 0
   for line in f:
      if len(line) > 1 and line[0] != '#':
         fields = line.split()
         gpio = int(fields[0])
         gpios.append(gpio)
         levels = int(fields[1], 2)
         edges.append([1<<gpio, levels])
         if len(fields[1]) > num_bits:
            num_bits = len(fields[1])

bit_time = int(sys.argv[2])

for bit in range(num_bits-1, -1, -1):
   on = 0
   off = 0
   for e in edges:
      if (1<<bit) & e[1]:
         on |= e[0]
      else:
         off |= e[0]
   pulses.append((on, off, bit_time))

if "py" in (str(i).lower() for i in sys.argv):

   print("""#!/usr/bin/env python

import time
import pigpio

pi = pigpio.pi()
if not pi.connected:
   exit()""")

   s = "\ngpios = ["
   for g in gpios:
      s = s + str(g) + ","
   s += "]"
   print(s)

   print("\npulses = [")
   for p in pulses:
      print("   (0x{:x}, 0x{:x}, {}), ".format(p[0], p[1], p[2]))
   print("]")

   print("""
for g in gpios:
   pi.set_mode(g, pigpio.OUTPUT)

wf = []
for p in pulses:
   wf.append(pigpio.pulse(p[0], p[1], p[2]))

pi.wave_clear()
pi.wave_add_generic(wf)

wid = pi.wave_create()
if wid >= 0:
   pi.wave_send_repeat(wid)
   time.sleep(30)
   pi.wave_tx_stop()
   pi.wave_delete(wid)

pi.stop()
""")

elif "c" in (str(i).lower() for i in sys.argv):

   print("""#include <stdio.h>

#include <pigpio.h>

/*
gcc -pthread -o wave wave.c -lpigpio
*/
""")

   s = "int gpios[] = {"
   for g in gpios:
      s = s + str(g) + ","
   s += "};"
   print(s)

   print("\ngpioPulse_t pulses[] =\n{")
   for p in pulses:
      print("   {{0x{:x}, 0x{:x}, {}}}, ".format(p[0], p[1], p[2]))
   print("};")

   print("""
int main(int argc, char *argv[])
{
   int g, wid=-1;

   if (gpioInitialise() < 0) return 1;

   for (g=0; g<sizeof(gpios)/sizeof(gpios[0]); g++)
      gpioSetMode(gpios[g], PI_OUTPUT);

   gpioWaveClear();
   gpioWaveAddGeneric(sizeof(pulses)/sizeof(pulses[0]), pulses);
   wid = gpioWaveCreate();
   if (wid >= 0)
   {
      gpioWaveTxSend(wid, PI_WAVE_MODE_REPEAT);
      time_sleep(30);
      gpioWaveTxStop();
      gpioWaveDelete(wid);
   }

   gpioTerminate();
}
""")

elif "pdif" in (str(i).lower() for i in sys.argv):

   print("""#include <stdio.h>

#include <pigpiod_if2.h>

/*
gcc -pthread -o wave wave.c -lpigpiod_if2
*/
""")

   s = "int gpios[] = {"
   for g in gpios:
      s = s + str(g) + ","
   s += "};"
   print(s)

   print("\ngpioPulse_t pulses[] =\n{")
   for p in pulses:
      print("   {{0x{:x}, 0x{:x}, {}}}, ".format(p[0], p[1], p[2]))
   print("};")

   print("""
int main(int argc, char *argv[])
{
   int pi, g, wid=-1;

   pi = pigpio_start(0, 0); /* Connect to local Pi. */

   if (pi < 0)
   {
      printf("Can't connect to pigpio daemon\\n");
      return 1;
   }

   for (g=0; g<sizeof(gpios)/sizeof(gpios[0]); g++)
      set_mode(pi, gpios[g], PI_OUTPUT);

   wave_clear(pi);
   wave_add_generic(pi, sizeof(pulses)/sizeof(pulses[0]), pulses);
   wid = wave_create(pi);
   if (wid >= 0)
   {
      wave_send_using_mode(pi, wid, PI_WAVE_MODE_REPEAT);
      time_sleep(30);
      wave_tx_stop(pi);
      wave_delete(pi, wid);
   }

   pigpio_stop(pi);
}
""")

else:

   import time
   import pigpio

   pi = pigpio.pi()
   if not pi.connected:
      exit()

   for g in gpios:
      pi.set_mode(g, pigpio.OUTPUT)

   wf = []
   for p in pulses:
      wf.append(pigpio.pulse(p[0], p[1], p[2]))

   pi.wave_clear()
   pi.wave_add_generic(wf)

   wid = pi.wave_create()
   if wid >= 0:
      pi.wave_send_repeat(wid)
      time.sleep(30)
      pi.wave_tx_stop()
      pi.wave_delete(wid)

   pi.stop()

