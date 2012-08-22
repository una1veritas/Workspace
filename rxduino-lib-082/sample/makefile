PROJECTS=_adc _accel _analogWrite _ethercapture _gpio_hal _gpio_rxduino _serial_rxduino _serial_hal _math_and_printf _rtc _KXM52 _AE-GYRO-SMD _fatfs _liquidcrystal _tcpip _pwm _tone _nahimon _multisci
MAKE=make --no-print-directory 

all:	$(PROJECTS)

_adc:
	cd adc && $(MAKE) 
	echo ""

_accel:
	cd accel && $(MAKE) 
	echo ""

_analogWrite:
	cd analogWrite && $(MAKE) 
	echo ""

_ethercapture:
	cd ethercapture && $(MAKE) 
	echo ""

_gpio_hal:
	cd gpio_hal && $(MAKE) 
	echo ""

_gpio_rxduino:
	cd gpio_rxduino && $(MAKE) 
	echo ""

_serial_rxduino:
	cd serial_rxduino && $(MAKE) 
	echo ""

_serial_hal:
	cd serial_hal && $(MAKE) 
	echo ""

_math_and_printf:
	cd math_and_printf && $(MAKE) 
	echo ""

_rtc:
	cd rtc && $(MAKE) 
	echo ""

_nokialcd:
	cd nokialcd && $(MAKE) 
	echo ""

_oled:
	cd oled && $(MAKE) 
	echo ""

_fatfs:
	cd fatfs && $(MAKE) 
	echo ""

_liquidcrystal:
	cd liquidcrystal && $(MAKE) 
	echo ""

_KXM52:
	cd KXM52 && $(MAKE) 
	echo ""

_AE-GYRO-SMD:
	cd AE-GYRO-SMD && $(MAKE) 
	echo ""
	
_fractal:
	cd fractal && $(MAKE) 
	echo ""

_tcpip:
	cd tcpip && $(MAKE) 
	echo ""

_pwm:
	cd pwm && $(MAKE) 
	echo ""

_tone:
	cd tone && $(MAKE) 
	echo ""

_nahimon:
	cd nahimon && $(MAKE) 
	echo ""

_multisci:
	cd multisci && $(MAKE) 
	echo ""

clean:
	rm -f common/*.o
	cd adc && $(MAKE) clean
	cd accel && $(MAKE) clean
	cd analogWrite && $(MAKE) clean
	cd ethercapture && $(MAKE) clean
	cd gpio_hal && $(MAKE) clean
	cd gpio_rxduino && $(MAKE) clean
	cd math_and_printf && $(MAKE) clean
	cd rtc && $(MAKE) clean
	cd nokialcd && $(MAKE) clean
	cd oled && $(MAKE) clean
	cd serial_hal && $(MAKE) clean
	cd serial_rxduino && $(MAKE) clean
	cd KXM52 && $(MAKE) clean
	cd AE-GYRO-SMD && $(MAKE) clean
	cd liquidcrystal && $(MAKE) clean
	cd fatfs && $(MAKE) clean
	cd fractal && $(MAKE) clean
	cd tcpip && $(MAKE) clean
	cd pwm && $(MAKE) clean
	cd tone && $(MAKE) clean
	cd multisci && $(MAKE) clean
	rm -R -f *.o
	rm -R -f *.elf
