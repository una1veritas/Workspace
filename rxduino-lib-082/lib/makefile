PROJECTS= _fatfs _liquidcrystal _spi _servo _sdmmc _eeprom
#_nokialcd _oled 
MAKE=make --no-print-directory
all:	$(PROJECTS)

_fatfs:
	cd fatfs && $(MAKE) "TARGET_LIB=$(TARGET_LIB)"
	echo ""

_nokialcd:
	cd nokialcd && $(MAKE) "TARGET_LIB=$(TARGET_LIB)"
	echo ""

_oled:
	cd oled && $(MAKE) "TARGET_LIB=$(TARGET_LIB)"
	echo ""

_liquidcrystal:
	cd liquidcrystal && $(MAKE) "TARGET_LIB=$(TARGET_LIB)"
	echo ""

_spi:
	cd spi && $(MAKE) "TARGET_LIB=$(TARGET_LIB)"
	echo ""

_servo:
	cd servo && $(MAKE) "TARGET_LIB=$(TARGET_LIB)"
	echo ""

_sdmmc:
	cd sdmmc && $(MAKE) "TARGET_LIB=$(TARGET_LIB)"
	echo ""

_eeprom:
	cd eeprom && $(MAKE) "TARGET_LIB=$(TARGET_LIB)"
	echo ""

uselib:
	cp tgtbrd/libtkdnhal.$(TARGET_LIB).a libtkdnhal.a 
	cp tgtbrd/librxduino.$(TARGET_LIB).a librxduino.a 
	cp tgtbrd/libfatfs.$(TARGET_LIB).a libfatfs.a 
	cp tgtbrd/libliquidcrystal.$(TARGET_LIB).a libliquidcrystal.a 
	cp tgtbrd/libspi.$(TARGET_LIB).a libspi.a 
	cp tgtbrd/libservo.$(TARGET_LIB).a libservo.a 
	cp tgtbrd/libsdmmc.$(TARGET_LIB).a libsdmmc.a 
	cp tgtbrd/libeeprom.$(TARGET_LIB).a libeeprom.a 

#	cp tgtbrd/libnokialcd.$(TARGET_LIB).a libnokialcd.a 
#	cp tgtbrd/liboled.$(TARGET_LIB).a liboled.a 

clean:
	cd fatfs && $(MAKE) clean
	cd nokialcd && $(MAKE) clean
	cd oled && $(MAKE) clean
	cd liquidcrystal && $(MAKE) clean

allclean:
	rm -f *.a 
	rm -f *.o 
	rm -f tgtbrd/*.a 
	rm -f tgtbrd/*.o 
	cd fatfs && $(MAKE) clean
	cd nokialcd && $(MAKE) clean
	cd oled && $(MAKE) clean
	cd liquidcrystal && $(MAKE) clean

