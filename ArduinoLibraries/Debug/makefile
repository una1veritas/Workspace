################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include libraries/glcd/examples/clockFace/subdir.mk
-include libraries/glcd/examples/GLCD_BigDemo/subdir.mk
-include libraries/glcd/subdir.mk
-include libraries/WaveHC/subdir.mk
-include libraries/UsbKeyboard/subdir.mk
-include libraries/UartBridge_SC16IS750/subdir.mk
-include libraries/USB_Host_Shield/examples/HID/le3dp/subdir.mk
-include libraries/USB_Host_Shield/examples/HID/USBHIDJoystick/subdir.mk
-include libraries/USB_Host_Shield/subdir.mk
-include libraries/TouchScreen/subdir.mk
-include libraries/Tlc5940/subdir.mk
-include libraries/TimerOne/subdir.mk
-include libraries/TimerCounter/subdir.mk
-include libraries/TextStream/subdir.mk
-include libraries/TFTLCD/subdir.mk
-include libraries/Sparkfun6DOF/subdir.mk
-include libraries/SerialSRAM_SPI/subdir.mk
-include libraries/SerialRAM/examples/Kerry_D_Wong/subdir.mk
-include libraries/SerialRAM/subdir.mk
-include libraries/Sensirion/subdir.mk
-include libraries/SdFat/subdir.mk
-include libraries/SD_SPI/utility/subdir.mk
-include libraries/SD_SPI/subdir.mk
-include libraries/SCP1000D01/subdir.mk
-include libraries/Roomba500/subdir.mk
-include libraries/Roomba/subdir.mk
-include libraries/RTC_M41T62/subdir.mk
-include libraries/RTC_DS3234/subdir.mk
-include libraries/RTC_DS1307/subdir.mk
-include libraries/RFID_StrongLink_UART/subdir.mk
-include libraries/RFID_StrongLink/subdir.mk
-include libraries/RFID_RCS620S/subdir.mk
-include libraries/RFID_PaSoRi/subdir.mk
-include libraries/RFID_PN532_SPI/subdir.mk
-include libraries/RFID_PN532_I2C/subdir.mk
-include libraries/Pololu/PololuWheelEncoders/subdir.mk
-include libraries/Pololu/PololuQTRSensors/subdir.mk
-include libraries/Pololu/Pololu3pi/subdir.mk
-include libraries/Pololu/OrangutanResources/subdir.mk
-include libraries/Pololu/OrangutanPushbuttons/subdir.mk
-include libraries/Pololu/OrangutanMotors/subdir.mk
-include libraries/Pololu/OrangutanLEDs/subdir.mk
-include libraries/Pololu/OrangutanLCD/subdir.mk
-include libraries/Pololu/OrangutanDigital/subdir.mk
-include libraries/Pololu/OrangutanBuzzer/subdir.mk
-include libraries/Pololu/OrangutanAnalog/subdir.mk
-include libraries/OneWireMaster/subdir.mk
-include libraries/OneWire2/examples/subdir.mk
-include libraries/OneWire2/subdir.mk
-include libraries/OneWire/subdir.mk
-include libraries/NewSoftSerial/subdir.mk
-include libraries/MindSet/mdt/tgcd/win32/subdir.mk
-include libraries/MindSet/mdt/tg/subdir.mk
-include libraries/MindSet/subdir.mk
-include libraries/MCP23009/subdir.mk
-include libraries/KeyMatrix/subdir.mk
-include libraries/HMC5843/subdir.mk
-include libraries/HC595/subdir.mk
-include libraries/GPS/subdir.mk
-include libraries/FeliCaPlug/subdir.mk
-include libraries/Fat16/subdir.mk
-include libraries/Ethernet_W5100/utility/subdir.mk
-include libraries/Ethernet_W5100/subdir.mk
-include libraries/Digital6DOF/subdir.mk
-include libraries/DataFlash_SPI/subdir.mk
-include libraries/DES_OUTE/subdir.mk
-include libraries/ColorLCDShield/subdir.mk
-include libraries/ChibiOS/utility/subdir.mk
-include libraries/ChibiOS/SdFat/subdir.mk
-include libraries/ChibiOS/BufferedWriter/subdir.mk
-include libraries/ChibiOS/subdir.mk
-include libraries/ChLCD_SoftwareSerial/subdir.mk
-include libraries/ChLCD_ST7032i/subdir.mk
-include libraries/ChLCD_MCP2300x/subdir.mk
-include libraries/ChLCD_HC595/subdir.mk
-include libraries/CapacitiveSense/subdir.mk
-include libraries/CameraC328R/subdir.mk
-include libraries/BMP085/ATmega328/subdir.mk
-include libraries/BMP085/subdir.mk
-include libraries/Adafruit_PCD8544_Nokia_LCD/subdir.mk
-include libraries/Adafruit_LED_Backpack/subdir.mk
-include libraries/Adafruit_GFX/subdir.mk
-include libraries/AP_GPS/subdir.mk
-include libraries/APM_Compass/subdir.mk
-include libraries/AFMotor/subdir.mk
-include libraries/AES128/subdir.mk
-include libraries/ADC_MCP320X/subdir.mk
-include hardware/Quaranta/optiboot/subdir.mk
-include hardware/Quaranta/cores/arduino/subdir.mk
-include hardware/Quaranta/bootloaders/atmega644p/subdir.mk
-include hardware/Quaranta/bootloaders/atmega/subdir.mk
-include hardware/ProMicro/cores/arduino/subdir.mk
-include hardware/ProMicro/bootloaders/caterina/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(ASM_DEPS)),)
-include $(ASM_DEPS)
endif
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(S_DEPS)),)
-include $(S_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
ifneq ($(strip $(S_UPPER_DEPS)),)
-include $(S_UPPER_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 
LSS += \
ArduinoLibraries.lss \

SIZEDUMMY += \
sizedummy \


# All Target
all: ArduinoLibraries.elf secondary-outputs

# Tool invocations
ArduinoLibraries.elf: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: AVR C++ Linker'
	avr-g++ -Wl,-Map,ArduinoLibraries.map,--cref -mmcu=atmega16 -o "ArduinoLibraries.elf" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

ArduinoLibraries.lss: ArduinoLibraries.elf
	@echo 'Invoking: AVR Create Extended Listing'
	-avr-objdump -h -S ArduinoLibraries.elf  >"ArduinoLibraries.lss"
	@echo 'Finished building: $@'
	@echo ' '

sizedummy: ArduinoLibraries.elf
	@echo 'Invoking: Print Size'
	-avr-size --format=berkeley -t ArduinoLibraries.elf
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)$(C_DEPS)$(ELFS)$(LSS)$(S_DEPS)$(CXX_DEPS)$(S_UPPER_DEPS)$(C++_DEPS)$(ASM_DEPS)$(CC_DEPS)$(CPP_DEPS)$(C_UPPER_DEPS)$(SIZEDUMMY) ArduinoLibraries.elf
	-@echo ' '

secondary-outputs: $(LSS) $(SIZEDUMMY)

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
