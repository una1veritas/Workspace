################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/RFID_PaSoRi/Max3421e.cpp \
/Users/sin/Documents/Arduino/libraries/RFID_PaSoRi/PaSoRi.cpp \
/Users/sin/Documents/Arduino/libraries/RFID_PaSoRi/Usb.cpp 

OBJS += \
./Home\ libraries/RFID_PaSoRi/Max3421e.o \
./Home\ libraries/RFID_PaSoRi/PaSoRi.o \
./Home\ libraries/RFID_PaSoRi/Usb.o 

CPP_DEPS += \
./Home\ libraries/RFID_PaSoRi/Max3421e.d \
./Home\ libraries/RFID_PaSoRi/PaSoRi.d \
./Home\ libraries/RFID_PaSoRi/Usb.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/RFID_PaSoRi/Max3421e.o: /Users/sin/Documents/Arduino/libraries/RFID_PaSoRi/Max3421e.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/RFID_PaSoRi/Max3421e.d" -MT"Home\ libraries/RFID_PaSoRi/Max3421e.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/RFID_PaSoRi/PaSoRi.o: /Users/sin/Documents/Arduino/libraries/RFID_PaSoRi/PaSoRi.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/RFID_PaSoRi/PaSoRi.d" -MT"Home\ libraries/RFID_PaSoRi/PaSoRi.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/RFID_PaSoRi/Usb.o: /Users/sin/Documents/Arduino/libraries/RFID_PaSoRi/Usb.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/RFID_PaSoRi/Usb.d" -MT"Home\ libraries/RFID_PaSoRi/Usb.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


