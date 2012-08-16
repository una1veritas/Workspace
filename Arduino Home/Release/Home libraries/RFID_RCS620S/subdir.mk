################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/RFID_RCS620S/RCS620S.cpp 

OBJS += \
./Home\ libraries/RFID_RCS620S/RCS620S.o 

CPP_DEPS += \
./Home\ libraries/RFID_RCS620S/RCS620S.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/RFID_RCS620S/RCS620S.o: /Users/sin/Documents/Arduino/libraries/RFID_RCS620S/RCS620S.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/RFID_RCS620S/RCS620S.d" -MT"Home\ libraries/RFID_RCS620S/RCS620S.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


