################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/libraries/Firmata/Firmata.cpp 

OBJS += \
./Arduino\ base/libraries/Firmata/Firmata.o 

CPP_DEPS += \
./Arduino\ base/libraries/Firmata/Firmata.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ base/libraries/Firmata/Firmata.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Firmata/Firmata.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/libraries/Firmata/Firmata.d" -MT"Arduino\ base/libraries/Firmata/Firmata.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


