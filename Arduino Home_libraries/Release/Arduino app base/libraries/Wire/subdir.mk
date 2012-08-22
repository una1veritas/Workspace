################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/Wire.cpp 

OBJS += \
./Arduino\ app\ base/libraries/Wire/Wire.o 

CPP_DEPS += \
./Arduino\ app\ base/libraries/Wire/Wire.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ app\ base/libraries/Wire/Wire.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/Wire.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/libraries/Wire/Wire.d" -MT"Arduino\ app\ base/libraries/Wire/Wire.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


