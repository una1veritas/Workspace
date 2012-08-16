################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Users/sin/Documents/Arduino/libraries/OneWire2/examples/sample.c 

OBJS += \
./Home\ libraries/OneWire2/examples/sample.o 

C_DEPS += \
./Home\ libraries/OneWire2/examples/sample.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/OneWire2/examples/sample.o: /Users/sin/Documents/Arduino/libraries/OneWire2/examples/sample.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/OneWire2/examples/sample.d" -MT"Home\ libraries/OneWire2/examples/sample.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


