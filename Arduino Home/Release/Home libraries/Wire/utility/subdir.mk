################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Users/sin/Documents/Arduino/libraries/Wire/utility/twi.c 

OBJS += \
./Home\ libraries/Wire/utility/twi.o 

C_DEPS += \
./Home\ libraries/Wire/utility/twi.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/Wire/utility/twi.o: /Users/sin/Documents/Arduino/libraries/Wire/utility/twi.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/Wire/utility/twi.d" -MT"Home\ libraries/Wire/utility/twi.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


