################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Users/sin/Documents/Arduino/libraries/MyWire/utility/twi.c 

OBJS += \
./Home\ libraries/MyWire/utility/twi.o 

C_DEPS += \
./Home\ libraries/MyWire/utility/twi.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/MyWire/utility/twi.o: /Users/sin/Documents/Arduino/libraries/MyWire/utility/twi.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Home libraries/MyWire/utility/twi.d" -MT"Home\ libraries/MyWire/utility/twi.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


