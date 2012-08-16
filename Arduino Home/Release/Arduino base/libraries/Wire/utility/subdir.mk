################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility/twi.c 

OBJS += \
./Arduino\ base/libraries/Wire/utility/twi.o 

C_DEPS += \
./Arduino\ base/libraries/Wire/utility/twi.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ base/libraries/Wire/utility/twi.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility/twi.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Arduino base/libraries/Wire/utility/twi.d" -MT"Arduino\ base/libraries/Wire/utility/twi.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


