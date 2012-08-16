################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/bootloaders/atmega/ATmegaBOOT_168.c 

OBJS += \
./Arduino\ base/hardware/arduino/bootloaders/atmega/ATmegaBOOT_168.o 

C_DEPS += \
./Arduino\ base/hardware/arduino/bootloaders/atmega/ATmegaBOOT_168.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ base/hardware/arduino/bootloaders/atmega/ATmegaBOOT_168.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/bootloaders/atmega/ATmegaBOOT_168.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/bootloaders/atmega/ATmegaBOOT_168.d" -MT"Arduino\ base/hardware/arduino/bootloaders/atmega/ATmegaBOOT_168.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


