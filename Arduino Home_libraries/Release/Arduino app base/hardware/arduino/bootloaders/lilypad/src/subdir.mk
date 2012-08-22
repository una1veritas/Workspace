################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/bootloaders/lilypad/src/ATmegaBOOT.c 

OBJS += \
./Arduino\ app\ base/hardware/arduino/bootloaders/lilypad/src/ATmegaBOOT.o 

C_DEPS += \
./Arduino\ app\ base/hardware/arduino/bootloaders/lilypad/src/ATmegaBOOT.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ app\ base/hardware/arduino/bootloaders/lilypad/src/ATmegaBOOT.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/bootloaders/lilypad/src/ATmegaBOOT.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/hardware/arduino/bootloaders/lilypad/src/ATmegaBOOT.d" -MT"Arduino\ app\ base/hardware/arduino/bootloaders/lilypad/src/ATmegaBOOT.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


