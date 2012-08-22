################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/bootloaders/caterina/Caterina.c \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/bootloaders/caterina/Descriptors.c 

OBJS += \
./Arduino\ app\ base/hardware/arduino/bootloaders/caterina/Caterina.o \
./Arduino\ app\ base/hardware/arduino/bootloaders/caterina/Descriptors.o 

C_DEPS += \
./Arduino\ app\ base/hardware/arduino/bootloaders/caterina/Caterina.d \
./Arduino\ app\ base/hardware/arduino/bootloaders/caterina/Descriptors.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ app\ base/hardware/arduino/bootloaders/caterina/Caterina.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/bootloaders/caterina/Caterina.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/hardware/arduino/bootloaders/caterina/Caterina.d" -MT"Arduino\ app\ base/hardware/arduino/bootloaders/caterina/Caterina.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ app\ base/hardware/arduino/bootloaders/caterina/Descriptors.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/bootloaders/caterina/Descriptors.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/hardware/arduino/bootloaders/caterina/Descriptors.d" -MT"Arduino\ app\ base/hardware/arduino/bootloaders/caterina/Descriptors.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


