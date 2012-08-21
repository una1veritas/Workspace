################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/firmwares/arduino-usbdfu/Arduino-usbdfu.c \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/firmwares/arduino-usbdfu/Descriptors.c 

OBJS += \
./Arduino\ app\ base/hardware/arduino/firmwares/arduino-usbdfu/Arduino-usbdfu.o \
./Arduino\ app\ base/hardware/arduino/firmwares/arduino-usbdfu/Descriptors.o 

C_DEPS += \
./Arduino\ app\ base/hardware/arduino/firmwares/arduino-usbdfu/Arduino-usbdfu.d \
./Arduino\ app\ base/hardware/arduino/firmwares/arduino-usbdfu/Descriptors.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ app\ base/hardware/arduino/firmwares/arduino-usbdfu/Arduino-usbdfu.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/firmwares/arduino-usbdfu/Arduino-usbdfu.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Users/sin/Documents/Arduino/libraries/SD__/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/hardware/arduino/firmwares/arduino-usbdfu/Arduino-usbdfu.d" -MT"Arduino\ app\ base/hardware/arduino/firmwares/arduino-usbdfu/Arduino-usbdfu.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ app\ base/hardware/arduino/firmwares/arduino-usbdfu/Descriptors.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/firmwares/arduino-usbdfu/Descriptors.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Users/sin/Documents/Arduino/libraries/SD__/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/hardware/arduino/firmwares/arduino-usbdfu/Descriptors.d" -MT"Arduino\ app\ base/hardware/arduino/firmwares/arduino-usbdfu/Descriptors.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


