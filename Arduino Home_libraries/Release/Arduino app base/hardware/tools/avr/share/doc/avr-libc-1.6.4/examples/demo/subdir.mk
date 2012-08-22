################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/demo/demo.c 

OBJS += \
./Arduino\ app\ base/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/demo/demo.o 

C_DEPS += \
./Arduino\ app\ base/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/demo/demo.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ app\ base/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/demo/demo.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/demo/demo.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/demo/demo.d" -MT"Arduino\ app\ base/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/demo/demo.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


