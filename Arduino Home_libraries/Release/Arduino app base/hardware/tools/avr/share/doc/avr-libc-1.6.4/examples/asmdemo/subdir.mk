################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/asmdemo/asmdemo.c 

S_UPPER_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/asmdemo/isrs.S 

OBJS += \
./Arduino\ app\ base/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/asmdemo/asmdemo.o \
./Arduino\ app\ base/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/asmdemo/isrs.o 

C_DEPS += \
./Arduino\ app\ base/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/asmdemo/asmdemo.d 

S_UPPER_DEPS += \
./Arduino\ app\ base/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/asmdemo/isrs.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ app\ base/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/asmdemo/asmdemo.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/asmdemo/asmdemo.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/asmdemo/asmdemo.d" -MT"Arduino\ app\ base/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/asmdemo/asmdemo.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ app\ base/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/asmdemo/isrs.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/asmdemo/isrs.S
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Assembler'
	avr-gcc -x assembler-with-cpp -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -mmcu=atmega328p -MMD -MP -MF"Arduino app base/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/asmdemo/isrs.d" -MT"Arduino\ app\ base/hardware/tools/avr/share/doc/avr-libc-1.6.4/examples/asmdemo/isrs.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


