################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
/Users/sin/Documents/Arduino/libraries/UsbKeyboard/usbdrv.o \
/Users/sin/Documents/Arduino/libraries/UsbKeyboard/usbdrvasm.o 

C_SRCS += \
/Users/sin/Documents/Arduino/libraries/UsbKeyboard/oddebug.c \
/Users/sin/Documents/Arduino/libraries/UsbKeyboard/usbdrv.c 

S_UPPER_SRCS += \
/Users/sin/Documents/Arduino/libraries/UsbKeyboard/usbdrvasm.S 

ASM_SRCS += \
/Users/sin/Documents/Arduino/libraries/UsbKeyboard/usbdrvasm.asm 

OBJS += \
./Home\ libraries/UsbKeyboard/oddebug.o \
./Home\ libraries/UsbKeyboard/usbdrv.o \
./Home\ libraries/UsbKeyboard/usbdrvasm.o 

C_DEPS += \
./Home\ libraries/UsbKeyboard/oddebug.d \
./Home\ libraries/UsbKeyboard/usbdrv.d 

S_UPPER_DEPS += \
./Home\ libraries/UsbKeyboard/usbdrvasm.d 

ASM_DEPS += \
./Home\ libraries/UsbKeyboard/usbdrvasm.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/UsbKeyboard/oddebug.o: /Users/sin/Documents/Arduino/libraries/UsbKeyboard/oddebug.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/UsbKeyboard/oddebug.d" -MT"Home\ libraries/UsbKeyboard/oddebug.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/UsbKeyboard/usbdrv.o: /Users/sin/Documents/Arduino/libraries/UsbKeyboard/usbdrv.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/UsbKeyboard/usbdrv.d" -MT"Home\ libraries/UsbKeyboard/usbdrv.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/UsbKeyboard/usbdrvasm.o: /Users/sin/Documents/Arduino/libraries/UsbKeyboard/usbdrvasm.S
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Assembler'
	avr-gcc -x assembler-with-cpp -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -mmcu=atmega328p -MMD -MP -MF"Home libraries/UsbKeyboard/usbdrvasm.d" -MT"Home\ libraries/UsbKeyboard/usbdrvasm.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/UsbKeyboard/usbdrvasm.o: /Users/sin/Documents/Arduino/libraries/UsbKeyboard/usbdrvasm.asm
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Assembler'
	avr-gcc -x assembler-with-cpp -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -mmcu=atmega328p -MMD -MP -MF"Home libraries/UsbKeyboard/usbdrvasm.d" -MT"Home\ libraries/UsbKeyboard/usbdrvasm.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


