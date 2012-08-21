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
./libraries/UsbKeyboard/oddebug.o \
./libraries/UsbKeyboard/usbdrv.o \
./libraries/UsbKeyboard/usbdrvasm.o 

C_DEPS += \
./libraries/UsbKeyboard/oddebug.d \
./libraries/UsbKeyboard/usbdrv.d 

S_UPPER_DEPS += \
./libraries/UsbKeyboard/usbdrvasm.d 

ASM_DEPS += \
./libraries/UsbKeyboard/usbdrvasm.d 


# Each subdirectory must supply rules for building sources it contributes
libraries/UsbKeyboard/oddebug.o: /Users/sin/Documents/Arduino/libraries/UsbKeyboard/oddebug.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Users/sin/Documents/Arduino/libraries/SD__/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

libraries/UsbKeyboard/usbdrv.o: /Users/sin/Documents/Arduino/libraries/UsbKeyboard/usbdrv.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Users/sin/Documents/Arduino/libraries/SD__/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

libraries/UsbKeyboard/usbdrvasm.o: /Users/sin/Documents/Arduino/libraries/UsbKeyboard/usbdrvasm.S
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Assembler'
	avr-gcc -x assembler-with-cpp -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Users/sin/Documents/Arduino/libraries/SD__/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -mmcu=atmega328p -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

libraries/UsbKeyboard/usbdrvasm.o: /Users/sin/Documents/Arduino/libraries/UsbKeyboard/usbdrvasm.asm
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Assembler'
	avr-gcc -x assembler-with-cpp -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Users/sin/Documents/Arduino/libraries/SD__/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -mmcu=atmega328p -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


