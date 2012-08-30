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
./home_libraries/UsbKeyboard/oddebug.o \
./home_libraries/UsbKeyboard/usbdrv.o \
./home_libraries/UsbKeyboard/usbdrvasm.o 

C_DEPS += \
./home_libraries/UsbKeyboard/oddebug.d \
./home_libraries/UsbKeyboard/usbdrv.d 

S_UPPER_DEPS += \
./home_libraries/UsbKeyboard/usbdrvasm.d 

ASM_DEPS += \
./home_libraries/UsbKeyboard/usbdrvasm.d 


# Each subdirectory must supply rules for building sources it contributes
home_libraries/UsbKeyboard/oddebug.o: /Users/sin/Documents/Arduino/libraries/UsbKeyboard/oddebug.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DNON_ARDUINO_IDE -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/UsbKeyboard/usbdrv.o: /Users/sin/Documents/Arduino/libraries/UsbKeyboard/usbdrv.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DNON_ARDUINO_IDE -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/UsbKeyboard/usbdrvasm.o: /Users/sin/Documents/Arduino/libraries/UsbKeyboard/usbdrvasm.S
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Assembler'
	avr-gcc -x assembler-with-cpp -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -mmcu=atmega328p -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/UsbKeyboard/usbdrvasm.o: /Users/sin/Documents/Arduino/libraries/UsbKeyboard/usbdrvasm.asm
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Assembler'
	avr-gcc -x assembler-with-cpp -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -mmcu=atmega328p -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


