################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
/Users/sin/Documents/Arduino/hardware/ProMicro/bootloaders/caterina/Caterina.o \
/Users/sin/Documents/Arduino/hardware/ProMicro/bootloaders/caterina/Descriptors.o 

C_SRCS += \
/Users/sin/Documents/Arduino/hardware/ProMicro/bootloaders/caterina/Caterina.c \
/Users/sin/Documents/Arduino/hardware/ProMicro/bootloaders/caterina/Descriptors.c 

OBJS += \
./Home\ Arduino/hardware/ProMicro/bootloaders/caterina/Caterina.o \
./Home\ Arduino/hardware/ProMicro/bootloaders/caterina/Descriptors.o 

C_DEPS += \
./Home\ Arduino/hardware/ProMicro/bootloaders/caterina/Caterina.d \
./Home\ Arduino/hardware/ProMicro/bootloaders/caterina/Descriptors.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ Arduino/hardware/ProMicro/bootloaders/caterina/Caterina.o: /Users/sin/Documents/Arduino/hardware/ProMicro/bootloaders/caterina/Caterina.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/bootloaders/caterina/Caterina.d" -MT"Home\ Arduino/hardware/ProMicro/bootloaders/caterina/Caterina.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/bootloaders/caterina/Descriptors.o: /Users/sin/Documents/Arduino/hardware/ProMicro/bootloaders/caterina/Descriptors.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/bootloaders/caterina/Descriptors.d" -MT"Home\ Arduino/hardware/ProMicro/bootloaders/caterina/Descriptors.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


