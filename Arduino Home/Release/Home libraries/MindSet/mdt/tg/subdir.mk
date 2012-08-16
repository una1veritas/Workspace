################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Users/sin/Documents/Arduino/libraries/MindSet/mdt/tg/ThinkGearStreamParser.c 

OBJS += \
./Home\ libraries/MindSet/mdt/tg/ThinkGearStreamParser.o 

C_DEPS += \
./Home\ libraries/MindSet/mdt/tg/ThinkGearStreamParser.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/MindSet/mdt/tg/ThinkGearStreamParser.o: /Users/sin/Documents/Arduino/libraries/MindSet/mdt/tg/ThinkGearStreamParser.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/MindSet/mdt/tg/ThinkGearStreamParser.d" -MT"Home\ libraries/MindSet/mdt/tg/ThinkGearStreamParser.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


