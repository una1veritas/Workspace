################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Users/sin/Documents/Arduino/SparkLCD/fromOriginal/main.c 

OBJS += \
./Home\ Arduino/SparkLCD/fromOriginal/main.o 

C_DEPS += \
./Home\ Arduino/SparkLCD/fromOriginal/main.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ Arduino/SparkLCD/fromOriginal/main.o: /Users/sin/Documents/Arduino/SparkLCD/fromOriginal/main.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/SparkLCD/fromOriginal/main.d" -MT"Home\ Arduino/SparkLCD/fromOriginal/main.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


