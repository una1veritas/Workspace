################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Users/sin/Documents/Arduino/SparkLCD/LCD_driver.c 

OBJS += \
./Home\ Arduino/SparkLCD/LCD_driver.o 

C_DEPS += \
./Home\ Arduino/SparkLCD/LCD_driver.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ Arduino/SparkLCD/LCD_driver.o: /Users/sin/Documents/Arduino/SparkLCD/LCD_driver.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/SparkLCD/LCD_driver.d" -MT"Home\ Arduino/SparkLCD/LCD_driver.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


