################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/CharacterLCD_HC595/HC595LCD.cpp 

OBJS += \
./home_libraries/CharacterLCD_HC595/HC595LCD.o 

CPP_DEPS += \
./home_libraries/CharacterLCD_HC595/HC595LCD.d 


# Each subdirectory must supply rules for building sources it contributes
home_libraries/CharacterLCD_HC595/HC595LCD.o: /Users/sin/Documents/Arduino/libraries/CharacterLCD_HC595/HC595LCD.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

