################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../libraries/EEPROM/EEPROM.cpp 

OBJS += \
./libraries/EEPROM/EEPROM.o 

CPP_DEPS += \
./libraries/EEPROM/EEPROM.d 


# Each subdirectory must supply rules for building sources it contributes
libraries/EEPROM/%.o: ../libraries/EEPROM/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Users/sin/Documents/Eclipse/Workspace/libArduino/arduino/cores/arduino" -I"/Users/sin/Documents/Eclipse/Workspace/libArduino/arduino/variants/standard" -I"/Users/sin/Documents/Eclipse/Workspace/libArduino/libraries" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


