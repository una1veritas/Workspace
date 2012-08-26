################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/library/EEPROM/EEPROM.cpp 

OBJS += \
./src/library/EEPROM/EEPROM.o 

CPP_DEPS += \
./src/library/EEPROM/EEPROM.d 


# Each subdirectory must supply rules for building sources it contributes
src/library/EEPROM/%.o: ../src/library/EEPROM/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


