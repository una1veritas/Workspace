################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/AFMotor/AFMotor.cpp 

OBJS += \
./Home\ libraries/AFMotor/AFMotor.o 

CPP_DEPS += \
./Home\ libraries/AFMotor/AFMotor.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/AFMotor/AFMotor.o: /Users/sin/Documents/Arduino/libraries/AFMotor/AFMotor.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/AFMotor/AFMotor.d" -MT"Home\ libraries/AFMotor/AFMotor.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


