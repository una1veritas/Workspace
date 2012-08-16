################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/Pololu/OrangutanBuzzer/OrangutanBuzzer.cpp 

OBJS += \
./Home\ libraries/Pololu/OrangutanBuzzer/OrangutanBuzzer.o 

CPP_DEPS += \
./Home\ libraries/Pololu/OrangutanBuzzer/OrangutanBuzzer.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/Pololu/OrangutanBuzzer/OrangutanBuzzer.o: /Users/sin/Documents/Arduino/libraries/Pololu/OrangutanBuzzer/OrangutanBuzzer.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/Pololu/OrangutanBuzzer/OrangutanBuzzer.d" -MT"Home\ libraries/Pololu/OrangutanBuzzer/OrangutanBuzzer.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


