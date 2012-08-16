################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/Monitor/Monitor.cpp 

OBJS += \
./Home\ libraries/Monitor/Monitor.o 

CPP_DEPS += \
./Home\ libraries/Monitor/Monitor.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/Monitor/Monitor.o: /Users/sin/Documents/Arduino/libraries/Monitor/Monitor.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Home libraries/Monitor/Monitor.d" -MT"Home\ libraries/Monitor/Monitor.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


