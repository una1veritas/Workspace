################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/Roomba/Roomba.cpp 

OBJS += \
./home_libraries/Roomba/Roomba.o 

CPP_DEPS += \
./home_libraries/Roomba/Roomba.d 


# Each subdirectory must supply rules for building sources it contributes
home_libraries/Roomba/Roomba.o: /Users/sin/Documents/Arduino/libraries/Roomba/Roomba.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


