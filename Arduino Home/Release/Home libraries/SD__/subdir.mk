################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/SD__/File.cpp \
/Users/sin/Documents/Arduino/libraries/SD__/SD.cpp 

OBJS += \
./Home\ libraries/SD__/File.o \
./Home\ libraries/SD__/SD.o 

CPP_DEPS += \
./Home\ libraries/SD__/File.d \
./Home\ libraries/SD__/SD.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/SD__/File.o: /Users/sin/Documents/Arduino/libraries/SD__/File.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/SD__/File.d" -MT"Home\ libraries/SD__/File.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/SD__/SD.o: /Users/sin/Documents/Arduino/libraries/SD__/SD.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/SD__/SD.d" -MT"Home\ libraries/SD__/SD.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


