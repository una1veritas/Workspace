################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/Fat16/Fat16.cpp \
/Users/sin/Documents/Arduino/libraries/Fat16/SdCard.cpp 

OBJS += \
./Home\ libraries/Fat16/Fat16.o \
./Home\ libraries/Fat16/SdCard.o 

CPP_DEPS += \
./Home\ libraries/Fat16/Fat16.d \
./Home\ libraries/Fat16/SdCard.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/Fat16/Fat16.o: /Users/sin/Documents/Arduino/libraries/Fat16/Fat16.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/Fat16/Fat16.d" -MT"Home\ libraries/Fat16/Fat16.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/Fat16/SdCard.o: /Users/sin/Documents/Arduino/libraries/Fat16/SdCard.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/Fat16/SdCard.d" -MT"Home\ libraries/Fat16/SdCard.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


