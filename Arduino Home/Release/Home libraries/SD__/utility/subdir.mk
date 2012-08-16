################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/SD__/utility/Sd2Card.cpp \
/Users/sin/Documents/Arduino/libraries/SD__/utility/SdFile.cpp \
/Users/sin/Documents/Arduino/libraries/SD__/utility/SdVolume.cpp 

OBJS += \
./Home\ libraries/SD__/utility/Sd2Card.o \
./Home\ libraries/SD__/utility/SdFile.o \
./Home\ libraries/SD__/utility/SdVolume.o 

CPP_DEPS += \
./Home\ libraries/SD__/utility/Sd2Card.d \
./Home\ libraries/SD__/utility/SdFile.d \
./Home\ libraries/SD__/utility/SdVolume.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/SD__/utility/Sd2Card.o: /Users/sin/Documents/Arduino/libraries/SD__/utility/Sd2Card.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/SD__/utility/Sd2Card.d" -MT"Home\ libraries/SD__/utility/Sd2Card.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/SD__/utility/SdFile.o: /Users/sin/Documents/Arduino/libraries/SD__/utility/SdFile.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/SD__/utility/SdFile.d" -MT"Home\ libraries/SD__/utility/SdFile.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/SD__/utility/SdVolume.o: /Users/sin/Documents/Arduino/libraries/SD__/utility/SdVolume.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/SD__/utility/SdVolume.d" -MT"Home\ libraries/SD__/utility/SdVolume.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


