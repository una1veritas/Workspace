################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/libraries/SoftwareSerial/SoftwareSerial.cpp 

OBJS += \
./Arduino\ app\ base/libraries/SoftwareSerial/SoftwareSerial.o 

CPP_DEPS += \
./Arduino\ app\ base/libraries/SoftwareSerial/SoftwareSerial.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ app\ base/libraries/SoftwareSerial/SoftwareSerial.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/SoftwareSerial/SoftwareSerial.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Users/sin/Documents/Arduino/libraries/SD__/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/libraries/SoftwareSerial/SoftwareSerial.d" -MT"Arduino\ app\ base/libraries/SoftwareSerial/SoftwareSerial.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


