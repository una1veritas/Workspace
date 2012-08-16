################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/libraries/SD/File.cpp \
/Applications/Arduino.app/Contents/Resources/Java/libraries/SD/SD.cpp 

OBJS += \
./Arduino\ base/libraries/SD/File.o \
./Arduino\ base/libraries/SD/SD.o 

CPP_DEPS += \
./Arduino\ base/libraries/SD/File.d \
./Arduino\ base/libraries/SD/SD.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ base/libraries/SD/File.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/SD/File.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Arduino base/libraries/SD/File.d" -MT"Arduino\ base/libraries/SD/File.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/libraries/SD/SD.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/SD/SD.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Arduino base/libraries/SD/SD.d" -MT"Arduino\ base/libraries/SD/SD.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


