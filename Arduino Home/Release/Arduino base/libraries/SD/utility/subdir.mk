################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/libraries/SD/utility/Sd2Card.cpp \
/Applications/Arduino.app/Contents/Resources/Java/libraries/SD/utility/SdFile.cpp \
/Applications/Arduino.app/Contents/Resources/Java/libraries/SD/utility/SdVolume.cpp 

OBJS += \
./Arduino\ base/libraries/SD/utility/Sd2Card.o \
./Arduino\ base/libraries/SD/utility/SdFile.o \
./Arduino\ base/libraries/SD/utility/SdVolume.o 

CPP_DEPS += \
./Arduino\ base/libraries/SD/utility/Sd2Card.d \
./Arduino\ base/libraries/SD/utility/SdFile.d \
./Arduino\ base/libraries/SD/utility/SdVolume.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ base/libraries/SD/utility/Sd2Card.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/SD/utility/Sd2Card.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Arduino base/libraries/SD/utility/Sd2Card.d" -MT"Arduino\ base/libraries/SD/utility/Sd2Card.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/libraries/SD/utility/SdFile.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/SD/utility/SdFile.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Arduino base/libraries/SD/utility/SdFile.d" -MT"Arduino\ base/libraries/SD/utility/SdFile.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/libraries/SD/utility/SdVolume.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/SD/utility/SdVolume.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Arduino base/libraries/SD/utility/SdVolume.d" -MT"Arduino\ base/libraries/SD/utility/SdVolume.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


