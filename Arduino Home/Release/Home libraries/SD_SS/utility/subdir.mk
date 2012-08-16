################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/SD_SS/utility/Sd2Card_SS.cpp \
/Users/sin/Documents/Arduino/libraries/SD_SS/utility/SdFile.cpp \
/Users/sin/Documents/Arduino/libraries/SD_SS/utility/SdVolume.cpp 

OBJS += \
./Home\ libraries/SD_SS/utility/Sd2Card_SS.o \
./Home\ libraries/SD_SS/utility/SdFile.o \
./Home\ libraries/SD_SS/utility/SdVolume.o 

CPP_DEPS += \
./Home\ libraries/SD_SS/utility/Sd2Card_SS.d \
./Home\ libraries/SD_SS/utility/SdFile.d \
./Home\ libraries/SD_SS/utility/SdVolume.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/SD_SS/utility/Sd2Card_SS.o: /Users/sin/Documents/Arduino/libraries/SD_SS/utility/Sd2Card_SS.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Home libraries/SD_SS/utility/Sd2Card_SS.d" -MT"Home\ libraries/SD_SS/utility/Sd2Card_SS.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/SD_SS/utility/SdFile.o: /Users/sin/Documents/Arduino/libraries/SD_SS/utility/SdFile.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Home libraries/SD_SS/utility/SdFile.d" -MT"Home\ libraries/SD_SS/utility/SdFile.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/SD_SS/utility/SdVolume.o: /Users/sin/Documents/Arduino/libraries/SD_SS/utility/SdVolume.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Home libraries/SD_SS/utility/SdVolume.d" -MT"Home\ libraries/SD_SS/utility/SdVolume.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


