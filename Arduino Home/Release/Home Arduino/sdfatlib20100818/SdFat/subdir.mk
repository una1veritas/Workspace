################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/sdfatlib20100818/SdFat/Sd2Card.cpp \
/Users/sin/Documents/Arduino/sdfatlib20100818/SdFat/SdFile.cpp \
/Users/sin/Documents/Arduino/sdfatlib20100818/SdFat/SdVolume.cpp 

OBJS += \
./Home\ Arduino/sdfatlib20100818/SdFat/Sd2Card.o \
./Home\ Arduino/sdfatlib20100818/SdFat/SdFile.o \
./Home\ Arduino/sdfatlib20100818/SdFat/SdVolume.o 

CPP_DEPS += \
./Home\ Arduino/sdfatlib20100818/SdFat/Sd2Card.d \
./Home\ Arduino/sdfatlib20100818/SdFat/SdFile.d \
./Home\ Arduino/sdfatlib20100818/SdFat/SdVolume.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ Arduino/sdfatlib20100818/SdFat/Sd2Card.o: /Users/sin/Documents/Arduino/sdfatlib20100818/SdFat/Sd2Card.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/sdfatlib20100818/SdFat/Sd2Card.d" -MT"Home\ Arduino/sdfatlib20100818/SdFat/Sd2Card.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/sdfatlib20100818/SdFat/SdFile.o: /Users/sin/Documents/Arduino/sdfatlib20100818/SdFat/SdFile.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/sdfatlib20100818/SdFat/SdFile.d" -MT"Home\ Arduino/sdfatlib20100818/SdFat/SdFile.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/sdfatlib20100818/SdFat/SdVolume.o: /Users/sin/Documents/Arduino/sdfatlib20100818/SdFat/SdVolume.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/sdfatlib20100818/SdFat/SdVolume.d" -MT"Home\ Arduino/sdfatlib20100818/SdFat/SdVolume.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


