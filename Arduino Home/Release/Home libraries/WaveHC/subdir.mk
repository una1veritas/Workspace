################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/WaveHC/FatReader.cpp \
/Users/sin/Documents/Arduino/libraries/WaveHC/SdReader.cpp \
/Users/sin/Documents/Arduino/libraries/WaveHC/WaveHC.cpp \
/Users/sin/Documents/Arduino/libraries/WaveHC/WaveUtil.cpp 

OBJS += \
./Home\ libraries/WaveHC/FatReader.o \
./Home\ libraries/WaveHC/SdReader.o \
./Home\ libraries/WaveHC/WaveHC.o \
./Home\ libraries/WaveHC/WaveUtil.o 

CPP_DEPS += \
./Home\ libraries/WaveHC/FatReader.d \
./Home\ libraries/WaveHC/SdReader.d \
./Home\ libraries/WaveHC/WaveHC.d \
./Home\ libraries/WaveHC/WaveUtil.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/WaveHC/FatReader.o: /Users/sin/Documents/Arduino/libraries/WaveHC/FatReader.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/WaveHC/FatReader.d" -MT"Home\ libraries/WaveHC/FatReader.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/WaveHC/SdReader.o: /Users/sin/Documents/Arduino/libraries/WaveHC/SdReader.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/WaveHC/SdReader.d" -MT"Home\ libraries/WaveHC/SdReader.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/WaveHC/WaveHC.o: /Users/sin/Documents/Arduino/libraries/WaveHC/WaveHC.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/WaveHC/WaveHC.d" -MT"Home\ libraries/WaveHC/WaveHC.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/WaveHC/WaveUtil.o: /Users/sin/Documents/Arduino/libraries/WaveHC/WaveUtil.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/WaveHC/WaveUtil.d" -MT"Home\ libraries/WaveHC/WaveUtil.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


