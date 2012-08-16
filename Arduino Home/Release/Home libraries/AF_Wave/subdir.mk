################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/AF_Wave/AF_Wave.cpp \
/Users/sin/Documents/Arduino/libraries/AF_Wave/fat16.cpp \
/Users/sin/Documents/Arduino/libraries/AF_Wave/partition.cpp \
/Users/sin/Documents/Arduino/libraries/AF_Wave/sd_raw.cpp \
/Users/sin/Documents/Arduino/libraries/AF_Wave/util.cpp \
/Users/sin/Documents/Arduino/libraries/AF_Wave/wave.cpp 

OBJS += \
./Home\ libraries/AF_Wave/AF_Wave.o \
./Home\ libraries/AF_Wave/fat16.o \
./Home\ libraries/AF_Wave/partition.o \
./Home\ libraries/AF_Wave/sd_raw.o \
./Home\ libraries/AF_Wave/util.o \
./Home\ libraries/AF_Wave/wave.o 

CPP_DEPS += \
./Home\ libraries/AF_Wave/AF_Wave.d \
./Home\ libraries/AF_Wave/fat16.d \
./Home\ libraries/AF_Wave/partition.d \
./Home\ libraries/AF_Wave/sd_raw.d \
./Home\ libraries/AF_Wave/util.d \
./Home\ libraries/AF_Wave/wave.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/AF_Wave/AF_Wave.o: /Users/sin/Documents/Arduino/libraries/AF_Wave/AF_Wave.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/AF_Wave/AF_Wave.d" -MT"Home\ libraries/AF_Wave/AF_Wave.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/AF_Wave/fat16.o: /Users/sin/Documents/Arduino/libraries/AF_Wave/fat16.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/AF_Wave/fat16.d" -MT"Home\ libraries/AF_Wave/fat16.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/AF_Wave/partition.o: /Users/sin/Documents/Arduino/libraries/AF_Wave/partition.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/AF_Wave/partition.d" -MT"Home\ libraries/AF_Wave/partition.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/AF_Wave/sd_raw.o: /Users/sin/Documents/Arduino/libraries/AF_Wave/sd_raw.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/AF_Wave/sd_raw.d" -MT"Home\ libraries/AF_Wave/sd_raw.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/AF_Wave/util.o: /Users/sin/Documents/Arduino/libraries/AF_Wave/util.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/AF_Wave/util.d" -MT"Home\ libraries/AF_Wave/util.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/AF_Wave/wave.o: /Users/sin/Documents/Arduino/libraries/AF_Wave/wave.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/AF_Wave/wave.d" -MT"Home\ libraries/AF_Wave/wave.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


