################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/a1ronzo-6DOF-Digital-3585b54/ADXL345.cpp \
/Users/sin/Documents/Arduino/a1ronzo-6DOF-Digital-3585b54/ITG3200.cpp \
/Users/sin/Documents/Arduino/a1ronzo-6DOF-Digital-3585b54/main.cpp 

C_SRCS += \
/Users/sin/Documents/Arduino/a1ronzo-6DOF-Digital-3585b54/twi.c 

OBJS += \
./Home\ Arduino/a1ronzo-6DOF-Digital-3585b54/ADXL345.o \
./Home\ Arduino/a1ronzo-6DOF-Digital-3585b54/ITG3200.o \
./Home\ Arduino/a1ronzo-6DOF-Digital-3585b54/main.o \
./Home\ Arduino/a1ronzo-6DOF-Digital-3585b54/twi.o 

C_DEPS += \
./Home\ Arduino/a1ronzo-6DOF-Digital-3585b54/twi.d 

CPP_DEPS += \
./Home\ Arduino/a1ronzo-6DOF-Digital-3585b54/ADXL345.d \
./Home\ Arduino/a1ronzo-6DOF-Digital-3585b54/ITG3200.d \
./Home\ Arduino/a1ronzo-6DOF-Digital-3585b54/main.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ Arduino/a1ronzo-6DOF-Digital-3585b54/ADXL345.o: /Users/sin/Documents/Arduino/a1ronzo-6DOF-Digital-3585b54/ADXL345.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/a1ronzo-6DOF-Digital-3585b54/ADXL345.d" -MT"Home\ Arduino/a1ronzo-6DOF-Digital-3585b54/ADXL345.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/a1ronzo-6DOF-Digital-3585b54/ITG3200.o: /Users/sin/Documents/Arduino/a1ronzo-6DOF-Digital-3585b54/ITG3200.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/a1ronzo-6DOF-Digital-3585b54/ITG3200.d" -MT"Home\ Arduino/a1ronzo-6DOF-Digital-3585b54/ITG3200.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/a1ronzo-6DOF-Digital-3585b54/main.o: /Users/sin/Documents/Arduino/a1ronzo-6DOF-Digital-3585b54/main.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/a1ronzo-6DOF-Digital-3585b54/main.d" -MT"Home\ Arduino/a1ronzo-6DOF-Digital-3585b54/main.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/a1ronzo-6DOF-Digital-3585b54/twi.o: /Users/sin/Documents/Arduino/a1ronzo-6DOF-Digital-3585b54/twi.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/a1ronzo-6DOF-Digital-3585b54/twi.d" -MT"Home\ Arduino/a1ronzo-6DOF-Digital-3585b54/twi.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


