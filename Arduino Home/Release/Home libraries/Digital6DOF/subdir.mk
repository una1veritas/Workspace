################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/Digital6DOF/ADXL345.cpp \
/Users/sin/Documents/Arduino/libraries/Digital6DOF/ITG3200.cpp 

OBJS += \
./Home\ libraries/Digital6DOF/ADXL345.o \
./Home\ libraries/Digital6DOF/ITG3200.o 

CPP_DEPS += \
./Home\ libraries/Digital6DOF/ADXL345.d \
./Home\ libraries/Digital6DOF/ITG3200.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/Digital6DOF/ADXL345.o: /Users/sin/Documents/Arduino/libraries/Digital6DOF/ADXL345.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/Digital6DOF/ADXL345.d" -MT"Home\ libraries/Digital6DOF/ADXL345.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/Digital6DOF/ITG3200.o: /Users/sin/Documents/Arduino/libraries/Digital6DOF/ITG3200.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/Digital6DOF/ITG3200.d" -MT"Home\ libraries/Digital6DOF/ITG3200.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


