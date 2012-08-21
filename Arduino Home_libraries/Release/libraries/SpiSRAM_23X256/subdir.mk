################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/SpiSRAM_23X256/SPISRAM.cpp 

OBJS += \
./libraries/SpiSRAM_23X256/SPISRAM.o 

CPP_DEPS += \
./libraries/SpiSRAM_23X256/SPISRAM.d 


# Each subdirectory must supply rules for building sources it contributes
libraries/SpiSRAM_23X256/SPISRAM.o: /Users/sin/Documents/Arduino/libraries/SpiSRAM_23X256/SPISRAM.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Users/sin/Documents/Arduino/libraries/SD__/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


