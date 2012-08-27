################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../libraries/SPI/SPI.cpp 

OBJS += \
./libraries/SPI/SPI.o 

CPP_DEPS += \
./libraries/SPI/SPI.d 


# Each subdirectory must supply rules for building sources it contributes
libraries/SPI/%.o: ../libraries/SPI/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Users/sin/Documents/Workspace/Arduino_eclipse_build/cores/arduino" -I"/Users/sin/Documents/Workspace/Arduino_eclipse_build/variants/standard" -I"/Users/sin/Documents/Workspace/Arduino_eclipse_build/libraries" -I"/Users/sin/Documents/Arduino/libraries" -DARDUINO=101 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


