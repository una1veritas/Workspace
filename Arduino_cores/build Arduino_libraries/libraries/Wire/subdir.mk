################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../libraries/Wire/Wire.cpp 

OBJS += \
./libraries/Wire/Wire.o 

CPP_DEPS += \
./libraries/Wire/Wire.d 


# Each subdirectory must supply rules for building sources it contributes
libraries/Wire/%.o: ../libraries/Wire/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Users/sin/Documents/Eclipse/Workspace/Arduino_cores/cores/arduino" -I"/Users/sin/Documents/Eclipse/Workspace/Arduino_cores/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


