################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../cli/TinyBasicPlus.cpp \
../cli/main.cpp 

OBJS += \
./cli/TinyBasicPlus.o \
./cli/main.o 

CPP_DEPS += \
./cli/TinyBasicPlus.d \
./cli/main.d 


# Each subdirectory must supply rules for building sources it contributes
cli/%.o: ../cli/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Users/sin/Documents/Eclipse/Workspace/TinyBasicPlus/cores/arduino" -I"/Users/sin/Documents/Eclipse/Workspace/TinyBasicPlus/libraries/SD" -I"/Users/sin/Documents/Eclipse/Workspace/TinyBasicPlus/variants/standard" -DARDUINO=101 -DENABLE_FILEIO=1 -Wall -g2 -gstabs -O0 -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

