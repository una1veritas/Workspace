################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../libraries/Ethernet/utility/socket.cpp \
../libraries/Ethernet/utility/w5100.cpp 

OBJS += \
./libraries/Ethernet/utility/socket.o \
./libraries/Ethernet/utility/w5100.o 

CPP_DEPS += \
./libraries/Ethernet/utility/socket.d \
./libraries/Ethernet/utility/w5100.d 


# Each subdirectory must supply rules for building sources it contributes
libraries/Ethernet/utility/%.o: ../libraries/Ethernet/utility/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Users/sin/Documents/Eclipse/Workspace/libArduino/arduino/cores/arduino" -I"/Users/sin/Documents/Eclipse/Workspace/libArduino/arduino/variants/standard" -I"/Users/sin/Documents/Eclipse/Workspace/libArduino/libraries" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


