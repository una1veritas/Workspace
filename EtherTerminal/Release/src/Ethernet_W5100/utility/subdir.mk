################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Ethernet_W5100/utility/socket.cpp \
../src/Ethernet_W5100/utility/w5100.cpp 

OBJS += \
./src/Ethernet_W5100/utility/socket.o \
./src/Ethernet_W5100/utility/w5100.o 

CPP_DEPS += \
./src/Ethernet_W5100/utility/socket.d \
./src/Ethernet_W5100/utility/w5100.d 


# Each subdirectory must supply rules for building sources it contributes
src/Ethernet_W5100/utility/%.o: ../src/Ethernet_W5100/utility/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Users/sin/Documents/Workspace/EtherTerminal/arduino/cores/arduino" -I"/Users/sin/Documents/Workspace/EtherTerminal/arduino/variants/quaranta" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


