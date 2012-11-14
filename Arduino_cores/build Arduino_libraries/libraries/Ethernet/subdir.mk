################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../libraries/Ethernet/Dhcp.cpp \
../libraries/Ethernet/Dns.cpp \
../libraries/Ethernet/Ethernet.cpp \
../libraries/Ethernet/EthernetClient.cpp \
../libraries/Ethernet/EthernetServer.cpp \
../libraries/Ethernet/EthernetUdp.cpp 

OBJS += \
./libraries/Ethernet/Dhcp.o \
./libraries/Ethernet/Dns.o \
./libraries/Ethernet/Ethernet.o \
./libraries/Ethernet/EthernetClient.o \
./libraries/Ethernet/EthernetServer.o \
./libraries/Ethernet/EthernetUdp.o 

CPP_DEPS += \
./libraries/Ethernet/Dhcp.d \
./libraries/Ethernet/Dns.d \
./libraries/Ethernet/Ethernet.d \
./libraries/Ethernet/EthernetClient.d \
./libraries/Ethernet/EthernetServer.d \
./libraries/Ethernet/EthernetUdp.d 


# Each subdirectory must supply rules for building sources it contributes
libraries/Ethernet/%.o: ../libraries/Ethernet/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Users/sin/Documents/Eclipse/Workspace/Arduino_cores/cores/arduino" -I"/Users/sin/Documents/Eclipse/Workspace/Arduino_cores/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


