################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/SD_SPI/File.cpp \
../src/SD_SPI/SD_SPI.cpp 

OBJS += \
./src/SD_SPI/File.o \
./src/SD_SPI/SD_SPI.o 

CPP_DEPS += \
./src/SD_SPI/File.d \
./src/SD_SPI/SD_SPI.d 


# Each subdirectory must supply rules for building sources it contributes
src/SD_SPI/%.o: ../src/SD_SPI/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Users/sin/Documents/Workspace/EtherTerminal/arduino/cores/arduino" -I"/Users/sin/Documents/Workspace/EtherTerminal/arduino/variants/quaranta" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


