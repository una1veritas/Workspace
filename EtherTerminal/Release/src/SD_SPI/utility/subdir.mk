################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/SD_SPI/utility/Sd2Card.cpp \
../src/SD_SPI/utility/SdFile.cpp \
../src/SD_SPI/utility/SdVolume.cpp 

OBJS += \
./src/SD_SPI/utility/Sd2Card.o \
./src/SD_SPI/utility/SdFile.o \
./src/SD_SPI/utility/SdVolume.o 

CPP_DEPS += \
./src/SD_SPI/utility/Sd2Card.d \
./src/SD_SPI/utility/SdFile.d \
./src/SD_SPI/utility/SdVolume.d 


# Each subdirectory must supply rules for building sources it contributes
src/SD_SPI/utility/%.o: ../src/SD_SPI/utility/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Users/sin/Documents/Workspace/EtherTerminal/arduino/cores/arduino" -I"/Users/sin/Documents/Workspace/EtherTerminal/arduino/variants/quaranta" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


