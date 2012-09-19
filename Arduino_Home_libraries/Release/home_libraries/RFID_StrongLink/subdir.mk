################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/RFID_StrongLink/StrongLinkI2C.cpp 

OBJS += \
./home_libraries/RFID_StrongLink/StrongLinkI2C.o 

CPP_DEPS += \
./home_libraries/RFID_StrongLink/StrongLinkI2C.d 


# Each subdirectory must supply rules for building sources it contributes
home_libraries/RFID_StrongLink/StrongLinkI2C.o: /Users/sin/Documents/Arduino/libraries/RFID_StrongLink/StrongLinkI2C.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


