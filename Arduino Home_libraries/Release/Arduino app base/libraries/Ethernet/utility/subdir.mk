################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/utility/socket.cpp \
/Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/utility/w5100.cpp 

OBJS += \
./Arduino\ app\ base/libraries/Ethernet/utility/socket.o \
./Arduino\ app\ base/libraries/Ethernet/utility/w5100.o 

CPP_DEPS += \
./Arduino\ app\ base/libraries/Ethernet/utility/socket.d \
./Arduino\ app\ base/libraries/Ethernet/utility/w5100.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ app\ base/libraries/Ethernet/utility/socket.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/utility/socket.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/libraries/Ethernet/utility/socket.d" -MT"Arduino\ app\ base/libraries/Ethernet/utility/socket.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ app\ base/libraries/Ethernet/utility/w5100.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/utility/w5100.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/libraries/Ethernet/utility/w5100.d" -MT"Arduino\ app\ base/libraries/Ethernet/utility/w5100.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


