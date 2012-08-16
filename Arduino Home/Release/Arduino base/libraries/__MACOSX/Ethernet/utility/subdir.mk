################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/libraries/__MACOSX/Ethernet/utility/._socket.cpp \
/Applications/Arduino.app/Contents/Resources/Java/libraries/__MACOSX/Ethernet/utility/._w5100.cpp 

OBJS += \
./Arduino\ base/libraries/__MACOSX/Ethernet/utility/._socket.o \
./Arduino\ base/libraries/__MACOSX/Ethernet/utility/._w5100.o 

CPP_DEPS += \
./Arduino\ base/libraries/__MACOSX/Ethernet/utility/._socket.d \
./Arduino\ base/libraries/__MACOSX/Ethernet/utility/._w5100.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ base/libraries/__MACOSX/Ethernet/utility/._socket.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/__MACOSX/Ethernet/utility/._socket.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/libraries/__MACOSX/Ethernet/utility/._socket.d" -MT"Arduino\ base/libraries/__MACOSX/Ethernet/utility/._socket.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/libraries/__MACOSX/Ethernet/utility/._w5100.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/__MACOSX/Ethernet/utility/._w5100.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/libraries/__MACOSX/Ethernet/utility/._w5100.d" -MT"Arduino\ base/libraries/__MACOSX/Ethernet/utility/._w5100.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


