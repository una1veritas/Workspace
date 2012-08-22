################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/Dhcp.cpp \
/Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/Dns.cpp \
/Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/Ethernet.cpp \
/Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/EthernetClient.cpp \
/Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/EthernetServer.cpp \
/Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/EthernetUdp.cpp 

OBJS += \
./Arduino\ app\ base/libraries/Ethernet/Dhcp.o \
./Arduino\ app\ base/libraries/Ethernet/Dns.o \
./Arduino\ app\ base/libraries/Ethernet/Ethernet.o \
./Arduino\ app\ base/libraries/Ethernet/EthernetClient.o \
./Arduino\ app\ base/libraries/Ethernet/EthernetServer.o \
./Arduino\ app\ base/libraries/Ethernet/EthernetUdp.o 

CPP_DEPS += \
./Arduino\ app\ base/libraries/Ethernet/Dhcp.d \
./Arduino\ app\ base/libraries/Ethernet/Dns.d \
./Arduino\ app\ base/libraries/Ethernet/Ethernet.d \
./Arduino\ app\ base/libraries/Ethernet/EthernetClient.d \
./Arduino\ app\ base/libraries/Ethernet/EthernetServer.d \
./Arduino\ app\ base/libraries/Ethernet/EthernetUdp.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ app\ base/libraries/Ethernet/Dhcp.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/Dhcp.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/libraries/Ethernet/Dhcp.d" -MT"Arduino\ app\ base/libraries/Ethernet/Dhcp.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ app\ base/libraries/Ethernet/Dns.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/Dns.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/libraries/Ethernet/Dns.d" -MT"Arduino\ app\ base/libraries/Ethernet/Dns.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ app\ base/libraries/Ethernet/Ethernet.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/Ethernet.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/libraries/Ethernet/Ethernet.d" -MT"Arduino\ app\ base/libraries/Ethernet/Ethernet.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ app\ base/libraries/Ethernet/EthernetClient.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/EthernetClient.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/libraries/Ethernet/EthernetClient.d" -MT"Arduino\ app\ base/libraries/Ethernet/EthernetClient.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ app\ base/libraries/Ethernet/EthernetServer.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/EthernetServer.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/libraries/Ethernet/EthernetServer.d" -MT"Arduino\ app\ base/libraries/Ethernet/EthernetServer.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ app\ base/libraries/Ethernet/EthernetUdp.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/EthernetUdp.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino app base/libraries/Ethernet/EthernetUdp.d" -MT"Arduino\ app\ base/libraries/Ethernet/EthernetUdp.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


