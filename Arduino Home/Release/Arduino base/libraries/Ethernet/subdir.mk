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
./Arduino\ base/libraries/Ethernet/Dhcp.o \
./Arduino\ base/libraries/Ethernet/Dns.o \
./Arduino\ base/libraries/Ethernet/Ethernet.o \
./Arduino\ base/libraries/Ethernet/EthernetClient.o \
./Arduino\ base/libraries/Ethernet/EthernetServer.o \
./Arduino\ base/libraries/Ethernet/EthernetUdp.o 

CPP_DEPS += \
./Arduino\ base/libraries/Ethernet/Dhcp.d \
./Arduino\ base/libraries/Ethernet/Dns.d \
./Arduino\ base/libraries/Ethernet/Ethernet.d \
./Arduino\ base/libraries/Ethernet/EthernetClient.d \
./Arduino\ base/libraries/Ethernet/EthernetServer.d \
./Arduino\ base/libraries/Ethernet/EthernetUdp.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ base/libraries/Ethernet/Dhcp.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/Dhcp.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Arduino base/libraries/Ethernet/Dhcp.d" -MT"Arduino\ base/libraries/Ethernet/Dhcp.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/libraries/Ethernet/Dns.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/Dns.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Arduino base/libraries/Ethernet/Dns.d" -MT"Arduino\ base/libraries/Ethernet/Dns.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/libraries/Ethernet/Ethernet.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/Ethernet.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Arduino base/libraries/Ethernet/Ethernet.d" -MT"Arduino\ base/libraries/Ethernet/Ethernet.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/libraries/Ethernet/EthernetClient.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/EthernetClient.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Arduino base/libraries/Ethernet/EthernetClient.d" -MT"Arduino\ base/libraries/Ethernet/EthernetClient.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/libraries/Ethernet/EthernetServer.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/EthernetServer.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Arduino base/libraries/Ethernet/EthernetServer.d" -MT"Arduino\ base/libraries/Ethernet/EthernetServer.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/libraries/Ethernet/EthernetUdp.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/Ethernet/EthernetUdp.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Arduino base/libraries/Ethernet/EthernetUdp.d" -MT"Arduino\ base/libraries/Ethernet/EthernetUdp.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


