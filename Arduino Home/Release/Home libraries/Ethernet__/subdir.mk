################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/Ethernet__/Dhcp.cpp \
/Users/sin/Documents/Arduino/libraries/Ethernet__/Dns.cpp \
/Users/sin/Documents/Arduino/libraries/Ethernet__/Ethernet.cpp \
/Users/sin/Documents/Arduino/libraries/Ethernet__/EthernetClient.cpp \
/Users/sin/Documents/Arduino/libraries/Ethernet__/EthernetServer.cpp \
/Users/sin/Documents/Arduino/libraries/Ethernet__/EthernetUdp.cpp 

OBJS += \
./Home\ libraries/Ethernet__/Dhcp.o \
./Home\ libraries/Ethernet__/Dns.o \
./Home\ libraries/Ethernet__/Ethernet.o \
./Home\ libraries/Ethernet__/EthernetClient.o \
./Home\ libraries/Ethernet__/EthernetServer.o \
./Home\ libraries/Ethernet__/EthernetUdp.o 

CPP_DEPS += \
./Home\ libraries/Ethernet__/Dhcp.d \
./Home\ libraries/Ethernet__/Dns.d \
./Home\ libraries/Ethernet__/Ethernet.d \
./Home\ libraries/Ethernet__/EthernetClient.d \
./Home\ libraries/Ethernet__/EthernetServer.d \
./Home\ libraries/Ethernet__/EthernetUdp.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/Ethernet__/Dhcp.o: /Users/sin/Documents/Arduino/libraries/Ethernet__/Dhcp.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/Ethernet__/Dhcp.d" -MT"Home\ libraries/Ethernet__/Dhcp.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/Ethernet__/Dns.o: /Users/sin/Documents/Arduino/libraries/Ethernet__/Dns.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/Ethernet__/Dns.d" -MT"Home\ libraries/Ethernet__/Dns.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/Ethernet__/Ethernet.o: /Users/sin/Documents/Arduino/libraries/Ethernet__/Ethernet.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/Ethernet__/Ethernet.d" -MT"Home\ libraries/Ethernet__/Ethernet.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/Ethernet__/EthernetClient.o: /Users/sin/Documents/Arduino/libraries/Ethernet__/EthernetClient.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/Ethernet__/EthernetClient.d" -MT"Home\ libraries/Ethernet__/EthernetClient.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/Ethernet__/EthernetServer.o: /Users/sin/Documents/Arduino/libraries/Ethernet__/EthernetServer.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/Ethernet__/EthernetServer.d" -MT"Home\ libraries/Ethernet__/EthernetServer.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/Ethernet__/EthernetUdp.o: /Users/sin/Documents/Arduino/libraries/Ethernet__/EthernetUdp.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/Ethernet__/EthernetUdp.d" -MT"Home\ libraries/Ethernet__/EthernetUdp.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


