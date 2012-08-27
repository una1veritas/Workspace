################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/Ethernet_W5100/Dhcp.cpp \
/Users/sin/Documents/Arduino/libraries/Ethernet_W5100/Dns.cpp \
/Users/sin/Documents/Arduino/libraries/Ethernet_W5100/EthernetClient.cpp \
/Users/sin/Documents/Arduino/libraries/Ethernet_W5100/EthernetServer.cpp \
/Users/sin/Documents/Arduino/libraries/Ethernet_W5100/EthernetUdp.cpp \
/Users/sin/Documents/Arduino/libraries/Ethernet_W5100/Ethernet_w5100.cpp 

OBJS += \
./home_libraries/Ethernet_W5100/Dhcp.o \
./home_libraries/Ethernet_W5100/Dns.o \
./home_libraries/Ethernet_W5100/EthernetClient.o \
./home_libraries/Ethernet_W5100/EthernetServer.o \
./home_libraries/Ethernet_W5100/EthernetUdp.o \
./home_libraries/Ethernet_W5100/Ethernet_w5100.o 

CPP_DEPS += \
./home_libraries/Ethernet_W5100/Dhcp.d \
./home_libraries/Ethernet_W5100/Dns.d \
./home_libraries/Ethernet_W5100/EthernetClient.d \
./home_libraries/Ethernet_W5100/EthernetServer.d \
./home_libraries/Ethernet_W5100/EthernetUdp.d \
./home_libraries/Ethernet_W5100/Ethernet_w5100.d 


# Each subdirectory must supply rules for building sources it contributes
home_libraries/Ethernet_W5100/Dhcp.o: /Users/sin/Documents/Arduino/libraries/Ethernet_W5100/Dhcp.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/Ethernet_W5100/Dns.o: /Users/sin/Documents/Arduino/libraries/Ethernet_W5100/Dns.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/Ethernet_W5100/EthernetClient.o: /Users/sin/Documents/Arduino/libraries/Ethernet_W5100/EthernetClient.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/Ethernet_W5100/EthernetServer.o: /Users/sin/Documents/Arduino/libraries/Ethernet_W5100/EthernetServer.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/Ethernet_W5100/EthernetUdp.o: /Users/sin/Documents/Arduino/libraries/Ethernet_W5100/EthernetUdp.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/Ethernet_W5100/Ethernet_w5100.o: /Users/sin/Documents/Arduino/libraries/Ethernet_W5100/Ethernet_w5100.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


