################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/Ethernet_SS/utility/socket.cpp \
/Users/sin/Documents/Arduino/libraries/Ethernet_SS/utility/w5100.cpp 

OBJS += \
./Home\ libraries/Ethernet_SS/utility/socket.o \
./Home\ libraries/Ethernet_SS/utility/w5100.o 

CPP_DEPS += \
./Home\ libraries/Ethernet_SS/utility/socket.d \
./Home\ libraries/Ethernet_SS/utility/w5100.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/Ethernet_SS/utility/socket.o: /Users/sin/Documents/Arduino/libraries/Ethernet_SS/utility/socket.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Home libraries/Ethernet_SS/utility/socket.d" -MT"Home\ libraries/Ethernet_SS/utility/socket.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/Ethernet_SS/utility/w5100.o: /Users/sin/Documents/Arduino/libraries/Ethernet_SS/utility/w5100.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Home libraries/Ethernet_SS/utility/w5100.d" -MT"Home\ libraries/Ethernet_SS/utility/w5100.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


