################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/examples/HID/le3dp/le3dp_rptparser.cpp 

OBJS += \
./libraries/USB_Host_Shield/examples/HID/le3dp/le3dp_rptparser.o 

CPP_DEPS += \
./libraries/USB_Host_Shield/examples/HID/le3dp/le3dp_rptparser.d 


# Each subdirectory must supply rules for building sources it contributes
libraries/USB_Host_Shield/examples/HID/le3dp/le3dp_rptparser.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/examples/HID/le3dp/le3dp_rptparser.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


