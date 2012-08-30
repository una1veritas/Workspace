################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/PS3BT.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/PS3USB.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/RFCOMM.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/Usb.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/XBOXUSB.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/adk.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/cdcacm.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/cdcftdi.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/cdcprolific.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/hid.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/hidboot.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/hidescriptorparser.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/hiduniversal.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/hidusagetitlearrays.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/masstorage.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/max_LCD.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/message.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/parsetools.cpp \
/Users/sin/Documents/Arduino/libraries/USB_Host_Shield/usbhub.cpp 

OBJS += \
./home_libraries/USB_Host_Shield/PS3BT.o \
./home_libraries/USB_Host_Shield/PS3USB.o \
./home_libraries/USB_Host_Shield/RFCOMM.o \
./home_libraries/USB_Host_Shield/Usb.o \
./home_libraries/USB_Host_Shield/XBOXUSB.o \
./home_libraries/USB_Host_Shield/adk.o \
./home_libraries/USB_Host_Shield/cdcacm.o \
./home_libraries/USB_Host_Shield/cdcftdi.o \
./home_libraries/USB_Host_Shield/cdcprolific.o \
./home_libraries/USB_Host_Shield/hid.o \
./home_libraries/USB_Host_Shield/hidboot.o \
./home_libraries/USB_Host_Shield/hidescriptorparser.o \
./home_libraries/USB_Host_Shield/hiduniversal.o \
./home_libraries/USB_Host_Shield/hidusagetitlearrays.o \
./home_libraries/USB_Host_Shield/masstorage.o \
./home_libraries/USB_Host_Shield/max_LCD.o \
./home_libraries/USB_Host_Shield/message.o \
./home_libraries/USB_Host_Shield/parsetools.o \
./home_libraries/USB_Host_Shield/usbhub.o 

CPP_DEPS += \
./home_libraries/USB_Host_Shield/PS3BT.d \
./home_libraries/USB_Host_Shield/PS3USB.d \
./home_libraries/USB_Host_Shield/RFCOMM.d \
./home_libraries/USB_Host_Shield/Usb.d \
./home_libraries/USB_Host_Shield/XBOXUSB.d \
./home_libraries/USB_Host_Shield/adk.d \
./home_libraries/USB_Host_Shield/cdcacm.d \
./home_libraries/USB_Host_Shield/cdcftdi.d \
./home_libraries/USB_Host_Shield/cdcprolific.d \
./home_libraries/USB_Host_Shield/hid.d \
./home_libraries/USB_Host_Shield/hidboot.d \
./home_libraries/USB_Host_Shield/hidescriptorparser.d \
./home_libraries/USB_Host_Shield/hiduniversal.d \
./home_libraries/USB_Host_Shield/hidusagetitlearrays.d \
./home_libraries/USB_Host_Shield/masstorage.d \
./home_libraries/USB_Host_Shield/max_LCD.d \
./home_libraries/USB_Host_Shield/message.d \
./home_libraries/USB_Host_Shield/parsetools.d \
./home_libraries/USB_Host_Shield/usbhub.d 


# Each subdirectory must supply rules for building sources it contributes
home_libraries/USB_Host_Shield/PS3BT.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/PS3BT.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/PS3USB.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/PS3USB.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/RFCOMM.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/RFCOMM.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/Usb.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/Usb.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/XBOXUSB.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/XBOXUSB.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/adk.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/adk.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/cdcacm.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/cdcacm.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/cdcftdi.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/cdcftdi.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/cdcprolific.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/cdcprolific.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/hid.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/hid.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/hidboot.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/hidboot.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/hidescriptorparser.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/hidescriptorparser.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/hiduniversal.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/hiduniversal.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/hidusagetitlearrays.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/hidusagetitlearrays.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/masstorage.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/masstorage.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/max_LCD.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/max_LCD.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/message.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/message.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/parsetools.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/parsetools.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/USB_Host_Shield/usbhub.o: /Users/sin/Documents/Arduino/libraries/USB_Host_Shield/usbhub.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


