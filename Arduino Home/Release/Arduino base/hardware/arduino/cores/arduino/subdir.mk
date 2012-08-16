################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/CDC.cpp \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/HID.cpp \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/HardwareSerial.cpp \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/IPAddress.cpp \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/Print.cpp \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/Stream.cpp \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/Tone.cpp \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/USBCore.cpp \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/WMath.cpp \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/WString.cpp \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/main.cpp \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/new.cpp 

C_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/WInterrupts.c \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/wiring.c \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/wiring_analog.c \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/wiring_digital.c \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/wiring_pulse.c \
/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/wiring_shift.c 

OBJS += \
./Arduino\ base/hardware/arduino/cores/arduino/CDC.o \
./Arduino\ base/hardware/arduino/cores/arduino/HID.o \
./Arduino\ base/hardware/arduino/cores/arduino/HardwareSerial.o \
./Arduino\ base/hardware/arduino/cores/arduino/IPAddress.o \
./Arduino\ base/hardware/arduino/cores/arduino/Print.o \
./Arduino\ base/hardware/arduino/cores/arduino/Stream.o \
./Arduino\ base/hardware/arduino/cores/arduino/Tone.o \
./Arduino\ base/hardware/arduino/cores/arduino/USBCore.o \
./Arduino\ base/hardware/arduino/cores/arduino/WInterrupts.o \
./Arduino\ base/hardware/arduino/cores/arduino/WMath.o \
./Arduino\ base/hardware/arduino/cores/arduino/WString.o \
./Arduino\ base/hardware/arduino/cores/arduino/main.o \
./Arduino\ base/hardware/arduino/cores/arduino/new.o \
./Arduino\ base/hardware/arduino/cores/arduino/wiring.o \
./Arduino\ base/hardware/arduino/cores/arduino/wiring_analog.o \
./Arduino\ base/hardware/arduino/cores/arduino/wiring_digital.o \
./Arduino\ base/hardware/arduino/cores/arduino/wiring_pulse.o \
./Arduino\ base/hardware/arduino/cores/arduino/wiring_shift.o 

C_DEPS += \
./Arduino\ base/hardware/arduino/cores/arduino/WInterrupts.d \
./Arduino\ base/hardware/arduino/cores/arduino/wiring.d \
./Arduino\ base/hardware/arduino/cores/arduino/wiring_analog.d \
./Arduino\ base/hardware/arduino/cores/arduino/wiring_digital.d \
./Arduino\ base/hardware/arduino/cores/arduino/wiring_pulse.d \
./Arduino\ base/hardware/arduino/cores/arduino/wiring_shift.d 

CPP_DEPS += \
./Arduino\ base/hardware/arduino/cores/arduino/CDC.d \
./Arduino\ base/hardware/arduino/cores/arduino/HID.d \
./Arduino\ base/hardware/arduino/cores/arduino/HardwareSerial.d \
./Arduino\ base/hardware/arduino/cores/arduino/IPAddress.d \
./Arduino\ base/hardware/arduino/cores/arduino/Print.d \
./Arduino\ base/hardware/arduino/cores/arduino/Stream.d \
./Arduino\ base/hardware/arduino/cores/arduino/Tone.d \
./Arduino\ base/hardware/arduino/cores/arduino/USBCore.d \
./Arduino\ base/hardware/arduino/cores/arduino/WMath.d \
./Arduino\ base/hardware/arduino/cores/arduino/WString.d \
./Arduino\ base/hardware/arduino/cores/arduino/main.d \
./Arduino\ base/hardware/arduino/cores/arduino/new.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino\ base/hardware/arduino/cores/arduino/CDC.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/CDC.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/CDC.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/CDC.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/HID.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/HID.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/HID.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/HID.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/HardwareSerial.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/HardwareSerial.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/HardwareSerial.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/HardwareSerial.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/IPAddress.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/IPAddress.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/IPAddress.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/IPAddress.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/Print.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/Print.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/Print.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/Print.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/Stream.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/Stream.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/Stream.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/Stream.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/Tone.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/Tone.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/Tone.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/Tone.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/USBCore.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/USBCore.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/USBCore.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/USBCore.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/WInterrupts.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/WInterrupts.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/WInterrupts.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/WInterrupts.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/WMath.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/WMath.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/WMath.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/WMath.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/WString.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/WString.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/WString.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/WString.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/main.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/main.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/main.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/main.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/new.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/new.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/new.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/new.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/wiring.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/wiring.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/wiring.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/wiring.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/wiring_analog.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/wiring_analog.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/wiring_analog.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/wiring_analog.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/wiring_digital.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/wiring_digital.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/wiring_digital.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/wiring_digital.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/wiring_pulse.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/wiring_pulse.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/wiring_pulse.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/wiring_pulse.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino\ base/hardware/arduino/cores/arduino/wiring_shift.o: /Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/wiring_shift.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Arduino base/hardware/arduino/cores/arduino/wiring_shift.d" -MT"Arduino\ base/hardware/arduino/cores/arduino/wiring_shift.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


