################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/CDC.cpp \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/HID.cpp \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/HardwareSerial.cpp \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/IPAddress.cpp \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/Print.cpp \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/Stream.cpp \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/Tone.cpp \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/USBCore.cpp \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/WMath.cpp \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/WString.cpp \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/main.cpp \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/new.cpp 

C_SRCS += \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/WInterrupts.c \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/wiring.c \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/wiring_analog.c \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/wiring_digital.c \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/wiring_pulse.c \
/Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/wiring_shift.c 

OBJS += \
./Home\ Arduino/hardware/ProMicro/cores/arduino/CDC.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/HID.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/HardwareSerial.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/IPAddress.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/Print.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/Stream.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/Tone.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/USBCore.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/WInterrupts.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/WMath.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/WString.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/main.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/new.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/wiring.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/wiring_analog.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/wiring_digital.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/wiring_pulse.o \
./Home\ Arduino/hardware/ProMicro/cores/arduino/wiring_shift.o 

C_DEPS += \
./Home\ Arduino/hardware/ProMicro/cores/arduino/WInterrupts.d \
./Home\ Arduino/hardware/ProMicro/cores/arduino/wiring.d \
./Home\ Arduino/hardware/ProMicro/cores/arduino/wiring_analog.d \
./Home\ Arduino/hardware/ProMicro/cores/arduino/wiring_digital.d \
./Home\ Arduino/hardware/ProMicro/cores/arduino/wiring_pulse.d \
./Home\ Arduino/hardware/ProMicro/cores/arduino/wiring_shift.d 

CPP_DEPS += \
./Home\ Arduino/hardware/ProMicro/cores/arduino/CDC.d \
./Home\ Arduino/hardware/ProMicro/cores/arduino/HID.d \
./Home\ Arduino/hardware/ProMicro/cores/arduino/HardwareSerial.d \
./Home\ Arduino/hardware/ProMicro/cores/arduino/IPAddress.d \
./Home\ Arduino/hardware/ProMicro/cores/arduino/Print.d \
./Home\ Arduino/hardware/ProMicro/cores/arduino/Stream.d \
./Home\ Arduino/hardware/ProMicro/cores/arduino/Tone.d \
./Home\ Arduino/hardware/ProMicro/cores/arduino/USBCore.d \
./Home\ Arduino/hardware/ProMicro/cores/arduino/WMath.d \
./Home\ Arduino/hardware/ProMicro/cores/arduino/WString.d \
./Home\ Arduino/hardware/ProMicro/cores/arduino/main.d \
./Home\ Arduino/hardware/ProMicro/cores/arduino/new.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ Arduino/hardware/ProMicro/cores/arduino/CDC.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/CDC.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/CDC.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/CDC.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/HID.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/HID.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/HID.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/HID.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/HardwareSerial.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/HardwareSerial.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/HardwareSerial.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/HardwareSerial.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/IPAddress.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/IPAddress.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/IPAddress.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/IPAddress.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/Print.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/Print.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/Print.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/Print.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/Stream.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/Stream.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/Stream.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/Stream.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/Tone.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/Tone.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/Tone.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/Tone.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/USBCore.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/USBCore.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/USBCore.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/USBCore.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/WInterrupts.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/WInterrupts.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/WInterrupts.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/WInterrupts.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/WMath.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/WMath.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/WMath.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/WMath.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/WString.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/WString.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/WString.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/WString.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/main.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/main.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/main.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/main.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/new.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/new.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/new.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/new.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/wiring.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/wiring.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/wiring.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/wiring.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/wiring_analog.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/wiring_analog.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/wiring_analog.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/wiring_analog.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/wiring_digital.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/wiring_digital.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/wiring_digital.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/wiring_digital.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/wiring_pulse.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/wiring_pulse.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/wiring_pulse.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/wiring_pulse.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ Arduino/hardware/ProMicro/cores/arduino/wiring_shift.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/wiring_shift.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home Arduino/hardware/ProMicro/cores/arduino/wiring_shift.d" -MT"Home\ Arduino/hardware/ProMicro/cores/arduino/wiring_shift.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


