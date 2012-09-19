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
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/CDC.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/HID.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/HardwareSerial.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/IPAddress.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/Print.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/Stream.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/Tone.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/USBCore.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/WInterrupts.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/WMath.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/WString.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/main.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/new.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/wiring.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/wiring_analog.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/wiring_digital.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/wiring_pulse.o \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/wiring_shift.o 

C_DEPS += \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/WInterrupts.d \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/wiring.d \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/wiring_analog.d \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/wiring_digital.d \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/wiring_pulse.d \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/wiring_shift.d 

CPP_DEPS += \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/CDC.d \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/HID.d \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/HardwareSerial.d \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/IPAddress.d \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/Print.d \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/Stream.d \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/Tone.d \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/USBCore.d \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/WMath.d \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/WString.d \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/main.d \
./Arduino_sketch_home/hardware/ProMicro/cores/arduino/new.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino_sketch_home/hardware/ProMicro/cores/arduino/CDC.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/CDC.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/HID.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/HID.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/HardwareSerial.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/HardwareSerial.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/IPAddress.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/IPAddress.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/Print.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/Print.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/Stream.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/Stream.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/Tone.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/Tone.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/USBCore.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/USBCore.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/WInterrupts.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/WInterrupts.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/WMath.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/WMath.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/WString.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/WString.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/main.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/main.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/new.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/new.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/wiring.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/wiring.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/wiring_analog.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/wiring_analog.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/wiring_digital.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/wiring_digital.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/wiring_pulse.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/wiring_pulse.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/cores/arduino/wiring_shift.o: /Users/sin/Documents/Arduino/hardware/ProMicro/cores/arduino/wiring_shift.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


