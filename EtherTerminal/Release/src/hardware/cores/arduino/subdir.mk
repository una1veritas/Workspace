################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/hardware/cores/arduino/CDC.cpp \
../src/hardware/cores/arduino/HID.cpp \
../src/hardware/cores/arduino/HardwareSerial.cpp \
../src/hardware/cores/arduino/IPAddress.cpp \
../src/hardware/cores/arduino/Print.cpp \
../src/hardware/cores/arduino/Stream.cpp \
../src/hardware/cores/arduino/Tone.cpp \
../src/hardware/cores/arduino/USBCore.cpp \
../src/hardware/cores/arduino/WMath.cpp \
../src/hardware/cores/arduino/WString.cpp \
../src/hardware/cores/arduino/main.cpp \
../src/hardware/cores/arduino/new.cpp 

C_SRCS += \
../src/hardware/cores/arduino/WInterrupts.c \
../src/hardware/cores/arduino/wiring.c \
../src/hardware/cores/arduino/wiring_analog.c \
../src/hardware/cores/arduino/wiring_digital.c \
../src/hardware/cores/arduino/wiring_pulse.c \
../src/hardware/cores/arduino/wiring_shift.c 

OBJS += \
./src/hardware/cores/arduino/CDC.o \
./src/hardware/cores/arduino/HID.o \
./src/hardware/cores/arduino/HardwareSerial.o \
./src/hardware/cores/arduino/IPAddress.o \
./src/hardware/cores/arduino/Print.o \
./src/hardware/cores/arduino/Stream.o \
./src/hardware/cores/arduino/Tone.o \
./src/hardware/cores/arduino/USBCore.o \
./src/hardware/cores/arduino/WInterrupts.o \
./src/hardware/cores/arduino/WMath.o \
./src/hardware/cores/arduino/WString.o \
./src/hardware/cores/arduino/main.o \
./src/hardware/cores/arduino/new.o \
./src/hardware/cores/arduino/wiring.o \
./src/hardware/cores/arduino/wiring_analog.o \
./src/hardware/cores/arduino/wiring_digital.o \
./src/hardware/cores/arduino/wiring_pulse.o \
./src/hardware/cores/arduino/wiring_shift.o 

C_DEPS += \
./src/hardware/cores/arduino/WInterrupts.d \
./src/hardware/cores/arduino/wiring.d \
./src/hardware/cores/arduino/wiring_analog.d \
./src/hardware/cores/arduino/wiring_digital.d \
./src/hardware/cores/arduino/wiring_pulse.d \
./src/hardware/cores/arduino/wiring_shift.d 

CPP_DEPS += \
./src/hardware/cores/arduino/CDC.d \
./src/hardware/cores/arduino/HID.d \
./src/hardware/cores/arduino/HardwareSerial.d \
./src/hardware/cores/arduino/IPAddress.d \
./src/hardware/cores/arduino/Print.d \
./src/hardware/cores/arduino/Stream.d \
./src/hardware/cores/arduino/Tone.d \
./src/hardware/cores/arduino/USBCore.d \
./src/hardware/cores/arduino/WMath.d \
./src/hardware/cores/arduino/WString.d \
./src/hardware/cores/arduino/main.d \
./src/hardware/cores/arduino/new.d 


# Each subdirectory must supply rules for building sources it contributes
src/hardware/cores/arduino/%.o: ../src/hardware/cores/arduino/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Users/sin/Documents/Workspace/EtherTerminal/src/hardware/variants/quaranta" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/hardware/cores/arduino/%.o: ../src/hardware/cores/arduino/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Users/sin/Documents/Workspace/EtherTerminal/src/hardware/variants/quaranta" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


