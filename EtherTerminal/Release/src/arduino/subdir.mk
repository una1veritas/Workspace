################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/arduino/CDC.cpp \
../src/arduino/HID.cpp \
../src/arduino/HardwareSerial.cpp \
../src/arduino/IPAddress.cpp \
../src/arduino/Print.cpp \
../src/arduino/Stream.cpp \
../src/arduino/Tone.cpp \
../src/arduino/USBCore.cpp \
../src/arduino/WMath.cpp \
../src/arduino/WString.cpp \
../src/arduino/main.cpp \
../src/arduino/new.cpp 

C_SRCS += \
../src/arduino/WInterrupts.c \
../src/arduino/wiring.c \
../src/arduino/wiring_analog.c \
../src/arduino/wiring_digital.c \
../src/arduino/wiring_pulse.c \
../src/arduino/wiring_shift.c 

OBJS += \
./src/arduino/CDC.o \
./src/arduino/HID.o \
./src/arduino/HardwareSerial.o \
./src/arduino/IPAddress.o \
./src/arduino/Print.o \
./src/arduino/Stream.o \
./src/arduino/Tone.o \
./src/arduino/USBCore.o \
./src/arduino/WInterrupts.o \
./src/arduino/WMath.o \
./src/arduino/WString.o \
./src/arduino/main.o \
./src/arduino/new.o \
./src/arduino/wiring.o \
./src/arduino/wiring_analog.o \
./src/arduino/wiring_digital.o \
./src/arduino/wiring_pulse.o \
./src/arduino/wiring_shift.o 

C_DEPS += \
./src/arduino/WInterrupts.d \
./src/arduino/wiring.d \
./src/arduino/wiring_analog.d \
./src/arduino/wiring_digital.d \
./src/arduino/wiring_pulse.d \
./src/arduino/wiring_shift.d 

CPP_DEPS += \
./src/arduino/CDC.d \
./src/arduino/HID.d \
./src/arduino/HardwareSerial.d \
./src/arduino/IPAddress.d \
./src/arduino/Print.d \
./src/arduino/Stream.d \
./src/arduino/Tone.d \
./src/arduino/USBCore.d \
./src/arduino/WMath.d \
./src/arduino/WString.d \
./src/arduino/main.d \
./src/arduino/new.d 


# Each subdirectory must supply rules for building sources it contributes
src/arduino/%.o: ../src/arduino/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/arduino/%.o: ../src/arduino/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


