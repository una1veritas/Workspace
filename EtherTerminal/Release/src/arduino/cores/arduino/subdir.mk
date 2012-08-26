################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/arduino/cores/arduino/CDC.cpp \
../src/arduino/cores/arduino/HID.cpp \
../src/arduino/cores/arduino/HardwareSerial.cpp \
../src/arduino/cores/arduino/IPAddress.cpp \
../src/arduino/cores/arduino/Print.cpp \
../src/arduino/cores/arduino/Stream.cpp \
../src/arduino/cores/arduino/Tone.cpp \
../src/arduino/cores/arduino/USBCore.cpp \
../src/arduino/cores/arduino/WMath.cpp \
../src/arduino/cores/arduino/WString.cpp \
../src/arduino/cores/arduino/main.cpp \
../src/arduino/cores/arduino/new.cpp 

C_SRCS += \
../src/arduino/cores/arduino/WInterrupts.c \
../src/arduino/cores/arduino/wiring.c \
../src/arduino/cores/arduino/wiring_analog.c \
../src/arduino/cores/arduino/wiring_digital.c \
../src/arduino/cores/arduino/wiring_pulse.c \
../src/arduino/cores/arduino/wiring_shift.c 

OBJS += \
./src/arduino/cores/arduino/CDC.o \
./src/arduino/cores/arduino/HID.o \
./src/arduino/cores/arduino/HardwareSerial.o \
./src/arduino/cores/arduino/IPAddress.o \
./src/arduino/cores/arduino/Print.o \
./src/arduino/cores/arduino/Stream.o \
./src/arduino/cores/arduino/Tone.o \
./src/arduino/cores/arduino/USBCore.o \
./src/arduino/cores/arduino/WInterrupts.o \
./src/arduino/cores/arduino/WMath.o \
./src/arduino/cores/arduino/WString.o \
./src/arduino/cores/arduino/main.o \
./src/arduino/cores/arduino/new.o \
./src/arduino/cores/arduino/wiring.o \
./src/arduino/cores/arduino/wiring_analog.o \
./src/arduino/cores/arduino/wiring_digital.o \
./src/arduino/cores/arduino/wiring_pulse.o \
./src/arduino/cores/arduino/wiring_shift.o 

C_DEPS += \
./src/arduino/cores/arduino/WInterrupts.d \
./src/arduino/cores/arduino/wiring.d \
./src/arduino/cores/arduino/wiring_analog.d \
./src/arduino/cores/arduino/wiring_digital.d \
./src/arduino/cores/arduino/wiring_pulse.d \
./src/arduino/cores/arduino/wiring_shift.d 

CPP_DEPS += \
./src/arduino/cores/arduino/CDC.d \
./src/arduino/cores/arduino/HID.d \
./src/arduino/cores/arduino/HardwareSerial.d \
./src/arduino/cores/arduino/IPAddress.d \
./src/arduino/cores/arduino/Print.d \
./src/arduino/cores/arduino/Stream.d \
./src/arduino/cores/arduino/Tone.d \
./src/arduino/cores/arduino/USBCore.d \
./src/arduino/cores/arduino/WMath.d \
./src/arduino/cores/arduino/WString.d \
./src/arduino/cores/arduino/main.d \
./src/arduino/cores/arduino/new.d 


# Each subdirectory must supply rules for building sources it contributes
src/arduino/cores/arduino/%.o: ../src/arduino/cores/arduino/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/arduino/cores/arduino/%.o: ../src/arduino/cores/arduino/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


