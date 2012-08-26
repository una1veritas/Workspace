################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../arduino/cores/arduino/CDC.cpp \
../arduino/cores/arduino/HID.cpp \
../arduino/cores/arduino/HardwareSerial.cpp \
../arduino/cores/arduino/IPAddress.cpp \
../arduino/cores/arduino/Print.cpp \
../arduino/cores/arduino/Stream.cpp \
../arduino/cores/arduino/Tone.cpp \
../arduino/cores/arduino/USBCore.cpp \
../arduino/cores/arduino/WMath.cpp \
../arduino/cores/arduino/WString.cpp \
../arduino/cores/arduino/main.cpp \
../arduino/cores/arduino/new.cpp 

C_SRCS += \
../arduino/cores/arduino/WInterrupts.c \
../arduino/cores/arduino/wiring.c \
../arduino/cores/arduino/wiring_analog.c \
../arduino/cores/arduino/wiring_digital.c \
../arduino/cores/arduino/wiring_pulse.c \
../arduino/cores/arduino/wiring_shift.c 

OBJS += \
./arduino/cores/arduino/CDC.o \
./arduino/cores/arduino/HID.o \
./arduino/cores/arduino/HardwareSerial.o \
./arduino/cores/arduino/IPAddress.o \
./arduino/cores/arduino/Print.o \
./arduino/cores/arduino/Stream.o \
./arduino/cores/arduino/Tone.o \
./arduino/cores/arduino/USBCore.o \
./arduino/cores/arduino/WInterrupts.o \
./arduino/cores/arduino/WMath.o \
./arduino/cores/arduino/WString.o \
./arduino/cores/arduino/main.o \
./arduino/cores/arduino/new.o \
./arduino/cores/arduino/wiring.o \
./arduino/cores/arduino/wiring_analog.o \
./arduino/cores/arduino/wiring_digital.o \
./arduino/cores/arduino/wiring_pulse.o \
./arduino/cores/arduino/wiring_shift.o 

C_DEPS += \
./arduino/cores/arduino/WInterrupts.d \
./arduino/cores/arduino/wiring.d \
./arduino/cores/arduino/wiring_analog.d \
./arduino/cores/arduino/wiring_digital.d \
./arduino/cores/arduino/wiring_pulse.d \
./arduino/cores/arduino/wiring_shift.d 

CPP_DEPS += \
./arduino/cores/arduino/CDC.d \
./arduino/cores/arduino/HID.d \
./arduino/cores/arduino/HardwareSerial.d \
./arduino/cores/arduino/IPAddress.d \
./arduino/cores/arduino/Print.d \
./arduino/cores/arduino/Stream.d \
./arduino/cores/arduino/Tone.d \
./arduino/cores/arduino/USBCore.d \
./arduino/cores/arduino/WMath.d \
./arduino/cores/arduino/WString.d \
./arduino/cores/arduino/main.d \
./arduino/cores/arduino/new.d 


# Each subdirectory must supply rules for building sources it contributes
arduino/cores/arduino/%.o: ../arduino/cores/arduino/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

arduino/cores/arduino/%.o: ../arduino/cores/arduino/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


