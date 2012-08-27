################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../cores/arduino/CDC.cpp \
../cores/arduino/HID.cpp \
../cores/arduino/HardwareSerial.cpp \
../cores/arduino/IPAddress.cpp \
../cores/arduino/Print.cpp \
../cores/arduino/Stream.cpp \
../cores/arduino/Tone.cpp \
../cores/arduino/USBCore.cpp \
../cores/arduino/WMath.cpp \
../cores/arduino/WString.cpp \
../cores/arduino/new.cpp 

C_SRCS += \
../cores/arduino/WInterrupts.c \
../cores/arduino/wiring.c \
../cores/arduino/wiring_analog.c \
../cores/arduino/wiring_digital.c \
../cores/arduino/wiring_pulse.c \
../cores/arduino/wiring_shift.c 

OBJS += \
./cores/arduino/CDC.o \
./cores/arduino/HID.o \
./cores/arduino/HardwareSerial.o \
./cores/arduino/IPAddress.o \
./cores/arduino/Print.o \
./cores/arduino/Stream.o \
./cores/arduino/Tone.o \
./cores/arduino/USBCore.o \
./cores/arduino/WInterrupts.o \
./cores/arduino/WMath.o \
./cores/arduino/WString.o \
./cores/arduino/new.o \
./cores/arduino/wiring.o \
./cores/arduino/wiring_analog.o \
./cores/arduino/wiring_digital.o \
./cores/arduino/wiring_pulse.o \
./cores/arduino/wiring_shift.o 

C_DEPS += \
./cores/arduino/WInterrupts.d \
./cores/arduino/wiring.d \
./cores/arduino/wiring_analog.d \
./cores/arduino/wiring_digital.d \
./cores/arduino/wiring_pulse.d \
./cores/arduino/wiring_shift.d 

CPP_DEPS += \
./cores/arduino/CDC.d \
./cores/arduino/HID.d \
./cores/arduino/HardwareSerial.d \
./cores/arduino/IPAddress.d \
./cores/arduino/Print.d \
./cores/arduino/Stream.d \
./cores/arduino/Tone.d \
./cores/arduino/USBCore.d \
./cores/arduino/WMath.d \
./cores/arduino/WString.d \
./cores/arduino/new.d 


# Each subdirectory must supply rules for building sources it contributes
cores/arduino/%.o: ../cores/arduino/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Users/sin/Documents/Workspace/Arduino_eclipse_build/cores/arduino" -I"/Users/sin/Documents/Workspace/Arduino_eclipse_build/variants/standard" -I"/Users/sin/Documents/Workspace/Arduino_eclipse_build/libraries" -I"/Users/sin/Documents/Arduino/libraries" -DARDUINO=101 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

cores/arduino/%.o: ../cores/arduino/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Users/sin/Documents/Workspace/Arduino_eclipse_build/cores/arduino" -I"/Users/sin/Documents/Workspace/Arduino_eclipse_build/variants/standard" -I"/Users/sin/Documents/Workspace/Arduino_eclipse_build/libraries" -I"/Users/sin/Documents/Arduino/libraries" -DARDUINO=101 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


