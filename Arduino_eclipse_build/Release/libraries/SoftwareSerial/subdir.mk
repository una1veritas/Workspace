################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../libraries/SoftwareSerial/SoftwareSerial.cpp 

OBJS += \
./libraries/SoftwareSerial/SoftwareSerial.o 

CPP_DEPS += \
./libraries/SoftwareSerial/SoftwareSerial.d 


# Each subdirectory must supply rules for building sources it contributes
libraries/SoftwareSerial/%.o: ../libraries/SoftwareSerial/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Users/sin/Documents/Workspace/Arduino_eclipse_build/cores/arduino" -I"/Users/sin/Documents/Workspace/Arduino_eclipse_build/variants/standard" -I"/Users/sin/Documents/Workspace/Arduino_eclipse_build/libraries" -I"/Users/sin/Documents/Arduino/libraries" -DARDUINO=101 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


