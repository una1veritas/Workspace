################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../bootloaders/caterina/Caterina.c \
../bootloaders/caterina/Descriptors.c 

OBJS += \
./bootloaders/caterina/Caterina.o \
./bootloaders/caterina/Descriptors.o 

C_DEPS += \
./bootloaders/caterina/Caterina.d \
./bootloaders/caterina/Descriptors.d 


# Each subdirectory must supply rules for building sources it contributes
bootloaders/caterina/%.o: ../bootloaders/caterina/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Users/sin/Documents/Eclipse/Workspace/TinyBasicPlus/cores/arduino" -I"/Users/sin/Documents/Eclipse/Workspace/TinyBasicPlus/variants/standard" -I"/Users/sin/Documents/Eclipse/Workspace/TinyBasicPlus/libraries/SD" -I"/Users/sin/Documents/Eclipse/Workspace/TinyBasicPlus/libraries/SD/utility" -DARDUINO=101 -Wall -g2 -gstabs -O0 -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


