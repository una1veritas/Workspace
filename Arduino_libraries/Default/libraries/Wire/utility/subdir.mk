################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../libraries/Wire/utility/twi.c 

OBJS += \
./libraries/Wire/utility/twi.o 

C_DEPS += \
./libraries/Wire/utility/twi.d 


# Each subdirectory must supply rules for building sources it contributes
libraries/Wire/utility/%.o: ../libraries/Wire/utility/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Users/sin/Documents/Eclipse/Workspace/Arduino_cores/arduino/cores/arduino" -I"/Users/sin/Documents/Eclipse/Workspace/Arduino_cores/arduino/variants/standard" -I"/Users/sin/Documents/Eclipse/Workspace/Arduino_libraries/libraries" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


