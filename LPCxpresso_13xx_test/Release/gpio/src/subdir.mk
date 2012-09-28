################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../gpio/src/cr_startup_lpc13.c \
../gpio/src/gpio.c \
../gpio/src/gpiotest.c 

OBJS += \
./gpio/src/cr_startup_lpc13.o \
./gpio/src/gpio.o \
./gpio/src/gpiotest.o 

C_DEPS += \
./gpio/src/cr_startup_lpc13.d \
./gpio/src/gpio.d \
./gpio/src/gpiotest.d 


# Each subdirectory must supply rules for building sources it contributes
gpio/src/%.o: ../gpio/src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -O2 -c -mthumb -mcpu=cortex-m3 -mfix-cortex-m3-ldrd -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


