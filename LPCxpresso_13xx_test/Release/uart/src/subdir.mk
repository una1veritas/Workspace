################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../uart/src/cr_startup_lpc13.c \
../uart/src/gpio.c \
../uart/src/timer32.c \
../uart/src/uart.c \
../uart/src/uarttest.c 

OBJS += \
./uart/src/cr_startup_lpc13.o \
./uart/src/gpio.o \
./uart/src/timer32.o \
./uart/src/uart.o \
./uart/src/uarttest.o 

C_DEPS += \
./uart/src/cr_startup_lpc13.d \
./uart/src/gpio.d \
./uart/src/timer32.d \
./uart/src/uart.d \
./uart/src/uarttest.d 


# Each subdirectory must supply rules for building sources it contributes
uart/src/%.o: ../uart/src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -O2 -c -mthumb -mcpu=cortex-m3 -mfix-cortex-m3-ldrd -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


