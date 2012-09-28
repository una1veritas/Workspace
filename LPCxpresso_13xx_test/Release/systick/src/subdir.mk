################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../systick/src/cr_startup_lpc13.c \
../systick/src/gpio.c \
../systick/src/systick_main.c 

OBJS += \
./systick/src/cr_startup_lpc13.o \
./systick/src/gpio.o \
./systick/src/systick_main.o 

C_DEPS += \
./systick/src/cr_startup_lpc13.d \
./systick/src/gpio.d \
./systick/src/systick_main.d 


# Each subdirectory must supply rules for building sources it contributes
systick/src/%.o: ../systick/src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -O2 -c -mthumb -mcpu=cortex-m3 -mfix-cortex-m3-ldrd -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


