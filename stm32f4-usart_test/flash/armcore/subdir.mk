################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../armcore/gpio.c \
../armcore/systick.c \
../armcore/usart.c 

OBJS += \
./armcore/gpio.o \
./armcore/systick.o \
./armcore/usart.o 

C_DEPS += \
./armcore/gpio.d \
./armcore/systick.d \
./armcore/usart.d 


# Each subdirectory must supply rules for building sources it contributes
armcore/%.o: ../armcore/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-usart_test" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/STM32F4xx_StdPeriph_Driver/inc" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


