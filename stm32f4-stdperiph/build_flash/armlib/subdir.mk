################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Users/sin/Documents/Eclipse/Workspace/stm32library/armlib/ST7032i.c 

OBJS += \
./armlib/ST7032i.o 

C_DEPS += \
./armlib/ST7032i.d 


# Each subdirectory must supply rules for building sources it contributes
armlib/ST7032i.o: /Users/sin/Documents/Eclipse/Workspace/stm32library/armlib/ST7032i.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/armcore" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-stdperiph" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/armlib" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


