################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../USART/main.c \
../USART/stm32f4xx_it.c \
../USART/system_stm32f4xx.c 

S_UPPER_SRCS += \
../USART/startup_stm32f4xx.S 

OBJS += \
./USART/main.o \
./USART/startup_stm32f4xx.o \
./USART/stm32f4xx_it.o \
./USART/system_stm32f4xx.o 

C_DEPS += \
./USART/main.d \
./USART/stm32f4xx_it.d \
./USART/system_stm32f4xx.d 


# Each subdirectory must supply rules for building sources it contributes
USART/%.o: ../USART/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -DHSE_VALUE=25000000 -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_USART/USART" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_USART/armcore" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -mfpu=fpv4-sp-d16 -g -Wall -c -fmessage-length=0 -ffreestanding -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

USART/%.o: ../USART/%.S
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Assembler'
	arm-none-eabi-as  -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


