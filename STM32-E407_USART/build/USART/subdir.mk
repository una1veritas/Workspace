################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../USART/main.cpp 

C_SRCS += \
../USART/system_stm32f4xx.c 

S_UPPER_SRCS += \
../USART/startup_stm32f4xx.gcc.S 

OBJS += \
./USART/main.o \
./USART/startup_stm32f4xx.gcc.o \
./USART/system_stm32f4xx.o 

C_DEPS += \
./USART/system_stm32f4xx.d 

CPP_DEPS += \
./USART/main.d 


# Each subdirectory must supply rules for building sources it contributes
USART/%.o: ../USART/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	arm-none-eabi-g++ -DUSE_STDPERIPH_DRIVER=1 -DHSE_VALUE=25000000 -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_USART/USART" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_USART/armcore" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/STM32F4xx_StdPeriph_Driver/inc" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -mfpu=fpv4-sp-d16 -g -Wall -c -fmessage-length=0 -ffreestanding -fno-exceptions -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

USART/%.o: ../USART/%.S
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Assembler'
	arm-none-eabi-as -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/STM32F4xx_StdPeriph_Driver/inc" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

USART/%.o: ../USART/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -DHSE_VALUE=25000000 -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_USART/USART" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_USART/armcore" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/STM32F4xx_StdPeriph_Driver/inc" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -mfpu=fpv4-sp-d16 -g -Wall -c -fmessage-length=0 -ffreestanding -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


