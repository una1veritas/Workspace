################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Users/sin/Documents/Eclipse/Workspace/STM32_library/stm32f4/Utilities/STM32F4-Discovery/stm32f4_discovery.c 

OBJS += \
./STM32F4-Discovery/stm32f4_discovery.o 

C_DEPS += \
./STM32F4-Discovery/stm32f4_discovery.d 


# Each subdirectory must supply rules for building sources it contributes
STM32F4-Discovery/stm32f4_discovery.o: /Users/sin/Documents/Eclipse/Workspace/STM32_library/stm32f4/Utilities/STM32F4-Discovery/stm32f4_discovery.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_SysTick_IO/SysTick" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/stm32f4/Utilities/STM32F4-Discovery" -O2 -mthumb -mcpu=cortex-m4 -mlittle-endian -ffreestanding -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


