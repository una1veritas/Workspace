################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../STM32F4-Discovery_FW_V1.1.0/Project/Peripheral_Examples/SysTick/main.c \
../STM32F4-Discovery_FW_V1.1.0/Project/Peripheral_Examples/SysTick/stm32f4xx_it.c \
../STM32F4-Discovery_FW_V1.1.0/Project/Peripheral_Examples/SysTick/system_stm32f4xx.c 

OBJS += \
./STM32F4-Discovery_FW_V1.1.0/Project/Peripheral_Examples/SysTick/main.o \
./STM32F4-Discovery_FW_V1.1.0/Project/Peripheral_Examples/SysTick/stm32f4xx_it.o \
./STM32F4-Discovery_FW_V1.1.0/Project/Peripheral_Examples/SysTick/system_stm32f4xx.o 

C_DEPS += \
./STM32F4-Discovery_FW_V1.1.0/Project/Peripheral_Examples/SysTick/main.d \
./STM32F4-Discovery_FW_V1.1.0/Project/Peripheral_Examples/SysTick/stm32f4xx_it.d \
./STM32F4-Discovery_FW_V1.1.0/Project/Peripheral_Examples/SysTick/system_stm32f4xx.d 


# Each subdirectory must supply rules for building sources it contributes
STM32F4-Discovery_FW_V1.1.0/Project/Peripheral_Examples/SysTick/%.o: ../STM32F4-Discovery_FW_V1.1.0/Project/Peripheral_Examples/SysTick/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_c++_test" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_c++_test/core" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -g3 -Wall -c -fmessage-length=0 -ffreestanding -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


