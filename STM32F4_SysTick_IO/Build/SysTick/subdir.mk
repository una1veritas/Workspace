################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../SysTick/delay.c \
../SysTick/main.c \
../SysTick/system_stm32f4xx.c 

S_UPPER_SRCS += \
../SysTick/startup_stm32f4xx.S 

OBJS += \
./SysTick/delay.o \
./SysTick/main.o \
./SysTick/startup_stm32f4xx.o \
./SysTick/system_stm32f4xx.o 

C_DEPS += \
./SysTick/delay.d \
./SysTick/main.d \
./SysTick/system_stm32f4xx.d 


# Each subdirectory must supply rules for building sources it contributes
SysTick/%.o: ../SysTick/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_SysTick_IO/SysTick" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/stm32f4/Utilities/STM32F4-Discovery" -O2 -mthumb -mcpu=cortex-m4 -mlittle-endian -ffreestanding -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

SysTick/%.o: ../SysTick/%.S
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Assembler'
	arm-none-eabi-as -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/stm32f4/Utilities/STM32F4-Discovery" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


