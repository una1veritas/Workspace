################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../IO_toggle/main.c \
../IO_toggle/stm32f4xx_it.c \
../IO_toggle/system_stm32f4xx.c 

ASM_SRCS += \
../IO_toggle/startup_stm32f4xx.asm 

OBJS += \
./IO_toggle/main.o \
./IO_toggle/startup_stm32f4xx.o \
./IO_toggle/stm32f4xx_it.o \
./IO_toggle/system_stm32f4xx.o 

C_DEPS += \
./IO_toggle/main.d \
./IO_toggle/stm32f4xx_it.d \
./IO_toggle/system_stm32f4xx.d 


# Each subdirectory must supply rules for building sources it contributes
IO_toggle/%.o: ../IO_toggle/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_IO_toggle/IO_toggle" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/Utilities/STM32F4-Discovery" -O2 -mcpu=cortex-m4 -mthumb -mlittle-endian -ffreestanding -g -Wall -c -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

IO_toggle/%.o: ../IO_toggle/%.asm
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Assembler'
	arm-none-eabi-as -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/Utilities/STM32F4-Discovery" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


