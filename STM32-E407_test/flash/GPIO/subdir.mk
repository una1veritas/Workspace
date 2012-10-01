################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../GPIO/main.c \
../GPIO/system_stm32f4xx.c 

S_UPPER_SRCS += \
../GPIO/startup_stm32f4xx.S 

OBJS += \
./GPIO/main.o \
./GPIO/startup_stm32f4xx.o \
./GPIO/system_stm32f4xx.o 

C_DEPS += \
./GPIO/main.d \
./GPIO/system_stm32f4xx.d 


# Each subdirectory must supply rules for building sources it contributes
GPIO/%.o: ../GPIO/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERPH -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/GPIO" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/mycore" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/USART" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -mfpu=vfpv4 -g -Wall -c -fmessage-length=0 -ffreestanding -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

GPIO/%.o: ../GPIO/%.S
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Assembler'
	arm-none-eabi-as -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/GPIO" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/mycore" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/USART" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


