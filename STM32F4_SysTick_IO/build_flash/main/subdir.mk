################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../main/main.cpp 

C_SRCS += \
../main/delay.c \
../main/stm32f4xx_it.c \
../main/system_stm32f4xx.c 

S_UPPER_SRCS += \
../main/startup_stm32f4xx.S 

OBJS += \
./main/delay.o \
./main/main.o \
./main/startup_stm32f4xx.o \
./main/stm32f4xx_it.o \
./main/system_stm32f4xx.o 

C_DEPS += \
./main/delay.d \
./main/stm32f4xx_it.d \
./main/system_stm32f4xx.d 

CPP_DEPS += \
./main/main.d 


# Each subdirectory must supply rules for building sources it contributes
main/%.o: ../main/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_SysTick_IO/main" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_SysTick_IO/board" -O2 -mthumb -mcpu=cortex-m4 -mlittle-endian -ffreestanding -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

main/%.o: ../main/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	arm-none-eabi-g++ -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_SysTick_IO/board" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_SysTick_IO/main" -O2 -mthumb -mcpu=cortex-m4 -mlittle-endian -ffreestanding -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

main/%.o: ../main/%.S
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Assembler'
	arm-none-eabi-as -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_SysTick_IO" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


