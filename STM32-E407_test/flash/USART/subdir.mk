################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../USART/main.cpp 

C_SRCS += \
../USART/olimex_stm32-e407.c \
../USART/system_stm32f4xx.c 

OBJS += \
./USART/main.o \
./USART/olimex_stm32-e407.o \
./USART/system_stm32f4xx.o 

C_DEPS += \
./USART/olimex_stm32-e407.d \
./USART/system_stm32f4xx.d 

CPP_DEPS += \
./USART/main.d 


# Each subdirectory must supply rules for building sources it contributes
USART/%.o: ../USART/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	arm-none-eabi-g++ -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/GPIO" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/mycore" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/USART" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -mfpu=vfpv4 -g -Wall -c -fmessage-length=0 -ffreestanding -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

USART/%.o: ../USART/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERPH -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/GPIO" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/mycore" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/USART" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -mfpu=vfpv4 -g -Wall -c -fmessage-length=0 -ffreestanding -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


