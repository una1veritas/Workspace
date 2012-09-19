################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../STM32F4-Discovery/stm32f4_discovery.c \
../STM32F4-Discovery/stm32f4_discovery_audio_codec.c \
../STM32F4-Discovery/stm32f4_discovery_lis302dl.c 

OBJS += \
./STM32F4-Discovery/stm32f4_discovery.o \
./STM32F4-Discovery/stm32f4_discovery_audio_codec.o \
./STM32F4-Discovery/stm32f4_discovery_lis302dl.o 

C_DEPS += \
./STM32F4-Discovery/stm32f4_discovery.d \
./STM32F4-Discovery/stm32f4_discovery_audio_codec.d \
./STM32F4-Discovery/stm32f4_discovery_lis302dl.d 


# Each subdirectory must supply rules for building sources it contributes
STM32F4-Discovery/%.o: ../STM32F4-Discovery/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -I"/Users/sin/Documents/Eclipse/Workspace/STM32/STM32F4xx_library/inc/core_support" -I"/Users/sin/Documents/Eclipse/Workspace/STM32/STM32F4xx_library/inc/device_support" -I"/Users/sin/Documents/Eclipse/Workspace/STM32/STM32F4xx_library/inc" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


