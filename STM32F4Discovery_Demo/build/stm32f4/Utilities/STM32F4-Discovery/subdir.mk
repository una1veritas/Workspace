################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../stm32f4/Utilities/STM32F4-Discovery/stm32f4_discovery.c \
../stm32f4/Utilities/STM32F4-Discovery/stm32f4_discovery_audio_codec.c \
../stm32f4/Utilities/STM32F4-Discovery/stm32f4_discovery_lis302dl.c 

OBJS += \
./stm32f4/Utilities/STM32F4-Discovery/stm32f4_discovery.o \
./stm32f4/Utilities/STM32F4-Discovery/stm32f4_discovery_audio_codec.o \
./stm32f4/Utilities/STM32F4-Discovery/stm32f4_discovery_lis302dl.o 

C_DEPS += \
./stm32f4/Utilities/STM32F4-Discovery/stm32f4_discovery.d \
./stm32f4/Utilities/STM32F4-Discovery/stm32f4_discovery_audio_codec.d \
./stm32f4/Utilities/STM32F4-Discovery/stm32f4_discovery_lis302dl.d 


# Each subdirectory must supply rules for building sources it contributes
stm32f4/Utilities/STM32F4-Discovery/%.o: ../stm32f4/Utilities/STM32F4-Discovery/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_USB_OTG_FS=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/Projects/discovery_demo" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/stm32f4/STM32_USB_Device_Library/Core/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/stm32f4/STM32_USB_Device_Library/Class/hid/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/stm32f4/STM32_USB_OTG_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/stm32f4/Utilities/STM32F4-Discovery" -O2 -mcpu=cortex-m4 -mthumb -mlittle-endian -ffreestanding -g -Wall -c -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


