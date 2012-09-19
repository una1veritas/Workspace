################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../CMSIS/Device/ST/STM32F10x/Source/Templates/system_stm32f10x.c 

OBJS += \
./CMSIS/Device/ST/STM32F10x/Source/Templates/system_stm32f10x.o 

C_DEPS += \
./CMSIS/Device/ST/STM32F10x/Source/Templates/system_stm32f10x.d 


# Each subdirectory must supply rules for building sources it contributes
CMSIS/Device/ST/STM32F10x/Source/Templates/%.o: ../CMSIS/Device/ST/STM32F10x/Source/Templates/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/STM32F4-Discovery" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/STM32_USB_Device_Library/Class/hid/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/CMSIS/Device/ST/STM32F4xx/Include" -O0 -mcpu=cortex-m4 -mthumb -mlittle-enduan -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


