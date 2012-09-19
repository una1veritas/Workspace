################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
ASM_SRCS += \
../CMSIS/Device/ST/STM32F4xx/Source/Templates/TASKING/cstart_thumb2.asm 

OBJS += \
./CMSIS/Device/ST/STM32F4xx/Source/Templates/TASKING/cstart_thumb2.o 


# Each subdirectory must supply rules for building sources it contributes
CMSIS/Device/ST/STM32F4xx/Source/Templates/TASKING/%.o: ../CMSIS/Device/ST/STM32F4xx/Source/Templates/TASKING/%.asm
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Assembler'
	arm-none-eabi-as -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/STM32F4-Discovery" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/STM32_USB_Device_Library/Class/hid/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/CMSIS/Device/ST/STM32F4xx/Include" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


