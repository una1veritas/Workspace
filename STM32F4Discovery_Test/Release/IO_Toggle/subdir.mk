################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../IO_Toggle/main.c \
../IO_Toggle/stm32f4xx_it.c \
../IO_Toggle/system_stm32f4xx.c 

OBJS += \
./IO_Toggle/main.o \
./IO_Toggle/stm32f4xx_it.o \
./IO_Toggle/system_stm32f4xx.o 

C_DEPS += \
./IO_Toggle/main.d \
./IO_Toggle/stm32f4xx_it.d \
./IO_Toggle/system_stm32f4xx.d 


# Each subdirectory must supply rules for building sources it contributes
IO_Toggle/%.o: ../IO_Toggle/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/STM32F4-Discovery" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/STM32_USB_Device_Library/Class/hid/inc" -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


