################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../SysTick/main.c \
../SysTick/stm32f4xx_it.c \
../SysTick/system_stm32f4xx.c 

OBJS += \
./SysTick/main.o \
./SysTick/stm32f4xx_it.o \
./SysTick/system_stm32f4xx.o 

C_DEPS += \
./SysTick/main.d \
./SysTick/stm32f4xx_it.d \
./SysTick/system_stm32f4xx.d 


# Each subdirectory must supply rules for building sources it contributes
SysTick/%.o: ../SysTick/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/STM32F4-Discovery" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/STM32_USB_Device_Library/Class/hid/inc" -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


