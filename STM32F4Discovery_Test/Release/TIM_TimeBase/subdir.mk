################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../TIM_TimeBase/main.c \
../TIM_TimeBase/stm32f4xx_it.c \
../TIM_TimeBase/system_stm32f4xx.c 

OBJS += \
./TIM_TimeBase/main.o \
./TIM_TimeBase/stm32f4xx_it.o \
./TIM_TimeBase/system_stm32f4xx.o 

C_DEPS += \
./TIM_TimeBase/main.d \
./TIM_TimeBase/stm32f4xx_it.d \
./TIM_TimeBase/system_stm32f4xx.d 


# Each subdirectory must supply rules for building sources it contributes
TIM_TimeBase/%.o: ../TIM_TimeBase/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/STM32F4-Discovery" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/STM32_USB_Device_Library/Class/hid/inc" -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


