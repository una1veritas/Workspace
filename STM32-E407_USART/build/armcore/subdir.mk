################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../armcore/gpio_digital.cpp \
../armcore/systick.cpp 

C_SRCS += \
../armcore/delay.c 

OBJS += \
./armcore/delay.o \
./armcore/gpio_digital.o \
./armcore/systick.o 

C_DEPS += \
./armcore/delay.d 

CPP_DEPS += \
./armcore/gpio_digital.d \
./armcore/systick.d 


# Each subdirectory must supply rules for building sources it contributes
armcore/%.o: ../armcore/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -DHSE_VALUE=25000000 -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_USART/USART" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_USART/armcore" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/STM32F4xx_StdPeriph_Driver/inc" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -mfpu=fpv4-sp-d16 -g -Wall -c -fmessage-length=0 -ffreestanding -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

armcore/%.o: ../armcore/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	arm-none-eabi-g++ -DUSE_STDPERIPH_DRIVER=1 -DHSE_VALUE=25000000 -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_USART/USART" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_USART/armcore" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/STM32F4xx_StdPeriph_Driver/inc" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -mfpu=fpv4-sp-d16 -g -Wall -c -fmessage-length=0 -ffreestanding -fno-exceptions -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


