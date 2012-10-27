################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore/gpio.c \
/Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore/systick.c \
/Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore/usart3.c 

OBJS += \
./armcore/gpio.o \
./armcore/systick.o \
./armcore/usart3.o 

C_DEPS += \
./armcore/gpio.d \
./armcore/systick.d \
./armcore/usart3.d 


# Each subdirectory must supply rules for building sources it contributes
armcore/gpio.o: /Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore/gpio.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-usart_test" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

armcore/systick.o: /Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore/systick.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-usart_test" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

armcore/usart3.o: /Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore/usart3.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-usart_test" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


