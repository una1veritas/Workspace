################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore/gpio_digital.c \
/Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore/systick.c \
/Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore/usart.c 

OBJS += \
./armcore/gpio_digital.o \
./armcore/systick.o \
./armcore/usart.o 

C_DEPS += \
./armcore/gpio_digital.d \
./armcore/systick.d \
./armcore/usart.d 


# Each subdirectory must supply rules for building sources it contributes
armcore/gpio_digital.o: /Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore/gpio_digital.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-float-math_test" -I"/Users/sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/Utilities/STM32F4-Discovery" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=hard -mfpu=fpv4-sp-d16 -g3 -Wall -c -fmessage-length=0 -fsingle-precision-constant -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

armcore/systick.o: /Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore/systick.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-float-math_test" -I"/Users/sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/Utilities/STM32F4-Discovery" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=hard -mfpu=fpv4-sp-d16 -g3 -Wall -c -fmessage-length=0 -fsingle-precision-constant -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

armcore/usart.o: /Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore/usart.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-float-math_test" -I"/Users/sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/Utilities/STM32F4-Discovery" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=hard -mfpu=fpv4-sp-d16 -g3 -Wall -c -fmessage-length=0 -fsingle-precision-constant -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


