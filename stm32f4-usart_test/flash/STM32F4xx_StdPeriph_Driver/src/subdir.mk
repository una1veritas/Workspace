################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../STM32F4xx_StdPeriph_Driver/src/stm32f4xx_gpio.c \
../STM32F4xx_StdPeriph_Driver/src/stm32f4xx_rcc.c \
../STM32F4xx_StdPeriph_Driver/src/stm32f4xx_usart.c 

OBJS += \
./STM32F4xx_StdPeriph_Driver/src/stm32f4xx_gpio.o \
./STM32F4xx_StdPeriph_Driver/src/stm32f4xx_rcc.o \
./STM32F4xx_StdPeriph_Driver/src/stm32f4xx_usart.o 

C_DEPS += \
./STM32F4xx_StdPeriph_Driver/src/stm32f4xx_gpio.d \
./STM32F4xx_StdPeriph_Driver/src/stm32f4xx_rcc.d \
./STM32F4xx_StdPeriph_Driver/src/stm32f4xx_usart.d 


# Each subdirectory must supply rules for building sources it contributes
STM32F4xx_StdPeriph_Driver/src/%.o: ../STM32F4xx_StdPeriph_Driver/src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-usart_test" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-usart_test/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-usart_test/armduino" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g3 -Wall -c -fmessage-length=0 -fsingle-precision-constant -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


