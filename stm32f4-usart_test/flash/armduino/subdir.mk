################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../armduino/usart.cpp 

C_SRCS += \
../armduino/gpio.c 

OBJS += \
./armduino/gpio.o \
./armduino/usart.o 

C_DEPS += \
./armduino/gpio.d 

CPP_DEPS += \
./armduino/usart.d 


# Each subdirectory must supply rules for building sources it contributes
armduino/%.o: ../armduino/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-usart_test" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-usart_test/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-usart_test/armduino" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g3 -Wall -c -fmessage-length=0 -fsingle-precision-constant -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

armduino/%.o: ../armduino/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	arm-none-eabi-g++ -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-usart_test" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-usart_test/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-usart_test/armduino" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian  -mfloat-abi=soft -g3 -Wall -c -fmessage-length=0 -fno-exceptions -fsingle-precision-constant -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


