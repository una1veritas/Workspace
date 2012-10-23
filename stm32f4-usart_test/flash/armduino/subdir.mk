################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Eclipse/Workspace/STM32Library/armduino/usart.cpp 

C_SRCS += \
/Users/sin/Documents/Eclipse/Workspace/STM32Library/armduino/gpio.c 

OBJS += \
./armduino/gpio.o \
./armduino/usart.o 

C_DEPS += \
./armduino/gpio.d 

CPP_DEPS += \
./armduino/usart.d 


# Each subdirectory must supply rules for building sources it contributes
armduino/gpio.o: /Users/sin/Documents/Eclipse/Workspace/STM32Library/armduino/gpio.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

armduino/usart.o: /Users/sin/Documents/Eclipse/Workspace/STM32Library/armduino/usart.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	arm-none-eabi-g++ -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian  -mfloat-abi=soft -g3 -Wall -c -fmessage-length=0 -fno-exceptions -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


