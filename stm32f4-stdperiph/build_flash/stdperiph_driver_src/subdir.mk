################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/src/misc.c \
/Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_gpio.c \
/Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_i2c.c \
/Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_rcc.c \
/Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_spi.c \
/Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_tim.c \
/Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_usart.c 

OBJS += \
./stdperiph_driver_src/misc.o \
./stdperiph_driver_src/stm32f4xx_gpio.o \
./stdperiph_driver_src/stm32f4xx_i2c.o \
./stdperiph_driver_src/stm32f4xx_rcc.o \
./stdperiph_driver_src/stm32f4xx_spi.o \
./stdperiph_driver_src/stm32f4xx_tim.o \
./stdperiph_driver_src/stm32f4xx_usart.o 

C_DEPS += \
./stdperiph_driver_src/misc.d \
./stdperiph_driver_src/stm32f4xx_gpio.d \
./stdperiph_driver_src/stm32f4xx_i2c.d \
./stdperiph_driver_src/stm32f4xx_rcc.d \
./stdperiph_driver_src/stm32f4xx_spi.d \
./stdperiph_driver_src/stm32f4xx_tim.d \
./stdperiph_driver_src/stm32f4xx_usart.d 


# Each subdirectory must supply rules for building sources it contributes
stdperiph_driver_src/misc.o: /Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/src/misc.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/armcore" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-stdperiph" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/armlib" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

stdperiph_driver_src/stm32f4xx_gpio.o: /Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_gpio.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/armcore" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-stdperiph" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/armlib" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

stdperiph_driver_src/stm32f4xx_i2c.o: /Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_i2c.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/armcore" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-stdperiph" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/armlib" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

stdperiph_driver_src/stm32f4xx_rcc.o: /Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_rcc.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/armcore" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-stdperiph" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/armlib" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

stdperiph_driver_src/stm32f4xx_spi.o: /Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_spi.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/armcore" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-stdperiph" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/armlib" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

stdperiph_driver_src/stm32f4xx_tim.o: /Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_tim.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/armcore" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-stdperiph" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/armlib" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

stdperiph_driver_src/stm32f4xx_usart.o: /Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_usart.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/armcore" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-stdperiph" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/stm32library/armlib" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


