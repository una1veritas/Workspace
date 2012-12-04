################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/src/misc.c \
C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_gpio.c \
C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_i2c.c \
C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_rcc.c \
C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_spi.c \
C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_tim.c \
C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_usart.c 

OBJS += \
./STM32F4xx_StdPeriph_Driver/misc.o \
./STM32F4xx_StdPeriph_Driver/stm32f4xx_gpio.o \
./STM32F4xx_StdPeriph_Driver/stm32f4xx_i2c.o \
./STM32F4xx_StdPeriph_Driver/stm32f4xx_rcc.o \
./STM32F4xx_StdPeriph_Driver/stm32f4xx_spi.o \
./STM32F4xx_StdPeriph_Driver/stm32f4xx_tim.o \
./STM32F4xx_StdPeriph_Driver/stm32f4xx_usart.o 

C_DEPS += \
./STM32F4xx_StdPeriph_Driver/misc.d \
./STM32F4xx_StdPeriph_Driver/stm32f4xx_gpio.d \
./STM32F4xx_StdPeriph_Driver/stm32f4xx_i2c.d \
./STM32F4xx_StdPeriph_Driver/stm32f4xx_rcc.d \
./STM32F4xx_StdPeriph_Driver/stm32f4xx_spi.d \
./STM32F4xx_StdPeriph_Driver/stm32f4xx_tim.d \
./STM32F4xx_StdPeriph_Driver/stm32f4xx_usart.d 


# Each subdirectory must supply rules for building sources it contributes
STM32F4xx_StdPeriph_Driver/misc.o: C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/src/misc.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32f4-stdperiph" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\armcore" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\CMSIS\Include" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\STM32F4xx_StdPeriph_Driver\inc" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\CMSIS\STM32F4xx" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

STM32F4xx_StdPeriph_Driver/stm32f4xx_gpio.o: C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_gpio.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32f4-stdperiph" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\armcore" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\CMSIS\Include" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\STM32F4xx_StdPeriph_Driver\inc" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\CMSIS\STM32F4xx" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

STM32F4xx_StdPeriph_Driver/stm32f4xx_i2c.o: C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_i2c.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32f4-stdperiph" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\armcore" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\CMSIS\Include" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\STM32F4xx_StdPeriph_Driver\inc" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\CMSIS\STM32F4xx" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

STM32F4xx_StdPeriph_Driver/stm32f4xx_rcc.o: C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_rcc.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32f4-stdperiph" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\armcore" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\CMSIS\Include" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\STM32F4xx_StdPeriph_Driver\inc" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\CMSIS\STM32F4xx" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

STM32F4xx_StdPeriph_Driver/stm32f4xx_spi.o: C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_spi.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32f4-stdperiph" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\armcore" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\CMSIS\Include" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\STM32F4xx_StdPeriph_Driver\inc" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\CMSIS\STM32F4xx" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

STM32F4xx_StdPeriph_Driver/stm32f4xx_tim.o: C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_tim.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32f4-stdperiph" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\armcore" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\CMSIS\Include" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\STM32F4xx_StdPeriph_Driver\inc" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\CMSIS\STM32F4xx" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

STM32F4xx_StdPeriph_Driver/stm32f4xx_usart.o: C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/STM32F4xx_StdPeriph_Driver/src/stm32f4xx_usart.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32f4-stdperiph" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\armcore" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\CMSIS\Include" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\STM32F4xx_StdPeriph_Driver\inc" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\CMSIS\STM32F4xx" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


