################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../mycore/delay.c \
../mycore/gpio_digital.c \
../mycore/stm32f4xx_it.c 

OBJS += \
./mycore/delay.o \
./mycore/gpio_digital.o \
./mycore/stm32f4xx_it.o 

C_DEPS += \
./mycore/delay.d \
./mycore/gpio_digital.d \
./mycore/stm32f4xx_it.d 


# Each subdirectory must supply rules for building sources it contributes
mycore/%.o: ../mycore/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/mycore" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/GPIO" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/USART" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Include" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -g -Wall -c -fmessage-length=0 -ffreestanding -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


