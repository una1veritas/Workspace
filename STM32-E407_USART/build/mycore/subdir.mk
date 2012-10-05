################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../mycore/delay.c \
../mycore/gpio_digital.c \
../mycore/newlib_stubs.c 

OBJS += \
./mycore/delay.o \
./mycore/gpio_digital.o \
./mycore/newlib_stubs.o 

C_DEPS += \
./mycore/delay.d \
./mycore/gpio_digital.d \
./mycore/newlib_stubs.d 


# Each subdirectory must supply rules for building sources it contributes
mycore/%.o: ../mycore/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -DHSE_VALUE=25000000 -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_USART/mycore" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_USART/USART" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/CMSIS/Include" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -mfpu=fpv4-sp-d16 -g -Wall -c -fmessage-length=0 -ffreestanding -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


