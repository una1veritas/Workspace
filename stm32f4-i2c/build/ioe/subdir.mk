################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../ioe/fonts.c \
../ioe/main.c \
../ioe/stm324xg_eval.c \
../ioe/stm324xg_eval_ioe.c \
../ioe/stm324xg_eval_lcd.c 

OBJS += \
./ioe/fonts.o \
./ioe/main.o \
./ioe/stm324xg_eval.o \
./ioe/stm324xg_eval_ioe.o \
./ioe/stm324xg_eval_lcd.o 

C_DEPS += \
./ioe/fonts.d \
./ioe/main.d \
./ioe/stm324xg_eval.d \
./ioe/stm324xg_eval_ioe.d \
./ioe/stm324xg_eval_lcd.d 


# Each subdirectory must supply rules for building sources it contributes
ioe/%.o: ../ioe/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-i2c" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/STM32F4xx_StdPeriph_Driver/inc" -O0 -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


