################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../cmsis/device/system_stm32f10x.c 

OBJS += \
./cmsis/device/system_stm32f10x.o 

C_DEPS += \
./cmsis/device/system_stm32f10x.d 


# Each subdirectory must supply rules for building sources it contributes
cmsis/device/%.o: ../cmsis/device/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -I"/Users/sin/Documents/Eclipse/Workspace/yagarto-test/cmsis/core" -I"/Users/sin/Documents/Eclipse/Workspace/yagarto-test/cmsis/device" -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


