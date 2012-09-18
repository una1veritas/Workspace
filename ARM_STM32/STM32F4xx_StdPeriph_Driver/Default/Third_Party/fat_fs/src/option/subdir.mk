################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Third_Party/fat_fs/src/option/ccsbcs.c \
../Third_Party/fat_fs/src/option/syncobj.c 

OBJS += \
./Third_Party/fat_fs/src/option/ccsbcs.o \
./Third_Party/fat_fs/src/option/syncobj.o 

C_DEPS += \
./Third_Party/fat_fs/src/option/ccsbcs.d \
./Third_Party/fat_fs/src/option/syncobj.d 


# Each subdirectory must supply rules for building sources it contributes
Third_Party/fat_fs/src/option/%.o: ../Third_Party/fat_fs/src/option/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -I"/Users/sin/Documents/Eclipse/Workspace/STM32/STM32F4xx_library/inc/core_support" -I"/Users/sin/Documents/Eclipse/Workspace/STM32/STM32F4xx_library/inc/device_support" -I"/Users/sin/Documents/Eclipse/Workspace/STM32/STM32F4xx_library/inc" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


