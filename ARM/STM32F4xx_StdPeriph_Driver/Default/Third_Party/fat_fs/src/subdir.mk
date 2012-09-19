################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Third_Party/fat_fs/src/diskio.c \
../Third_Party/fat_fs/src/fattime.c \
../Third_Party/fat_fs/src/ff.c 

OBJS += \
./Third_Party/fat_fs/src/diskio.o \
./Third_Party/fat_fs/src/fattime.o \
./Third_Party/fat_fs/src/ff.o 

C_DEPS += \
./Third_Party/fat_fs/src/diskio.d \
./Third_Party/fat_fs/src/fattime.d \
./Third_Party/fat_fs/src/ff.d 


# Each subdirectory must supply rules for building sources it contributes
Third_Party/fat_fs/src/%.o: ../Third_Party/fat_fs/src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -I"/Users/sin/Documents/Eclipse/Workspace/STM32/STM32F4xx_library/inc/core_support" -I"/Users/sin/Documents/Eclipse/Workspace/STM32/STM32F4xx_library/inc/device_support" -I"/Users/sin/Documents/Eclipse/Workspace/STM32/STM32F4xx_library/inc" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


