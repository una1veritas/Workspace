################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../CMSISv2p00_LPC17xx/src/core_cm3.c \
../CMSISv2p00_LPC17xx/src/system_LPC17xx.c 

OBJS += \
./CMSISv2p00_LPC17xx/src/core_cm3.o \
./CMSISv2p00_LPC17xx/src/system_LPC17xx.o 

C_DEPS += \
./CMSISv2p00_LPC17xx/src/core_cm3.d \
./CMSISv2p00_LPC17xx/src/system_LPC17xx.d 


# Each subdirectory must supply rules for building sources it contributes
CMSISv2p00_LPC17xx/src/%.o: ../CMSISv2p00_LPC17xx/src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -I"/Users/sin/Documents/Eclipse/Workspace/LPC1xxx_library/CMSISv2p00_LPC17xx/inc" -O2 -c -mthumb -mcpu=cortex-m3 -mfix-cortex-m3-ldrd -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


