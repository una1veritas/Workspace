################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Projects/discovery_demo/main.c \
../Projects/discovery_demo/selftest.c \
../Projects/discovery_demo/stm32f4xx_it.c \
../Projects/discovery_demo/system_stm32f4xx.c \
../Projects/discovery_demo/usb_bsp.c \
../Projects/discovery_demo/usb_core.c \
../Projects/discovery_demo/usbd_desc.c \
../Projects/discovery_demo/usbd_usr.c 

ASM_SRCS += \
../Projects/discovery_demo/startup_stm32f4xx.asm 

OBJS += \
./Projects/discovery_demo/main.o \
./Projects/discovery_demo/selftest.o \
./Projects/discovery_demo/startup_stm32f4xx.o \
./Projects/discovery_demo/stm32f4xx_it.o \
./Projects/discovery_demo/system_stm32f4xx.o \
./Projects/discovery_demo/usb_bsp.o \
./Projects/discovery_demo/usb_core.o \
./Projects/discovery_demo/usbd_desc.o \
./Projects/discovery_demo/usbd_usr.o 

C_DEPS += \
./Projects/discovery_demo/main.d \
./Projects/discovery_demo/selftest.d \
./Projects/discovery_demo/stm32f4xx_it.d \
./Projects/discovery_demo/system_stm32f4xx.d \
./Projects/discovery_demo/usb_bsp.d \
./Projects/discovery_demo/usb_core.d \
./Projects/discovery_demo/usbd_desc.d \
./Projects/discovery_demo/usbd_usr.d 


# Each subdirectory must supply rules for building sources it contributes
Projects/discovery_demo/%.o: ../Projects/discovery_demo/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_USB_OTG_FS=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/Projects/discovery_demo" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/stm32f4/STM32_USB_Device_Library/Core/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/stm32f4/STM32_USB_Device_Library/Class/hid/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/stm32f4/STM32_USB_OTG_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/stm32f4/Utilities/STM32F4-Discovery" -O2 -mcpu=cortex-m4 -mthumb -mlittle-endian -ffreestanding -g -Wall -c -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Projects/discovery_demo/%.o: ../Projects/discovery_demo/%.asm
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Assembler'
	arm-none-eabi-as -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/Projects/discovery_demo" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/stm32f4/Utilities/STM32F4-Discovery" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


