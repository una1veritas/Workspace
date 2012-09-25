################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../discovery_demo/main.c \
../discovery_demo/selftest.c \
../discovery_demo/stm32f4xx_it.c \
../discovery_demo/system_stm32f4xx.c \
../discovery_demo/usb_bsp.c \
../discovery_demo/usb_core.c \
../discovery_demo/usbd_desc.c \
../discovery_demo/usbd_usr.c 

ASM_SRCS += \
../discovery_demo/startup_stm32f4xx.asm 

OBJS += \
./discovery_demo/main.o \
./discovery_demo/selftest.o \
./discovery_demo/startup_stm32f4xx.o \
./discovery_demo/stm32f4xx_it.o \
./discovery_demo/system_stm32f4xx.o \
./discovery_demo/usb_bsp.o \
./discovery_demo/usb_core.o \
./discovery_demo/usbd_desc.o \
./discovery_demo/usbd_usr.o 

C_DEPS += \
./discovery_demo/main.d \
./discovery_demo/selftest.d \
./discovery_demo/stm32f4xx_it.d \
./discovery_demo/system_stm32f4xx.d \
./discovery_demo/usb_bsp.d \
./discovery_demo/usb_core.d \
./discovery_demo/usbd_desc.d \
./discovery_demo/usbd_usr.d 


# Each subdirectory must supply rules for building sources it contributes
discovery_demo/%.o: ../discovery_demo/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_USB_OTG_FS=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32_USB_Device_Library/Class/hid/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32_USB_Device_Library/Core/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/Utilities/STM32F4-Discovery" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Demo/discovery_demo" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32_USB_OTG_Driver/inc" -O2 -mcpu=cortex-m4 -mthumb -mlittle-endian -ffreestanding -g -Wall -c -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

discovery_demo/%.o: ../discovery_demo/%.asm
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Assembler'
	arm-none-eabi-as -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32_USB_Device_Library/Class/hid/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32_USB_Device_Library/Core/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/Utilities/STM32F4-Discovery" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Demo/discovery_demo" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32_USB_OTG_Driver/inc" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


