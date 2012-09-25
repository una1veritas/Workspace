################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../STM32_USB_Device_Library/Class/dfu/src/usbd_dfu_core.c \
../STM32_USB_Device_Library/Class/dfu/src/usbd_dfu_mal.c \
../STM32_USB_Device_Library/Class/dfu/src/usbd_flash_if.c \
../STM32_USB_Device_Library/Class/dfu/src/usbd_mem_if_template.c \
../STM32_USB_Device_Library/Class/dfu/src/usbd_otp_if.c 

OBJS += \
./STM32_USB_Device_Library/Class/dfu/src/usbd_dfu_core.o \
./STM32_USB_Device_Library/Class/dfu/src/usbd_dfu_mal.o \
./STM32_USB_Device_Library/Class/dfu/src/usbd_flash_if.o \
./STM32_USB_Device_Library/Class/dfu/src/usbd_mem_if_template.o \
./STM32_USB_Device_Library/Class/dfu/src/usbd_otp_if.o 

C_DEPS += \
./STM32_USB_Device_Library/Class/dfu/src/usbd_dfu_core.d \
./STM32_USB_Device_Library/Class/dfu/src/usbd_dfu_mal.d \
./STM32_USB_Device_Library/Class/dfu/src/usbd_flash_if.d \
./STM32_USB_Device_Library/Class/dfu/src/usbd_mem_if_template.d \
./STM32_USB_Device_Library/Class/dfu/src/usbd_otp_if.d 


# Each subdirectory must supply rules for building sources it contributes
STM32_USB_Device_Library/Class/dfu/src/%.o: ../STM32_USB_Device_Library/Class/dfu/src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_USB_OTG_FS=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32_USB_Device_Library/Class/hid/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32_USB_Device_Library/Core/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/Utilities/STM32F4-Discovery" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Demo/discovery_demo" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32_USB_OTG_Driver/inc" -O2 -mcpu=cortex-m4 -mthumb -mlittle-endian -ffreestanding -g -Wall -c -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


