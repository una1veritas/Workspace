################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../STM32_USB_Driver/usb_dcd.c \
../STM32_USB_Driver/usb_dcd_int.c \
../STM32_USB_Driver/usbd_core.c \
../STM32_USB_Driver/usbd_hid_core.c \
../STM32_USB_Driver/usbd_ioreq.c \
../STM32_USB_Driver/usbd_req.c 

OBJS += \
./STM32_USB_Driver/usb_dcd.o \
./STM32_USB_Driver/usb_dcd_int.o \
./STM32_USB_Driver/usbd_core.o \
./STM32_USB_Driver/usbd_hid_core.o \
./STM32_USB_Driver/usbd_ioreq.o \
./STM32_USB_Driver/usbd_req.o 

C_DEPS += \
./STM32_USB_Driver/usb_dcd.d \
./STM32_USB_Driver/usb_dcd_int.d \
./STM32_USB_Driver/usbd_core.d \
./STM32_USB_Driver/usbd_hid_core.d \
./STM32_USB_Driver/usbd_ioreq.d \
./STM32_USB_Driver/usbd_req.d 


# Each subdirectory must supply rules for building sources it contributes
STM32_USB_Driver/%.o: ../STM32_USB_Driver/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_USB_OTG_FS=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32_USB_Device_Library/Class/hid/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32_USB_Device_Library/Core/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/Utilities/STM32F4-Discovery" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Demo/discovery_demo" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_library/stm32f4/STM32_USB_OTG_Driver/inc" -O2 -mcpu=cortex-m4 -mthumb -mlittle-endian -ffreestanding -g -Wall -c -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


