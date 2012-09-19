################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../STM32_USB_OTG_Driver/src/usb_bsp_template.c \
../STM32_USB_OTG_Driver/src/usb_core.c \
../STM32_USB_OTG_Driver/src/usb_dcd.c \
../STM32_USB_OTG_Driver/src/usb_dcd_int.c \
../STM32_USB_OTG_Driver/src/usb_hcd.c \
../STM32_USB_OTG_Driver/src/usb_hcd_int.c \
../STM32_USB_OTG_Driver/src/usb_otg.c 

OBJS += \
./STM32_USB_OTG_Driver/src/usb_bsp_template.o \
./STM32_USB_OTG_Driver/src/usb_core.o \
./STM32_USB_OTG_Driver/src/usb_dcd.o \
./STM32_USB_OTG_Driver/src/usb_dcd_int.o \
./STM32_USB_OTG_Driver/src/usb_hcd.o \
./STM32_USB_OTG_Driver/src/usb_hcd_int.o \
./STM32_USB_OTG_Driver/src/usb_otg.o 

C_DEPS += \
./STM32_USB_OTG_Driver/src/usb_bsp_template.d \
./STM32_USB_OTG_Driver/src/usb_core.d \
./STM32_USB_OTG_Driver/src/usb_dcd.d \
./STM32_USB_OTG_Driver/src/usb_dcd_int.d \
./STM32_USB_OTG_Driver/src/usb_hcd.d \
./STM32_USB_OTG_Driver/src/usb_hcd_int.d \
./STM32_USB_OTG_Driver/src/usb_otg.d 


# Each subdirectory must supply rules for building sources it contributes
STM32_USB_OTG_Driver/src/%.o: ../STM32_USB_OTG_Driver/src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/STM32_USB_Device_Library/Class/hid/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/STM32_USB_OTG_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/STM32F4-Discovery" -O0 -mcpu=cortex-m4 -mthumb -mlittle-endian -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


