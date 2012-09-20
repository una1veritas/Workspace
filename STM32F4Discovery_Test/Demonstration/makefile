################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include stm32f4/Utilities/STM32F4-Discovery/subdir.mk
-include stm32f4/STM32_USB_OTG_Driver/src/subdir.mk
-include stm32f4/STM32_USB_Device_Library/Core/src/subdir.mk
-include stm32f4/STM32_USB_Device_Library/Class/hid/src/subdir.mk
-include Projects/discovery_demo/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: STM32F4Discovery_Test.elf

# Tool invocations
STM32F4Discovery_Test.elf: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: Cross G++ Linker'
	arm-none-eabi-g++ -nostdlib -L"/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/stm32f4/STM32F4xx_StdPeriph_Driver" -Wl,-T,/Users/sin/Documents/Eclipse/Workspace/STM32F4Discovery_Test/Projects/discovery_demo/stm32_flash.ld -o "STM32F4Discovery_Test.elf" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '
	$(MAKE) --no-print-directory post-build

# Other Targets
clean:
	-$(RM) $(C++_DEPS)$(OBJS)$(C_DEPS)$(CC_DEPS)$(CPP_DEPS)$(EXECUTABLES)$(CXX_DEPS)$(C_UPPER_DEPS) STM32F4Discovery_Test.elf
	-@echo ' '

post-build:
	-@echo 'Making binary image.'
	-arm-none-eabi-objcopy -O binary STM32F4Discovery_Test.elf STM32F4Discovery_Test.bin; arm-none-eabi-objcopy -O ihex STM32F4Discovery_Test.elf STM32F4Discovery_Test.hex
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY: post-build

-include ../makefile.targets
