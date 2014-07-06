################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include AVR/xsvf/subdir.mk
-include AVR/SdFat/subdir.mk
-include AVR/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(ASM_DEPS)),)
-include $(ASM_DEPS)
endif
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(S_DEPS)),)
-include $(S_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
ifneq ($(strip $(S_UPPER_DEPS)),)
-include $(S_UPPER_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 
LSS += \
mac-floppy-emu-1.0.lss \

FLASH_IMAGE += \
mac-floppy-emu-1.0.hex \

SIZEDUMMY += \
sizedummy \


# All Target
all: mac-floppy-emu-1.0.elf secondary-outputs

# Tool invocations
mac-floppy-emu-1.0.elf: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: AVR C++ Linker'
	avr-g++ -Wl,-Map,mac-floppy-emu-1.0.map,--cref -mrelax -Wl,--gc-sections -mmcu=atmega2560 -o "mac-floppy-emu-1.0.elf" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

mac-floppy-emu-1.0.lss: mac-floppy-emu-1.0.elf
	@echo 'Invoking: AVR Create Extended Listing'
	-avr-objdump -h -S mac-floppy-emu-1.0.elf  >"mac-floppy-emu-1.0.lss"
	@echo 'Finished building: $@'
	@echo ' '

mac-floppy-emu-1.0.hex: mac-floppy-emu-1.0.elf
	@echo 'Create Flash image (ihex format)'
	-avr-objcopy -R .eeprom -R .fuse -R .lock -R .signature -O ihex mac-floppy-emu-1.0.elf  "mac-floppy-emu-1.0.hex"
	@echo 'Finished building: $@'
	@echo ' '

sizedummy: mac-floppy-emu-1.0.elf
	@echo 'Invoking: Print Size'
	-avr-size --format=berkeley -t mac-floppy-emu-1.0.elf
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)$(C_DEPS)$(FLASH_IMAGE)$(ELFS)$(LSS)$(S_DEPS)$(CXX_DEPS)$(S_UPPER_DEPS)$(C++_DEPS)$(ASM_DEPS)$(CC_DEPS)$(CPP_DEPS)$(C_UPPER_DEPS)$(SIZEDUMMY) mac-floppy-emu-1.0.elf
	-@echo ' '

secondary-outputs: $(LSS) $(FLASH_IMAGE) $(SIZEDUMMY)

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
