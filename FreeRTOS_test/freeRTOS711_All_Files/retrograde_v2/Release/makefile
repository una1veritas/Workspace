################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(ASM_DEPS)),)
-include $(ASM_DEPS)
endif
ifneq ($(strip $(S_DEPS)),)
-include $(S_DEPS)
endif
ifneq ($(strip $(S_UPPER_DEPS)),)
-include $(S_UPPER_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 
LSS += \
retrograde_v2.lss \

FLASH_IMAGE += \
retrograde_v2.hex \

EEPROM_IMAGE += \
retrograde_v2.eep \

SIZEDUMMY += \
sizedummy \


# All Target
all: retrograde_v2.elf secondary-outputs

# Tool invocations
retrograde_v2.elf: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: AVR C Linker'
	avr-gcc -Wl,-Map,retrograde_v2.map -Wl,-gc-sections -mmcu=atmega328p -o "retrograde_v2.elf" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

retrograde_v2.lss: retrograde_v2.elf
	@echo 'Invoking: AVR Create Extended Listing'
	-avr-objdump -h -S retrograde_v2.elf  >"retrograde_v2.lss"
	@echo 'Finished building: $@'
	@echo ' '

retrograde_v2.hex: retrograde_v2.elf
	@echo 'Create Flash image (ihex format)'
	-avr-objcopy -R .eeprom -O ihex retrograde_v2.elf  "retrograde_v2.hex"
	@echo 'Finished building: $@'
	@echo ' '

retrograde_v2.eep: retrograde_v2.elf
	@echo 'Create eeprom image (ihex format)'
	-avr-objcopy -j .eeprom --no-change-warnings --change-section-lma .eeprom=0 -O ihex retrograde_v2.elf  "retrograde_v2.eep"
	@echo 'Finished building: $@'
	@echo ' '

sizedummy: retrograde_v2.elf
	@echo 'Invoking: Print Size'
	-avr-size --format=avr --mcu=atmega328p retrograde_v2.elf
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)$(C_DEPS)$(ASM_DEPS)$(EEPROM_IMAGE)$(FLASH_IMAGE)$(ELFS)$(LSS)$(S_DEPS)$(SIZEDUMMY)$(S_UPPER_DEPS) retrograde_v2.elf
	-@echo ' '

secondary-outputs: $(LSS) $(FLASH_IMAGE) $(EEPROM_IMAGE) $(SIZEDUMMY)

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
