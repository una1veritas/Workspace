################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include src/IDCardReaderServer/subdir.mk
-include libraries/Wire/utility/subdir.mk
-include libraries/Wire/subdir.mk
-include libraries/TextMonitor/subdir.mk
-include libraries/SPI/subdir.mk
-include libraries/RTC_DS3234/subdir.mk
-include libraries/RFID_PN532_I2C/subdir.mk
-include libraries/Ethernet/utility/subdir.mk
-include libraries/Ethernet/subdir.mk
-include libraries/EEPROM/subdir.mk
-include libraries/ColorLCDShield/subdir.mk
-include arduino/subdir.mk
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
arduinogcc.lss \

FLASH_IMAGE += \
arduinogcc.hex \

SIZEDUMMY += \
sizedummy \


# All Target
all: arduinogcc.elf secondary-outputs

# Tool invocations
arduinogcc.elf: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: AVR C++ Linker'
	avr-g++ -Wl,-Map,arduinogcc.map,--gc-sections,--cref -s -mmcu=atmega328p -o "arduinogcc.elf" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

arduinogcc.lss: arduinogcc.elf
	@echo 'Invoking: AVR Create Extended Listing'
	-avr-objdump -h -S arduinogcc.elf  >"arduinogcc.lss"
	@echo 'Finished building: $@'
	@echo ' '

arduinogcc.hex: arduinogcc.elf
	@echo 'Create Flash image (ihex format)'
	-avr-objcopy -R .eeprom -O ihex arduinogcc.elf  "arduinogcc.hex"
	@echo 'Finished building: $@'
	@echo ' '

sizedummy: arduinogcc.elf
	@echo 'Invoking: Print Size'
	-avr-size --format=berkeley -t arduinogcc.elf
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)$(C_DEPS)$(ELFS)$(FLASH_IMAGE)$(LSS)$(S_DEPS)$(CXX_DEPS)$(S_UPPER_DEPS)$(C++_DEPS)$(ASM_DEPS)$(CC_DEPS)$(CPP_DEPS)$(C_UPPER_DEPS)$(SIZEDUMMY) arduinogcc.elf
	-@echo ' '

secondary-outputs: $(LSS) $(FLASH_IMAGE) $(SIZEDUMMY)

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
