################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include src/Wire/utility/subdir.mk
-include src/Wire/subdir.mk
-include src/SPISRAM_Microchip/subdir.mk
-include src/SPI/subdir.mk
-include src/SD_SPI/utility/subdir.mk
-include src/SD_SPI/subdir.mk
-include src/RTC_DS3234/subdir.mk
-include src/RFID_PN532_I2C/subdir.mk
-include src/Monitor/subdir.mk
-include src/Ethernet_W5100/utility/subdir.mk
-include src/Ethernet_W5100/subdir.mk
-include src/EEPROM/subdir.mk
-include src/subdir.mk
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
EtherTerminal.lss \

FLASH_IMAGE += \
EtherTerminal.hex \

EEPROM_IMAGE += \
EtherTerminal.eep \

SIZEDUMMY += \
sizedummy \


# All Target
all: EtherTerminal.elf secondary-outputs

# Tool invocations
EtherTerminal.elf: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: AVR C++ Linker'
	avr-g++ -Wl,-Map,EtherTerminal.map,--cref -mmcu=atmega16 -o "EtherTerminal.elf" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

EtherTerminal.lss: EtherTerminal.elf
	@echo 'Invoking: AVR Create Extended Listing'
	-avr-objdump -h -S EtherTerminal.elf  >"EtherTerminal.lss"
	@echo 'Finished building: $@'
	@echo ' '

EtherTerminal.hex: EtherTerminal.elf
	@echo 'Create Flash image (ihex format)'
	-avr-objcopy -R .eeprom -O ihex EtherTerminal.elf  "EtherTerminal.hex"
	@echo 'Finished building: $@'
	@echo ' '

EtherTerminal.eep: EtherTerminal.elf
	@echo 'Create eeprom image (ihex format)'
	-avr-objcopy -j .eeprom --no-change-warnings --change-section-lma .eeprom=0 -O ihex EtherTerminal.elf  "EtherTerminal.eep"
	@echo 'Finished building: $@'
	@echo ' '

sizedummy: EtherTerminal.elf
	@echo 'Invoking: Print Size'
	-avr-size --format=avr --mcu=atmega16 EtherTerminal.elf
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)$(C_DEPS)$(EEPROM_IMAGE)$(ELFS)$(FLASH_IMAGE)$(LSS)$(S_DEPS)$(CXX_DEPS)$(S_UPPER_DEPS)$(C++_DEPS)$(ASM_DEPS)$(CC_DEPS)$(CPP_DEPS)$(C_UPPER_DEPS)$(SIZEDUMMY) EtherTerminal.elf
	-@echo ' '

secondary-outputs: $(LSS) $(FLASH_IMAGE) $(EEPROM_IMAGE) $(SIZEDUMMY)

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
