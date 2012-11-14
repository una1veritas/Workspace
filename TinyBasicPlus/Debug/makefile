################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include libraries/Wire/utility/subdir.mk
-include libraries/Wire/subdir.mk
-include libraries/Stepper/subdir.mk
-include libraries/SoftwareSerial/subdir.mk
-include libraries/Servo/subdir.mk
-include libraries/SPI/subdir.mk
-include libraries/SD/utility/subdir.mk
-include libraries/SD/subdir.mk
-include libraries/LiquidCrystal/subdir.mk
-include libraries/Firmata/subdir.mk
-include libraries/Ethernet/utility/subdir.mk
-include libraries/Ethernet/subdir.mk
-include libraries/EEPROM/subdir.mk
-include cores/arduino/subdir.mk
-include cli/subdir.mk
-include bootloaders/stk500v2/subdir.mk
-include bootloaders/optiboot/subdir.mk
-include bootloaders/lilypad/src/subdir.mk
-include bootloaders/caterina/subdir.mk
-include bootloaders/bt/subdir.mk
-include bootloaders/atmega8/subdir.mk
-include bootloaders/atmega/subdir.mk
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
TinyBasicPlus.lss \

FLASH_IMAGE += \
TinyBasicPlus.hex \

SIZEDUMMY += \
sizedummy \


# All Target
all: TinyBasicPlus.elf secondary-outputs

# Tool invocations
TinyBasicPlus.elf: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: AVR C++ Linker'
	avr-g++ -Wl,-Map,TinyBasicPlus.map,--cref -mmcu=atmega328p -o "TinyBasicPlus.elf" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

TinyBasicPlus.lss: TinyBasicPlus.elf
	@echo 'Invoking: AVR Create Extended Listing'
	-avr-objdump -h -S TinyBasicPlus.elf  >"TinyBasicPlus.lss"
	@echo 'Finished building: $@'
	@echo ' '

TinyBasicPlus.hex: TinyBasicPlus.elf
	@echo 'Create Flash image (ihex format)'
	-avr-objcopy -R .eeprom -O ihex TinyBasicPlus.elf  "TinyBasicPlus.hex"
	@echo 'Finished building: $@'
	@echo ' '

sizedummy: TinyBasicPlus.elf
	@echo 'Invoking: Print Size'
	-avr-size --format=berkeley -t TinyBasicPlus.elf
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)$(C_DEPS)$(ELFS)$(FLASH_IMAGE)$(LSS)$(S_DEPS)$(CXX_DEPS)$(S_UPPER_DEPS)$(C++_DEPS)$(ASM_DEPS)$(CC_DEPS)$(CPP_DEPS)$(C_UPPER_DEPS)$(SIZEDUMMY) TinyBasicPlus.elf
	-@echo ' '

secondary-outputs: $(LSS) $(FLASH_IMAGE) $(SIZEDUMMY)

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
