################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include avr/libraries/Wire/utility/subdir.mk
-include avr/libraries/Wire/subdir.mk
-include avr/libraries/WiFi/utility/subdir.mk
-include avr/libraries/WiFi/subdir.mk
-include avr/libraries/Stepper/subdir.mk
-include avr/libraries/Servo/subdir.mk
-include avr/libraries/SPI/subdir.mk
-include avr/libraries/Firmata/subdir.mk
-include avr/libraries/Ethernet/utility/subdir.mk
-include avr/libraries/Ethernet/subdir.mk
-include avr/libraries/EEPROM/subdir.mk
-include avr/cores/arduino/subdir.mk
-include FreeRTOS_AVR/utility/subdir.mk
-include FreeRTOS_AVR/subdir.mk
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
Arduino_FreeRTOS.lss \

FLASH_IMAGE += \
Arduino_FreeRTOS.hex \

SIZEDUMMY += \
sizedummy \


# All Target
all: Arduino_FreeRTOS.elf secondary-outputs

# Tool invocations
Arduino_FreeRTOS.elf: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: AVR C++ Linker'
	avr-g++ -Wl,-Map,Arduino_FreeRTOS.map,--cref -mmcu=atmega328p -o "Arduino_FreeRTOS.elf" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

Arduino_FreeRTOS.lss: Arduino_FreeRTOS.elf
	@echo 'Invoking: AVR Create Extended Listing'
	-avr-objdump -h -S Arduino_FreeRTOS.elf  >"Arduino_FreeRTOS.lss"
	@echo 'Finished building: $@'
	@echo ' '

Arduino_FreeRTOS.hex: Arduino_FreeRTOS.elf
	@echo 'Create Flash image (ihex format)'
	-avr-objcopy -R .eeprom -O ihex Arduino_FreeRTOS.elf  "Arduino_FreeRTOS.hex"
	@echo 'Finished building: $@'
	@echo ' '

sizedummy: Arduino_FreeRTOS.elf
	@echo 'Invoking: Print Size'
	-avr-size --format=berkeley -t Arduino_FreeRTOS.elf
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)$(C_DEPS)$(ELFS)$(FLASH_IMAGE)$(LSS)$(S_DEPS)$(CXX_DEPS)$(S_UPPER_DEPS)$(C++_DEPS)$(ASM_DEPS)$(CC_DEPS)$(CPP_DEPS)$(C_UPPER_DEPS)$(SIZEDUMMY) Arduino_FreeRTOS.elf
	-@echo ' '

secondary-outputs: $(LSS) $(FLASH_IMAGE) $(SIZEDUMMY)

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
