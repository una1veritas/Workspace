################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include src/subdir.mk
-include libraries/Wire/utility/subdir.mk
-include libraries/Wire/subdir.mk
-include libraries/Stepper/subdir.mk
-include libraries/SoftwareSerial/subdir.mk
-include libraries/Servo/subdir.mk
-include libraries/SPI/subdir.mk
-include libraries/LiquidCrystal/subdir.mk
-include libraries/Firmata/subdir.mk
-include libraries/EEPROM/subdir.mk
-include home_libraries/SPISRAM_Microchip/subdir.mk
-include home_libraries/SD_SPI/utility/subdir.mk
-include home_libraries/SD_SPI/subdir.mk
-include home_libraries/RTC_DS3234/subdir.mk
-include home_libraries/RFID_PN532_I2C/subdir.mk
-include home_libraries/Monitor/subdir.mk
-include home_libraries/Ethernet_W5100/utility/subdir.mk
-include home_libraries/Ethernet_W5100/subdir.mk
-include home_libraries/Des/subdir.mk
-include home_libraries/DataFlash/subdir.mk
-include home_libraries/AES128/subdir.mk
-include cores/arduino/subdir.mk
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
Arduino_eclipse_build.lss \

FLASH_IMAGE += \
Arduino_eclipse_build.hex \

EEPROM_IMAGE += \
Arduino_eclipse_build.eep \

SIZEDUMMY += \
sizedummy \


# All Target
all: Arduino_eclipse_build.elf secondary-outputs

# Tool invocations
Arduino_eclipse_build.elf: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: AVR C++ Linker'
	avr-g++ -Wl,-Map,Arduino_eclipse_build.map,--cref -mmcu=atmega328p -o "Arduino_eclipse_build.elf" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

Arduino_eclipse_build.lss: Arduino_eclipse_build.elf
	@echo 'Invoking: AVR Create Extended Listing'
	-avr-objdump -h -S Arduino_eclipse_build.elf  >"Arduino_eclipse_build.lss"
	@echo 'Finished building: $@'
	@echo ' '

Arduino_eclipse_build.hex: Arduino_eclipse_build.elf
	@echo 'Create Flash image (ihex format)'
	-avr-objcopy -R .eeprom -O ihex Arduino_eclipse_build.elf  "Arduino_eclipse_build.hex"
	@echo 'Finished building: $@'
	@echo ' '

Arduino_eclipse_build.eep: Arduino_eclipse_build.elf
	@echo 'Create eeprom image (ihex format)'
	-avr-objcopy -j .eeprom --no-change-warnings --change-section-lma .eeprom=0 -O ihex Arduino_eclipse_build.elf  "Arduino_eclipse_build.eep"
	@echo 'Finished building: $@'
	@echo ' '

sizedummy: Arduino_eclipse_build.elf
	@echo 'Invoking: Print Size'
	-avr-size --format=avr --mcu=atmega328p Arduino_eclipse_build.elf
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)$(C_DEPS)$(EEPROM_IMAGE)$(ELFS)$(FLASH_IMAGE)$(LSS)$(S_DEPS)$(CXX_DEPS)$(S_UPPER_DEPS)$(C++_DEPS)$(ASM_DEPS)$(CC_DEPS)$(CPP_DEPS)$(C_UPPER_DEPS)$(SIZEDUMMY) Arduino_eclipse_build.elf
	-@echo ' '

secondary-outputs: $(LSS) $(FLASH_IMAGE) $(EEPROM_IMAGE) $(SIZEDUMMY)

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
