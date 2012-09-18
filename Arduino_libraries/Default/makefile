################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -f

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
libArduino_libraries.lss \

SIZEDUMMY += \
sizedummy \


# All Target
all: libArduino_libraries.a secondary-outputs

dependents:
	-cd /Users/sin/Documents/Eclipse/Workspace/Arduino_cores/Default && $(MAKE) clean all

# Tool invocations
libArduino_libraries.a: $(OBJS) /Users/sin/Documents/Eclipse/Workspace/Arduino_cores/Default/libArduino_cores.a
	@echo 'Building target: $@'
	@echo 'Invoking: AVR Archiver'
	avr-ar -r  "libArduino_libraries.a" $(OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

libArduino_libraries.lss: libArduino_libraries.a
	@echo 'Invoking: AVR Create Extended Listing'
	-avr-objdump -h -S libArduino_libraries.a  >"libArduino_libraries.lss"
	@echo 'Finished building: $@'
	@echo ' '

sizedummy: libArduino_libraries.a
	@echo 'Invoking: Print Size'
	-avr-size --format=berkeley -t libArduino_libraries.a
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)$(C_DEPS)$(LSS)$(S_DEPS)$(CXX_DEPS)$(S_UPPER_DEPS)$(C++_DEPS)$(ASM_DEPS)$(CC_DEPS)$(ARCHIVES)$(CPP_DEPS)$(C_UPPER_DEPS)$(SIZEDUMMY) libArduino_libraries.a
	-@echo ' '

secondary-outputs: $(LSS) $(SIZEDUMMY)

.PHONY: all clean dependents
.SECONDARY:
/Users/sin/Documents/Eclipse/Workspace/Arduino_cores/Default/libArduino_cores.a:

-include ../makefile.targets
