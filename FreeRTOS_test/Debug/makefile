################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include freeRTOS711_All_Files/retrograde_v2/Debug/subdir.mk
-include freeRTOS711_All_Files/retrograde_v2/subdir.mk
-include freeRTOS711_All_Files/retrograde/subdir.mk
-include freeRTOS711_All_Files/microbridge/max3421e/subdir.mk
-include freeRTOS711_All_Files/microbridge/subdir.mk
-include freeRTOS711_All_Files/freeRTOS711/portable/subdir.mk
-include freeRTOS711_All_Files/freeRTOS711/lib_w5100/subdir.mk
-include freeRTOS711_All_Files/freeRTOS711/lib_spi/subdir.mk
-include freeRTOS711_All_Files/freeRTOS711/lib_servo/subdir.mk
-include freeRTOS711_All_Files/freeRTOS711/lib_serial/subdir.mk
-include freeRTOS711_All_Files/freeRTOS711/lib_rtc/subdir.mk
-include freeRTOS711_All_Files/freeRTOS711/lib_inet/subdir.mk
-include freeRTOS711_All_Files/freeRTOS711/lib_i2c/subdir.mk
-include freeRTOS711_All_Files/freeRTOS711/lib_hd44780/subdir.mk
-include freeRTOS711_All_Files/freeRTOS711/lib_fatf/subdir.mk
-include freeRTOS711_All_Files/freeRTOS711/lib_ext_ram/subdir.mk
-include freeRTOS711_All_Files/freeRTOS711/lib_digitalAnalog/subdir.mk
-include freeRTOS711_All_Files/freeRTOS711/lib_crc/subdir.mk
-include freeRTOS711_All_Files/freeRTOS711/MemMang/subdir.mk
-include freeRTOS711_All_Files/freeRTOS711/subdir.mk
-include freeRTOS711_All_Files/UnoBlink/subdir.mk
-include freeRTOS711_All_Files/PeggyVideoPong/subdir.mk
-include freeRTOS711_All_Files/PeggyVideo16/subdir.mk
-include freeRTOS711_All_Files/MegaW5100Test/subdir.mk
-include freeRTOS711_All_Files/MegaSDTest/subdir.mk
-include freeRTOS711_All_Files/MegaBlink/subdir.mk
-include freeRTOS711_All_Files/LCD6100_Driver/subdir.mk
-include freeRTOS711_All_Files/LCD6100_Demo/subdir.mk
-include freeRTOS711_All_Files/DogBot/subdir.mk
-include freeRTOS711_All_Files/Danger_ADB/subdir.mk
-include freeRTOS711_All_Files/ConwayLifePeggy/subdir.mk
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
FreeRTOS_test.lss \

SIZEDUMMY += \
sizedummy \


# All Target
all: FreeRTOS_test.elf secondary-outputs

# Tool invocations
FreeRTOS_test.elf: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: AVR C++ Linker'
	avr-g++ -Wl,-Map,FreeRTOS_test.map,--cref -mmcu=atmega16 -o "FreeRTOS_test.elf" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

FreeRTOS_test.lss: FreeRTOS_test.elf
	@echo 'Invoking: AVR Create Extended Listing'
	-avr-objdump -h -S FreeRTOS_test.elf  >"FreeRTOS_test.lss"
	@echo 'Finished building: $@'
	@echo ' '

sizedummy: FreeRTOS_test.elf
	@echo 'Invoking: Print Size'
	-avr-size --format=avr --mcu=atmega16 FreeRTOS_test.elf
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)$(C_DEPS)$(ELFS)$(LSS)$(S_DEPS)$(CXX_DEPS)$(S_UPPER_DEPS)$(C++_DEPS)$(ASM_DEPS)$(CC_DEPS)$(CPP_DEPS)$(C_UPPER_DEPS)$(SIZEDUMMY) FreeRTOS_test.elf
	-@echo ' '

secondary-outputs: $(LSS) $(SIZEDUMMY)

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
