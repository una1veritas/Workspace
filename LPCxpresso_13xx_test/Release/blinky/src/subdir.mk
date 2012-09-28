################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../blinky/src/blinky_main.c \
../blinky/src/clkconfig.c \
../blinky/src/cr_startup_lpc13.c \
../blinky/src/gpio.c \
../blinky/src/timer32.c 

OBJS += \
./blinky/src/blinky_main.o \
./blinky/src/clkconfig.o \
./blinky/src/cr_startup_lpc13.o \
./blinky/src/gpio.o \
./blinky/src/timer32.o 

C_DEPS += \
./blinky/src/blinky_main.d \
./blinky/src/clkconfig.d \
./blinky/src/cr_startup_lpc13.d \
./blinky/src/gpio.d \
./blinky/src/timer32.d 


# Each subdirectory must supply rules for building sources it contributes
blinky/src/%.o: ../blinky/src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -I"/Users/sin/Documents/Eclipse/Workspace/LPC1xxx_library/CMSISv2p00_LPC13xx/inc" -O2 -c -mthumb -mcpu=cortex-m3 -mfix-cortex-m3-ldrd -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


