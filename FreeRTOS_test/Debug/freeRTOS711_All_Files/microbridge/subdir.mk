################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../freeRTOS711_All_Files/microbridge/adb.c \
../freeRTOS711_All_Files/microbridge/avr.c \
../freeRTOS711_All_Files/microbridge/example.c \
../freeRTOS711_All_Files/microbridge/logcat.c \
../freeRTOS711_All_Files/microbridge/main.c \
../freeRTOS711_All_Files/microbridge/usb.c 

OBJS += \
./freeRTOS711_All_Files/microbridge/adb.o \
./freeRTOS711_All_Files/microbridge/avr.o \
./freeRTOS711_All_Files/microbridge/example.o \
./freeRTOS711_All_Files/microbridge/logcat.o \
./freeRTOS711_All_Files/microbridge/main.o \
./freeRTOS711_All_Files/microbridge/usb.o 

C_DEPS += \
./freeRTOS711_All_Files/microbridge/adb.d \
./freeRTOS711_All_Files/microbridge/avr.d \
./freeRTOS711_All_Files/microbridge/example.d \
./freeRTOS711_All_Files/microbridge/logcat.d \
./freeRTOS711_All_Files/microbridge/main.d \
./freeRTOS711_All_Files/microbridge/usb.d 


# Each subdirectory must supply rules for building sources it contributes
freeRTOS711_All_Files/microbridge/%.o: ../freeRTOS711_All_Files/microbridge/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -Wall -g2 -gstabs -O0 -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


