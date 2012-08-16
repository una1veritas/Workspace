################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../freeRTOS711_All_Files/freeRTOS711/lib_w5100/md5.c \
../freeRTOS711_All_Files/freeRTOS711/lib_w5100/socket.c \
../freeRTOS711_All_Files/freeRTOS711/lib_w5100/socket_util.c \
../freeRTOS711_All_Files/freeRTOS711/lib_w5100/w5100.c 

OBJS += \
./freeRTOS711_All_Files/freeRTOS711/lib_w5100/md5.o \
./freeRTOS711_All_Files/freeRTOS711/lib_w5100/socket.o \
./freeRTOS711_All_Files/freeRTOS711/lib_w5100/socket_util.o \
./freeRTOS711_All_Files/freeRTOS711/lib_w5100/w5100.o 

C_DEPS += \
./freeRTOS711_All_Files/freeRTOS711/lib_w5100/md5.d \
./freeRTOS711_All_Files/freeRTOS711/lib_w5100/socket.d \
./freeRTOS711_All_Files/freeRTOS711/lib_w5100/socket_util.d \
./freeRTOS711_All_Files/freeRTOS711/lib_w5100/w5100.d 


# Each subdirectory must supply rules for building sources it contributes
freeRTOS711_All_Files/freeRTOS711/lib_w5100/%.o: ../freeRTOS711_All_Files/freeRTOS711/lib_w5100/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -Wall -g2 -gstabs -O0 -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


