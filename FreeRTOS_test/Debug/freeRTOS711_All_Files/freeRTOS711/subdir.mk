################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../freeRTOS711_All_Files/freeRTOS711/croutine.c \
../freeRTOS711_All_Files/freeRTOS711/list.c \
../freeRTOS711_All_Files/freeRTOS711/queue.c \
../freeRTOS711_All_Files/freeRTOS711/tasks.c \
../freeRTOS711_All_Files/freeRTOS711/timers.c 

OBJS += \
./freeRTOS711_All_Files/freeRTOS711/croutine.o \
./freeRTOS711_All_Files/freeRTOS711/list.o \
./freeRTOS711_All_Files/freeRTOS711/queue.o \
./freeRTOS711_All_Files/freeRTOS711/tasks.o \
./freeRTOS711_All_Files/freeRTOS711/timers.o 

C_DEPS += \
./freeRTOS711_All_Files/freeRTOS711/croutine.d \
./freeRTOS711_All_Files/freeRTOS711/list.d \
./freeRTOS711_All_Files/freeRTOS711/queue.d \
./freeRTOS711_All_Files/freeRTOS711/tasks.d \
./freeRTOS711_All_Files/freeRTOS711/timers.d 


# Each subdirectory must supply rules for building sources it contributes
freeRTOS711_All_Files/freeRTOS711/%.o: ../freeRTOS711_All_Files/freeRTOS711/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -Wall -g2 -gstabs -O0 -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


