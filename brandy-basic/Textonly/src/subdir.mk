################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/assign.c \
../src/brandy.c \
../src/commands.c \
../src/console.c \
../src/convert.c \
../src/editor.c \
../src/emulate.c \
../src/errors.c \
../src/evaluate.c \
../src/fileio.c \
../src/functions.c \
../src/heap.c \
../src/iostate.c \
../src/keyboard.c \
../src/lvalue.c \
../src/mainstate.c \
../src/miscprocs.c \
../src/stack.c \
../src/statement.c \
../src/strings.c \
../src/textonly.c \
../src/tokens.c \
../src/variables.c 

OBJS += \
./src/assign.o \
./src/brandy.o \
./src/commands.o \
./src/console.o \
./src/convert.o \
./src/editor.o \
./src/emulate.o \
./src/errors.o \
./src/evaluate.o \
./src/fileio.o \
./src/functions.o \
./src/heap.o \
./src/iostate.o \
./src/keyboard.o \
./src/lvalue.o \
./src/mainstate.o \
./src/miscprocs.o \
./src/stack.o \
./src/statement.o \
./src/strings.o \
./src/textonly.o \
./src/tokens.o \
./src/variables.o 

C_DEPS += \
./src/assign.d \
./src/brandy.d \
./src/commands.d \
./src/console.d \
./src/convert.d \
./src/editor.d \
./src/emulate.d \
./src/errors.d \
./src/evaluate.d \
./src/fileio.d \
./src/functions.d \
./src/heap.d \
./src/iostate.d \
./src/keyboard.d \
./src/lvalue.d \
./src/mainstate.d \
./src/miscprocs.d \
./src/stack.d \
./src/statement.d \
./src/strings.d \
./src/textonly.d \
./src/tokens.d \
./src/variables.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


