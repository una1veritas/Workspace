section     .data
    msg db  'Hello, macOS!', 0

section     .text
    global  _start

_start:
    ; Write system call (stdout)
    mov rdi, 1 ; File descriptor (stdout)
    mov rsi, msg ; Pointer to the message
    mov rdx, 14 ; Length of the message
    mov rax, 0x2000004 ; Syscall number for write (0x2000004 for x86_64)
    syscall ; Invoke the system call

    ; Exit system call (exit status 0)
    mov rdi, 0 ; Exit status 0
    mov rax, 0x2000001 ; Syscall number for exit (0x2000001 for x86_64)
    syscall ; Invoke the system call
