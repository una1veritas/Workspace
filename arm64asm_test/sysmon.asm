// macOS ARM64 Assembly System Monitor 
.global _main
.align 4

.text
_main:
    // Create stack frame (satisfies Apple's 16-byte stack alignment ABI)
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp

    // Print welcome header
    adrp    x0, msg_header@PAGE
    add     x0, x0, msg_header@PAGEOFF
    bl      _printf

    // Query Total Physical Memory using sysctlbyname("hw.memsize", ...)
    adrp    x0, sysctl_hw_memsize@PAGE
    add     x0, x0, sysctl_hw_memsize@PAGEOFF  // Arg 1: Name of property
    add     x1, x29, #16                       // Arg 2: Buffer address (on stack)
    add     x2, x29, #24                       // Arg 3: Pointer to buffer size
    mov     x3, #0                             // Arg 4: newp (NULL)
    mov     x4, #0                             // Arg 5: newlen (0)
    
    // Initialize buffer size variable to 8 bytes (sizeof uint64_t)
    mov     x5, #8
    str     x5, [x29, #24]

    // Call native macOS library sysctlbyname
    bl      _sysctlbyname

    // Check if sysctl call failed (returned non-zero)
    cmp     w0, #0
    b.ne    error_exit

    // Load returned memory value (in bytes) and convert to Gigabytes (GB)
    ldr     x1, [x29, #16]
    lsr     x1, x1, #30                        // Shift right by 30 (Divide by 1024^3)

    // Print the extracted memory value
    adrp    x0, msg_mem@PAGE
    add     x0, x0, msg_mem@PAGEOFF
    bl      _printf

    // Normal Exit
    mov     w0, #0
    b       program_exit

error_exit:
    adrp    x0, msg_error@PAGE
    add     x0, x0, msg_error@PAGEOFF
    bl      _printf
    mov     w0, #1

program_exit:
    // Restore stack frame and return
    ldp     x29, x30, [sp], #32
    ret

.data
.align 4
sysctl_hw_memsize: .asciz "hw.memsize"
msg_header:         .asciz "=== macOS ARM64 Assembly System Monitor ===\n"
msg_mem:            .asciz "Total Physical Memory: %llu GB\n"
msg_error:          .asciz "Error reading system metrics.\n"
