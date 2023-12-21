.text


.global aot_trampoline
aot_trampoline:
    # rdi <- fn_ptr: extern "C" fn()
    # rsi <- args_ptr: *const u64
    # rdx <- args_len: usize
    # rcx <- ret_ptr: &mut [u64; 2]

    push    rbp                     # Push rbp (callee-saved).
    push    rcx                     # Push rcx (ret_ptr).
    mov     rbp,    rsp             # Store the current stack pointer.
    sub     rsp,    8               # Align the stack.

    mov     r10,    rdi             # We'll need rdi.
    mov     r11,    rsi             # We'll need rsi.

    cmp     rdx,    6               # Check if there are more than 6 arguments.
    jbe     2f                      # If there are less than 6, skip to register arguments.

    #
    # Process stack arguments.
    #

    # Add padding to support an odd number of stack parameters.
    mov     rax,    rdx
    and     rax,    1
    lea     rsp,    [rsp + 8 * rax]

  1:
    dec     rdx                     # Decrement length.
    mov     rax,    [r11 + 8 * rdx] # Load the value.
    push    rax                     # Push it into the stack.

    cmp     rdx,    6               # Check if there are more than 6 arguments.
    ja      1b                      # If there still are, loop back and repeat.

  2:
    #
    # Process registers.
    #

    shl     rdx,    2               # Multiply remaining length by 4.
    lea     rax,    [rip + 3f]      # Load the PC-relative address of `3f`.
    sub     rax,    rdx             # Subtract 4 * remaining_len (rdx).
    jmp     rax                     # Jump to the resulting address.

    mov     r9,     [r11 + 0x28]    # Load argument #6.
    mov     r8,     [r11 + 0x20]    # Load argument #5.
    mov     rcx,    [r11 + 0x18]    # Load argument #4.
    mov     rdx,    [r11 + 0x10]    # Load argument #3.
    mov     rsi,    [r11 + 0x08]    # Load argument #2.
    nop                             # Note: The previous 5 `mov` instructions use 4 bytes each, but
                                    #   the last one only takes 3. This `nop` (1 byte) is used to
                                    #   align them all at 4 bytes so that the last jump instruction
                                    #   works correctly.
    mov     rdi,    [r11]           # Load argument #1.

  3:
    # Call the function.
    call    r10

    mov     rsp,    rbp
    pop     rcx
    pop     rbp

    # Store return registers.
    mov     [rcx],      rax
    mov     [rcx + 8],  rdx

    ret
