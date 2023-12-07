.text


.global aot_trampoline
aot_trampoline:
    # rdi <- fn_ptr: extern "C" fn()
    # rsi <- args_ptr: *const u64
    # rdx <- args__len: usize

    push    rbp             # Push rbp (callee-saved).
    mov     rbp,    rsp     # Store the current stack pointer.

    mov     r10,    rdi     # We'll need rdi.
    mov     r11,    rsi     # We'll need rsi.

    cmp     rdx,    6
    jbe     2f

    mov     rax,    rdx
    and     rax,    1
    lea     rsp,    [rsp + 8 * rax]

  1:
    dec     rdx
    mov     rax,    [r11 + 8 * rdx]
    push    rax

    cmp     rdx,    6
    ja      1b

  2:
    shl     rdx,    2
    lea     rax,    [rip + 3f]
    sub     rax,    rdx
    jmp     rax

    mov     r9,     [r11 + 0x28]
    mov     r8,     [r11 + 0x20]
    mov     rcx,    [r11 + 0x18]
    mov     rdx,    [r11 + 0x10]
    mov     rsi,    [r11 + 0x08]
    nop
    mov     rdi,    [r11]

  3:
    call    r10

    mov     rsp,    rbp
    pop     rbp
    ret
