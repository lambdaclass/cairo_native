.text


.global _aot_trampoline
_aot_trampoline:
    // x0 <- fn_ptr: extern "C" fn()
    // x1 <- args_ptr: *const u64
    // x2 <- args_len: usize

    stp     x29,    x30,    [sp, #-16]!
    str     x19,    [sp, #-16]!             // Necessary to restore the stack after the call.
    mov     x19,    sp

    mov     x9,     x0                      // We'll need x0.
    add     x10,    x1,     x2,     lsl 3   // Move the pointer to the end (past last element).

    cmp     x2,     8                       // Check if there are more than 8 arguments.
    ble     2f                              // If there are less than 8, skip to register arguments.

    //
    // Process stack arguments.
    //

    // Add padding to support an odd number of stack parameters.
    and     x0,     x2,     1
    sub     x4,     sp,     x0,     lsl 3   // ARM doesn't like `str reg, [sp]` instructions when
                                            // the stack isn't 16-byte aligned (not just on `bl`).

  1:
    sub     x2,     x2,     1               // Decrement length.
    ldr     x3,     [x10, #-8]!             // Decrement pointer, then load the value.
    str     x3,     [x4, #-8]!              // Reserve stack memory, then write the value.

    cmp     x2,     8                       // Check if there are more than 8 arguments.
    bgt     1b                              // If there still are, loop back and repeat.

    mov     sp,     x4

  2:
    //
    // Process registers.
    //

    adr     x0,     3f                      // Load address of label 3f.
    sub     x0,     x0,     x2,     lsl 2   // Subtract 4 * n_args.
    br      x0                              // Jump.

    ldr     x7,     [x10, #-8]!             // Decrement pointer, then load the value.
    ldr     x6,     [x10, #-8]!             // Decrement pointer, then load the value.
    ldr     x5,     [x10, #-8]!             // Decrement pointer, then load the value.
    ldr     x4,     [x10, #-8]!             // Decrement pointer, then load the value.
    ldr     x3,     [x10, #-8]!             // Decrement pointer, then load the value.
    ldr     x2,     [x10, #-8]!             // Decrement pointer, then load the value.
    ldr     x1,     [x10, #-8]!             // Decrement pointer, then load the value.
    ldr     x0,     [x10, #-8]!             // Decrement pointer, then load the value.

  3:
    // Call the function.
    blr     x9

    mov     sp,     x19
    ldr     x19,    [sp],   16
    ldp     x29,    x30,    [sp],   16
    ret
