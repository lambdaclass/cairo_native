.text


.global _aot_trampoline
_aot_trampoline:
    // x0 <- fn_ptr: extern "C" fn()
    // x1 <- args_ptr: *const u8
    // x2 <- args_len: usize

    //
    // Check zero arguments.
    //
    cmp     x2,     0                       // If there are no arguments,
    beq     3f                              // skip argument processing completely.

    //
    // Preparation.
    //
    add     x1,     x1,     x2,     lsl 3   // Move the pointer to the end (past last element).

    str     x19,    [sp, #-8]!              // Push the x19 register (we'll use it to store sp).
    mov     x19,    sp                      // Store the original stack pointer for later.

    //
    // Copy stack arguments (for n_args > 8).
    //
    cmp     x2,     8                       // Check if there are more than 8 arguments.
    ble     2f                              // If there are less than 8, skip to register arguments.

    //
    // Process stack arguments.
    //
  1:
    sub     x2,     x2,     1               // Decrement length.
    ldr     x3,     [x1, #-8]!              // Decrement pointer, then load the value.
    str     x3,     [sp, #-8]!              // Reserve stack memory, then write the value.

    cmp     x2,     8                       // Check if there are more than 8 arguments.
    bgt     1b                              // If there still are, loop back and repeat.

  2:
    //
    // Process registers.
    //
    mov     x9,     x0                      // We'll need x0.
    mov     x10,    x1                      // We'll need x1.

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
    blr     x0

    // Restore the original stack pointer and x19 register.
    mov     sp,     x19
    ldr     x19,    [sp],   8

    ret
