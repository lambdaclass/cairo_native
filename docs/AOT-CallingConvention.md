# AOT calling convention:

## Arguments

  - Written on registers, then the stack.
  - Structs' fields are treated as individual arguments (flattened).
  - Enums are structs internally, therefore they are also flattened (including the padding).
    - The default payload works as expected since it has the correct signature.
    - All other payloads require breaking it down into bytes and scattering it through the padding
      and default payload's space.

## Return values

  - Indivisible values that do not fit within a single register (ex. felt252) use multiple registers
    (x0-x3 for felt252).
  - Struct arguments, etc... use the stack.

In other words, complex values require a return pointer while simple values do not but may still use
multiple registers if they don't fit within one.
<!-- PLT: ACK -->
