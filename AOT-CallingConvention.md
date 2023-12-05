# AOT calling convention:

## Arguments

  - Written on registers.
  - Structs' fields are treated as individual arguments (flattened).

Enums are special, their payload always skips registers until it's stored within the stack. Values
following an enum will always be within the stack.

## Return values

  - Indivisible values that do not fit within a single register (ex. felt252) use multiple registers
    (x0-x3 for felt252).
  - Struct arguments, etc... use the stack.

In other words, complex values require a return pointer while simple values do not but may still use
multiple registers if they don't fit within one.
