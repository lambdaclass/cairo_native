# Code Layout

The code is laid out in the following sections:

## libfuncs

Path: `src/libfuncs`

Here are stored all the library function implementations in MLIR, this contains the majority of the code.

To store information about the different types of library functions sierra has, we divide them into the following using the enum `SierraLibFunc`:

- Branching: These functions are implemented inline, adding blocks and jumping as necessary based on given conditions.
- Constant: A constant value, this isn't represented as a function and is inserted inline.
- Function: Any other function.
- InlineDataFlow: Functions that can be implemented inline without much problem. For example: dup, store_temp

## Statements

Path: `src/statements`

Here is the code that processes the statements of non-library functions. It handles dataflow, branching, function calls, variable storage and also has implementations for the inline library functions.

## User functions

These are extra utility functions unrelated to sierra that aid in the development, such as wrapping return values and printing them.
