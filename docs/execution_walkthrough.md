# Execution Walkthrough

Given the following Cairo program:

```rust,noexecute
// This is the cairo program. It just adds two numbers together and returns the
// result in an enum whose variant is selected using the result's parity.
enum Parity<T> {
  Even: T,
  Odd: T, 
}
/// Add `lhs` and `rhs` together and return the result in `Parity::Even` if it's
/// even or `Parity::Odd` otherwise.
fn run(lhs: u128, rhs: u128) -> Parity<u128> {
  let res = lhs + rhs;
  if (res & 1) == 0 {
    Parity::Even(res)
  } else {
    Parity::Odd(res)
} }
```

Let's see how it is executed. We start with the following Rust code:

```rust,noexecute
let program = get_sierra_program();       // The result of the `cairo-compile` program.
let module = get_native_module(&program); // This compiles the Sierra program to
                                          // MLIR (not covered here).
```

## Execution engine preparation
Given a compiled Cairo program in an MLIR module, once it is lowered to the LLVM dialect we have two options to execute it: AOT and JIT.

### Using the JIT executor
If we decide to use the JIT executor we just create the jit runner and we're done.

```rust,noexecute
let program = get_sierra_program();
let module = get_native_module(&program);

// The optimization level can be `None`, `Less`, `Default` or `Aggressive`. They
// are equivalent to compiling a C program using `-O0`, `-O1`, `-O2` and `-O3`
// respectively.
let engine = JitNativeExecutor::from_native_module(module, OptLevel::Default);
```

### Using the AOT executor
Preparing the AOT executor is more complicated since we need to compile it into a shared library and load it from disk.

```rust,noexecute
let program = get_sierra_program();
let module = get_native_module(&program);

// Internally, this method will run all the steps mentioned before internally into
// temporary files and return a working `AotNativeExecutor`.
let engine = AotNativeExecutor::from_native_module(module, OptLevel::Default);
```

### Using caches
You can use caches to keep the compiled programs in memory or disk and reuse them between runs. You may use the `ProgramCache` type, or alternatively just `AotProgramCache` or `JitProgramCache` directly.

Adding programs to the program cache involves steps not covered here, but once they're inserted you can get executors like this:

```rust,noexecute
let engine = program_cache.get(key).expect("program not found");
```

## Invoking the program
Regardless of whether we decided to go with AOT or JIT, the program invocation involves the exact same steps. We need to know the entrypoint that we'll be calling and its arguments.

In a future we may be able to implement compile-time trampolines for known program signatures, but for now we need to call the `invoke_dynamic` or `invoke_dynamic_with_syscall_handler` methods which works with any signature.

> Note: A trampoline is a function that invokes an compiled MLIR function from Rust code.],

Now we need to find the function id:

```rust,noexecute
let program = get_sierra_program();

// The utility function needs the symbol of the entry point, which is built as
// follows:
//   <module-name>::<module-name>::<function-name>(<function-idx>)
//
// The `<function-idx>` comes from the Sierra program. It's the index of the
// function in the function declaration section.
let function_id = find_function_id(&program, "program::program::main(f0)");
```

The arguments must be placed in a list of `JitValue` instances. The builtins should be ignored since they are filled in automatically. The only builtins required are the `GasBuiltin` and `System` (aka. the syscall handler). They are only mandatory when required by the program itself.

```rust,noexecute
let engine = get_execution_engine(); // This creates the execution engine (covered before).

let args = [
  JitValue::Uint128(1234),
  JitValue::Uint128(4321),
];
```

> Note: Although it's called `JitValue` for now, it's not tied in any way to the JIT engine. `JitValue`s are used for both the AOT and JIT engines.],

Finally we can invoke the program like this:

```rust,noexecute
let engine = get_execution_engine();

let function_id = find_function_id(&program, "program::program::main(f0)");
let args = [
  JitValue::Uint128(1234),
  JitValue::Uint128(4321),
];

let execution_result = engine.invoke_dynamic(
  function_id, // The entry point function id.
  args,        // The slice of `JitValue`s.
  None,        // The available gas (if any).
)?;

// The return value has some useful information about the execution, like:
//   - The remaining gas, if any was supplied.
//   - The program's return value.
//   - The builtin usage statistics. These contain the number of times each builtin has been used.
println!("Remaining gas: {:?}",  execution_result.remaining_gas);
println!("Return value:  {:#?}", execution_result.return_value);
println!("Builtin stats: {:?}",  execution_result.builtin_stats);
```

Running the code above should print the following:

```rust,noexecute
Remaining gas: None
Return value:  Enum {
  tag: 0,
  value: Struct {
    fields: [
      Enum {
        tag: 1,
        value: Uint128(
          5555,
        ),
        debug_name: Some(
          "sample::sample::Parity::<core::integer::u128>",
        ),
      },
    ],
    debug_name: Some(
      "Tuple<sample::sample::Parity::<core::integer::u128>>",
    ),
  },
  debug_name: Some(
    "core::panics::PanicResult::<(sample::sample::Parity::<core::integer::u128>,)>",
  ),
}
Builtin stats: BuiltinStats { bitwise: 1, ec_op: 0, range_check: 1, pedersen: 0, poseidon: 0, segment_arena: 0 }
```

### Contracts
Contracts always have the same interface, therefore they have an alternative to `invoke_dynamic` called `invoke_contract_dynamic`.

```rust,noexecute
fn(Span<felt252>) -> PanicResult<Span<felt252>>;
```

This wrapper will attempt to deserialize the real contract arguments from the span of felts, invoke the contracts, and finally serialize and return the result. When this deserialization fails, the contract will panic with the mythical `Failed to deserialize param #N` error.

If the example program had the same interface as a contract (a span of felts) then it'd be invoked like this:

```rust,noexecute
let engine = get_execution_engine();

let function_id = find_function_id(&program, "program::program::main(f0)");
let args = [
  Felt::from(1234),
  Felt::from(4321),
];

let execution_result = engine.invoke_dynamic(
  function_id, // The entry point function id.
  args,        // The slice of `JitValue`s.
  None,        // The available gas (if any).
)?;

// The return value has some useful information about the execution, like:
//   - The remaining gas, if any was supplied.
//   - Whether the contract execution panicked.
//   - The contract's return values.
//   - The builtin usage statistics. These contain the number of times each builtin has been used.
println!("Remaining gas: {:?}", execution_result.remaining_gas);
println!("Failure flag:  {:?}", execution_result.failure_flag);
println!("Return value:  {:?}", execution_result.return_value);
println!("Builtin stats: {:?}", execution_result.builtin_stats);
```

Running the code above should print the following:

```
Remaining gas: None
Failure flag:  false
Return value:  [
  JitValue::Felt252(1),
  JitValue::Felt252(5555),
]
Builtin stats: BuiltinStats { bitwise: 1, ec_op: 0, range_check: 1, pedersen: 0, poseidon: 0, segment_arena: 0 }
```

## The Cairo Native runtime
Sometimes we need to use stuff that would be too complicated or error-prone to implement in MLIR, but that we have readily available from Rust. That's when we use the runtime library.

When using the JIT it'll be automatically linked (if compiled with support for it, which is enabled by default). If using the AOT, the `CAIRO_NATIVE_RUNTIME_LIBRARY` environment variable will have to be modified to point to `libcairo_native_runtime.a`, which is built and placed in the root folder by `make build`.

Although it's implemented in Rust, its functions use the C ABI and have Rust's name mangling disabled. This means that to the extern observer it's technically indistinguishible from a library written in C. By doing this we're making the functions callable from MLIR.

### Syscall handlers
The syscall handler is similar to the runtime in the sense that we have C-compatible functions called from MLIR, but it's different in that they're built into Cairo Native itself rather than an external library, and that their implementation is user-dependent.

To allow for user-provided syscall handler implementations we pass a pointer to a vtable every time we detect a `System` builtin. We need a vtable and cannot use function names because the methods themselves are generic over the syscall handler implementation.

> Note: The `System` is used only for syscalls; every syscall has it, therefore it's a perfect candidate for this use.

Those wrappers then receive a mutable reference to the syscall handler implementation. They are responsible of converting the MLIR-compatible inputs to the Rust representations, calling the implementation, and then converting the results back into MLIR-compatible formats.

This means that as far as the user is concerned, writing a syscall handler is equivalent to implementing the trait `StarknetSyscallHandler` for a custom type.

## Appendix: The C ABI and the trampoline
Normally, calling FFI functions in Rust is as easy as defining an extern function using C-compatible types. We can't do this here because we don't know the function's signature.

It all boils down to the [SystemV ABI](https://refspecs.linuxbase.org/elf/x86_64-abi-0.99.pdf) in `x86_64` or its equivalent for ARM. Both of them are really similar:
- The stack must be aligned to 16 bytes before calling.
- Function arguments are spread between some registers and the stack.
- Return values use either a few registers or require a pointer.
There's a few other quirks, like which registers are caller vs callee-saved, but they're not that relevant in this case.

### Arguments

Argument location in `x86_64`:
|  # | Reg.  | Description            |
|----|-------|------------------------|
|  1 | rdi   | A single 64-bit value. |
|  2 | rsi   | A single 64-bit value. |
|  3 | rdx   | A single 64-bit value. |
|  4 | rcx   | A single 64-bit value. |
|  5 | r8    | A single 64-bit value. |
|  6 | r9    | A single 64-bit value. |
| 7+ | Stack | Everything else.       |

Argument location in `aarch64`:

|  # | Reg.  | Description            |
|----|-------|------------------------|
| 1  | x0    | A single 64-bit value. |
| 2  | x1    | A single 64-bit value. |
| 3  | x2    | A single 64-bit value. |
| 4  | x3    | A single 64-bit value. |
| 5  | x4    | A single 64-bit value. |
| 6  | x5    | A single 64-bit value. |
| 7  | x6    | A single 64-bit value. |
| 8  | x7    | A single 64-bit value. |
| 9+ | Stack | Everything else.       |

Usually function calls have arguments of types other than just 64-bit integers. In those cases, for values smaller than 64 bits the smaller register variants are written. For values larger than 64 bits the value is split into multiple registers, but there's a catch: if when splitting the value only one value would remain in registers then that register is padded and the entire value goes into the stack. For example, an `u128` that would be split between registers and the stack is always padded and written entirely in the stack.

For complex values like structs, the types are flattened into a list of values when written into registers, or just written into the stack the same way they would be written into memory (aka. with the correct alignment, etc).

### Return values
As mentioned before, return values may be either returned in registers or memory (most likely the stack, but not necessarily).

Argument location in `x86_64`:

| # | Reg | Description                 |
|---|-----|-----------------------------|
| 1 | rax | A single 64-bit value.      |
| 2 | rdx | The "continuation" of `rax` |

Argument location in `aarch64`:

| # | Reg | Description                |
|---|-----|----------------------------|
| 1 | x0  | A single 64-bit value      |
| 2 | x1  | The "continuation" of `x0` |
| 3 | x2  | The "continuation" of `x1` |
| 4 | x3  | The "continuation" of `x2` |

Values are different that arguments in that only a single value is returned. If more than a single value needs to be returned then it'll use a pointer.

When a pointer is involved we need to pass it as the first argument. This means that every actual argument has to be shifted down one slot, pushing more stuff into the stack in the process.

### The trampoline
We cannot really influence what values are in the register or the stack from Rust, therefore we need something written in assembler to put everything into place and invoke the function pointer.

This is where the trampoline comes in. It's a simple assembler function that does three things:
1. Fill in the 6 or 8 argument registers with the first values in the data pointer and copy the rest into the stack as-is (no stack alignment or anything, we guarantee from the Rust side that the stack will end up properly aligned).
2. Invoke the function pointer.
3. Write the return values (in registers only) into the return pointer.

This function always has the same signature, which is C-compatible, and therefore can be used with Rust's FFI facilities without problems.

#### AOT calling convention:

##### Arguments
- Written on registers, then the stack.
- Structs' fields are treated as individual arguments (flattened).
- Enums are structs internally, therefore they are also flattened (including the padding).
  - The default payload works as expected since it has the correct signature.
  - All other payloads require breaking it down into bytes and scattering it through the padding
    and default payload's space.

##### Return values
- Indivisible values that do not fit within a single register (ex. felt252) use multiple registers (x0-x3 for felt252).
- Struct arguments, etc... use the stack.

In other words, complex values require a return pointer while simple values do not but may still use multiple registers if they don't fit within one.
