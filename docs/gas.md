# Gas

## Introduction

Gas management in a blockchain environment involves accounting for the amount of computation performed during the execution of a transaction. This is used to accurately charge the user at the end of the execution or to revert early if the transaction consumes more gas than provided by the sender.

This documentation assumes prior knowledge about Sierra and about the way gas accounting is performed in Sierra. This documentation assumes prior knowledge about Sierra and the way gas accounting is performed in Sierra. This documentation assumes familiarity with Sierra and its gas accounting mechanisms. For those seeking to deepen their understanding, refer to Enitrat’s Medium post about [Sierra](https://medium.com/nethermind-eth/under-the-hood-of-cairo-1-0-exploring-sierra-7f32808421f5) and greged’s about [gas accounting in Sierra](https://blog.kakarot.org/understanding-sierra-gas-accounting-19d6141d28b9).

## Gas builtin

The gas builtin is used in Sierra in order to perform gas accounting. It is passed as an input to all function calls and holds the current remaining gas. It is represented in MLIR by a simple `u128`.

## Gas metadata

The process of calculating gas begins at the very outset of the compilation process. During the initial setup of the Sierra program, metadata about the program, including gas information, is extracted. Using gas helper functions for the [Cairo compiler](https://github.com/starkware-libs/cairo/tree/main), the consumed cost (steps, memory holes, builtins usage) for each statement in the Sierra code is stored in a HashMap.

## Withdrawing gas

The action of withdrawing gas can be split in two steps:

- **Calculating Total Gas Cost**: Using the previously constructed HashMap, we iterate over the various cost tokens (including steps, built-in usage, and memory holes) for the statement, convert them into a [common gas unit](https://github.com/starkware-libs/cairo/blob/v2.7.0-dev.0/crates/cairo-lang-runner/src/lib.rs#L136), and sum them up to get the total gas cost for the statement.
- **Executing Gas Withdrawal**: The previously calculated gas cost is used when the current statement is a `withdraw_gas` libfunc call.

The `withdraw_gas` libfunc takes the current leftover gas as input and uses the calculated gas cost for the statement to deduct the appropriate amount from the gas builtin. In the compiled IR, gas withdrawal appears as the total gas being reduced by a predefined constant. Additionally, the libfunc branches based on whether the remaining gas is greater than or equal to the amount being withdrawn.

## Example

Let's illustrate this with a simple example using the following Cairo 1 code:

```rust
fn run_test() {
    let mut i: u8 = 0;
    let mut val = 0;
    while i < 5 {
        val = val + i;
        i = i + 1;
    }
}
```

As noted earlier, gas usage is initially computed by the Cairo compiler for each state. A snippet of the resulting HashMap shows the cost for each statement:

```json
...
(
    StatementIdx(
        26,
    ),
    Const,
): 2680,
(
    StatementIdx(
        26,
    ),
    Pedersen,
): 0,
(
    StatementIdx(
        26,
    ),
    Poseidon,
): 0,
...
```

For statement 26, the cost of the `Const` token type (a combination of step, memory hole, and range check costs) is 2680, while other costs are 0. Let's see which libfunc is called at statement 26:

```assembly
...
disable_ap_tracking() -> (); // 25
withdraw_gas([0], [1]) { fallthrough([4], [5]) 84([6], [7]) }; // 26
branch_align() -> (); // 27
const_as_immediate<Const<u8, 5>>() -> ([8]); // 28
...
```

When the Cairo native compiler reaches statement 26, it combines all costs into gas using the Cairo compiler code. In this example, the total cost is 2680 gas. This value is then used in the outputted IR to withdraw the gas and determine whether execution should revert or continue. This can be observed in the following MLIR dump:

```assembly
llvm.func @"test::test::run_test[expr16](f0)"(%arg0: i64 loc(unknown), %arg1: i128 loc(unknown), %arg2: i8 loc(unknown), %arg3: i8 loc(unknown)) -> !llvm.struct<(i64, i128, struct<(i64, array<24 x i8>)>)> attributes {llvm.emit_c_interface} {
...
%12 = llvm.mlir.constant(5 : i8) : i8 loc(#loc1)
%13 = llvm.mlir.constant(2680 : i128) : i128 loc(#loc1)
%14 = llvm.mlir.constant(1 : i64) : i64 loc(#loc1)
...
^bb1(%27: i64 loc(unknown), %28: i128 loc(unknown), %29: i8 loc(unknown), %30: i8 loc(unknown)):  // 2 preds: ^bb0, ^bb6
  %31 = llvm.add %27, %14  : i64 loc(#loc13)
  %32 = llvm.icmp "uge" %28, %13 : i128 loc(#loc13)
  %33 = llvm.intr.usub.sat(%28, %13)  : (i128, i128) -> i128 loc(#loc13)
  llvm.cond_br %32, ^bb2(%29 : i8), ^bb7(%5, %23, %23, %31 : i252, !llvm.ptr, !llvm.ptr, i64) loc(#loc13)
 ...
```

Here, we see the constant `2680` defined at the begining of the function's definition. In basic block 1, the withdraw_gas operations are performed: by comparing %28 (remaining gas) and %13 (gas cost), the result stored in %32 determines the conditional branching. A saturating subtraction between the remaining gas and the gas cost is then performed, updating the remaining gas in the IR.

## Final gas usage

The final gas usage can be easily retrieved from the gas builtin value returned by the function. This is accomplished when [parsing the return values](https://github.com/lambdaclass/cairo_native/blob/65face8194054b7ed396a34a60e7b1595197543a/src/executor.rs#L286) from the function call:

```rust
...
for type_id in &function_signature.ret_types {
    let type_info = registry.get_type(type_id).unwrap();
    match type_info {
        CoreTypeConcrete::GasBuiltin(_) => {
            remaining_gas = Some(match &mut return_ptr {
                Some(return_ptr) => unsafe { *read_value::<u128>(return_ptr) },
                None => {
                    // If there's no return ptr then the function only returned the gas. We don't
                    // need to bother with the syscall handler builtin.
                    ((ret_registers[1] as u128) << 64) | ret_registers[0] as u128
                }
            });
        }
        ...
    }
    ...
}
...
```

This code snippet extracts the remaining gas from the return pointer based on the function's signature. If the function only returns the gas value, the absence of a return pointer is handled appropriately, ensuring accurate gas accounting.
