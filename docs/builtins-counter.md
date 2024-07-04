# Builtins Counter

## Introduction

The Cairo Native compiler records the usage of each builtins in order to provide information about the program's builtins consumption. This information is NOT used for the gas calculation, as the gas cost of builtins is already taken into account during the [gas accounting process](./gas.md). The builtins counter types can each be found in the [types folder](../src/types/). Taking the [Pedersen hash](../src/types/pedersen.rs) as an example, we see that the counters will be represented as i64 integers in MLIR.
Counters are then simply incremented by one each time the builtins are called from within the program.

## Example

Let us consider the following Cairo program which uses the `pedersen` builtin:

```rust
use core::integer::bitwise;
use core::pedersen::pedersen;

fn run_test() {
    let mut hash = pedersen(1.into(), 2.into());
    hash += 1;
}
```

We expect Native to increment the `pedersen` counter by 1 given the above code.
Let's first check how this compiles to Sierra:

```assembly
const_as_immediate<Const<felt252, 1>>() -> ([1]); // 0
const_as_immediate<Const<felt252, 2>>() -> ([2]); // 1
store_temp<felt252>([1]) -> ([1]); // 2
store_temp<felt252>([2]) -> ([2]); // 3
pedersen([0], [1], [2]) -> ([3], [4]); // 4
drop<felt252>([4]) -> (); // 5
store_temp<Pedersen>([3]) -> ([3]); // 6
return([3]); // 7

contracts::run_test@0([0]: Pedersen) -> (Pedersen);
```

In the compiled Sierra, we can see that the `pedersen` builtin is passed with the call to the `run_test` which starts at statement `0`. It is then used in the call to the `pedersen` libfunc. We would expect to see the `pedersen` counter incremented by 1 in the Native compiler. Below is the compiled MLIR dump for the same program:

```assembly
...
llvm.func @"test::test::run_test(f0)"(%arg0: i64 loc(unknown)) -> i64 attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(2 : i256) : i256 loc(#loc1)
    %1 = llvm.mlir.constant(1 : i256) : i256 loc(#loc1)
    %2 = llvm.mlir.constant(1 : i64) : i64 loc(#loc1)
    %3 = llvm.alloca %2 x i256 {alignment = 16 : i64} : (i64) -> !llvm.ptr loc(#loc2)
    %4 = llvm.alloca %2 x i256 {alignment = 16 : i64} : (i64) -> !llvm.ptr loc(#loc2)
    %5 = llvm.alloca %2 x i256 {alignment = 16 : i64} : (i64) -> !llvm.ptr loc(#loc2)
    %6 = llvm.add %arg0, %2  : i64 loc(#loc2)
    %7 = llvm.intr.bswap(%1)  : (i256) -> i256 loc(#loc2)
    %8 = llvm.intr.bswap(%0)  : (i256) -> i256 loc(#loc2)
    llvm.store %7, %3 {alignment = 16 : i64} : i256, !llvm.ptr loc(#loc2)
    llvm.store %8, %4 {alignment = 16 : i64} : i256, !llvm.ptr loc(#loc2)
    llvm.call @cairo_native__libfunc__pedersen(%5, %3, %4) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> () loc(#loc2)
    llvm.return %6 : i64 loc(#loc3)
  } loc(#loc1)
  ...
```

The compiled MLIR function `run_test` takes a single argument as input, the `pedersen` counter and returns the incremented counter at the end of the call. The counter is incremented by 1 in the MLIR code, in the statement `%6 = llvm.add %arg0, %2  : i64 loc(#loc2)`, which takes the `%arg0` input and adds `%2` to it. We can see from statement `%2 = llvm.mlir.constant(1 : i64) : i64 loc(#loc1)` that `%2` holds the constant 1.
When this compiled MLIR code is called, the initial value of all builtin counters is set to `0` as can be seen in the [`invoke_dynamic` function](../src/executor.rs#L240).
