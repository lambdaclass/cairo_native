# Introduction

A compiler to convert Cairo's intermediate representation "Sierra" code to MLIR.

## Compilation process

```mermaid
stateDiagram-v2
    state "Load sierra program" as sierra
    state "Initialize compiler" as init
    state "Initialize execution engine" as engine
    state if_skip_jit <<choice>>
    state "Load MLIR dialects" as dialects
    state "Create builtin module" as module
    state "Create libc wrappers" as libc
    state "Process Types" as types
    state "Process Library functions" as libfuncs
    state "Save non-flow function info" as func_info
    state "Process functions" as funcs
    state "Calculate block ranges per function" as blocks
    state "Process statements" as statements
    state "Apply MLIR passes" as passes
    [*] --> Initialize
    state Initialize {
        sierra --> init
        init --> if_skip_jit
        if_skip_jit --> engine: if JIT
        if_skip_jit --> dialects: if Compile
        engine --> dialects
    }
    Initialize --> Compile
    state Compile {
        module --> libc
        libc --> types
        types --> libfuncs
        types --> func_info
        func_info --> libfuncs
        libfuncs --> funcs
        funcs --> blocks
        blocks --> statements
    }
    Compile --> passes
    passes --> Output
    Output --> [*]
```
