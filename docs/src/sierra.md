# Sierra: Cairo Intermediate Representation

## Online Resources
- [Cairo 1.0](https://medium.com/starkware/cairo-1-0-aa96eefb19a0)
- [Reading Sierra: Starknet's secret sauce for Cairo 1.0](https://medium.com/yagi-fi/reading-sierra-starknets-secret-sauce-for-cairo-1-0-5bc73409e43c)
- [StarkWare Sessions 23 | Sierra - Enforcing Safety Using Typesystems | Shahar Papini](https://www.youtube.com/watch?v=-EHwaQuPuAA)
- [docs.rs: cairo_lang_sierra](https://docs.rs/cairo-lang-sierra/latest/cairo_lang_sierra/)
- [docs.rs: cairo_lang_sierra_ap_change](https://docs.rs/cairo-lang-sierra-ap-change/latest/cairo_lang_sierra_ap_change/)
- [docs.rs: cairo_lang_sierra_gas](https://docs.rs/cairo-lang-sierra-gas/latest/cairo_lang_sierra_gas/)
- [docs.rs: cairo_lang_sierra_generator](https://docs.rs/cairo-lang-sierra-generator/latest/cairo_lang_sierra_generator/)
- [docs.rs: cairo_lang_sierra_to_casm](https://docs.rs/crate/cairo-lang-sierra-to-casm/latest)

## Summary

### [Cairo 1.0](https://medium.com/starkware/cairo-1-0-aa96eefb19a0)

Cairo 1.0 will give developers a safer, simpler, more usable programming language.

At the heart of Cairo 1.0 will be Sierra, an intermediary representation layer that promises greater long term stability for Cairo programs.

Sierra advances Cairo to serve in a permissionless network:
- Protecting the network: it allows more robust DoS protection
- Protecting the user: it allows Ethereum-grade censorship resistance

While Cairo made STARKs accessible, it was originally designed as an assembly language, and as such it was written as a low level language.
Cairo was gradually made more expressive and more developer-friendly.

Introducing Sierra: ensuring every Cairo run can be proven

The main addition in Cairo 1.0 is Sierra (Safe Intermediate Representation). 
Sierra constitutes a new intermediate representation layer between Cairo 1.0 and Cairo byte code. 
Sierra’s goal is to ensure that every Cairo run — i.e. a Cairo program and its input — can be proven.

Sierra promises Cairo devs better future-proof code. 
Further stability is provided by the fact that StarkNet contracts won’t need recompiling in the case of improvements to the underlying system (e.g., CPU AIR architecture changes, improvements of the final translation from Sierra to Cairo byte code).

Proving every Cairo run. 

In old Cairo, a Cairo run can result in three cases — TRUE, FALSE, or failure. 
Failed runs can’t be proven. Sierra, ensures that a Cairo run will never fail, and can only result in TRUE or FALSE. 
This in turn, ensures that every Cairo run can be proven.

This introduction of Sierra has important implications for StarkNet as a permissionless network. 
Sierra ensures that even reverted transactions can be included in StarkNet blocks. 
This property will allow the StarkNet protocol to remain lean and simple without the need to add complex crypto-economic mechanisms.

### [Reading Sierra: Starknet's secret sauce for Cairo 1.0](https://medium.com/yagi-fi/reading-sierra-starknets-secret-sauce-for-cairo-1-0-5bc73409e43c)

Starknet also has an intermediate language called Sierra, built as a stable layer to allow for rapid iteration and innovation at the Cairo 1.0 language level. While not absolutely essential to write efficient programs (Starknet contracts written in high-level Cairo will be orders of magnitude more efficient than hand-optimized Solidity contracts), reading Sierra code can help us get a deeper understanding of the underlying type system, security risks and improving efficiency where it matters (e.g., reducing the storage footprint or execution steps in highly recursive functions).

The Sierra output is organized into 4 separate sections always presented in the same order. First come the type declarations. Next, declarations of built-in library functions that will be used. Then the sequence of statements and finally the declared Cairo functions.

A couple of observations:
- The Unit type () is a special case of an empty struct and is used as a default return type for procedures (functions without a declared return type)
- New temporary variables are created and indexed using square brackets ( `[0]` and `[1]` )
- Statements have the following form `<libfunc>(<inputs>) -> (<outputs>);`, i.e., we don't see the more traditional `[0] = struct_construct<Unit>();`
- Code blocks are separate from Cairo function declarations. A function is tied to a specific block of code by starting at a dedicated statement index location (note the do_nothing@0 which indicates that the function begins at the first statement).

### [Twitter thread](https://twitter.com/p_e/status/1633105488731619328)

For more complex functions Sierra does much more:
- includes corelib code
- translates control flow in function calls and jumps
- expresses compound types in terms of built-in types
- automatically includes ZK built-ins (remember 0.x?)

### [Twitter thread](https://twitter.com/lucas_lvy/status/1578341084735758336)

We introduce SIERRA a Safe IntErmediate RepResentAtion of @CairoLang that is always provable which means that it never fails so asserts will be transformed into if statements. Don't be scared of this it adds almost no overhead to your bytecode !
As SIERRA is always provable it's also a way to prevent DOS attacks. The network will be able to include failed transactions and collect fees on them.

### Crate Documentation

Sierra is an intermediate representation between high level Cairo and compilation targets, such as CASM. 
Sierra code is guaranteed to be “safe”* by construction. 
Sierra has a primitive, yet rich typing system to express all high level code while guaranteeing safety and allowing for efficient compilation down to the target.

Safety - this means a few things:
- There are no “panics” / “runtime errors”. Every function is guaranteed to return.
- There are no infinite loops. Moreover, every program “counts” its own steps, and returns when the limit is reached.
- Builtin library functions are always used correctly.
