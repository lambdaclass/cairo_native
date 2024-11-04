# Sierra Resources

- [StarkWare Sessions 23 | Sierra - Enforcing Safety Using Typesystems | Shahar Papini](https://www.youtube.com/watch?v=-EHwaQuPuAA)
- [Reading Sierra: Starknet's secret sauce for Cairo 1.0](https://medium.com/yagi-fi/reading-sierra-starknets-secret-sauce-for-cairo-1-0-5bc73409e43c)
- [Under the hood of Cairo 1.0: Exploring Sierra Part 1](https://medium.com/nethermind-eth/under-the-hood-of-cairo-1-0-exploring-sierra-7f32808421f5)
- [Under the hood of Cairo 1.0: Exploring Sierra Part 2](https://medium.com/nethermind-eth/under-the-hood-of-cairo-1-0-exploring-sierra-9355d618b26f)
- [Under the hood of Cairo 1.0: Exploring Sierra Part 3](https://medium.com/nethermind-eth/under-the-hood-of-cairo-1-0-exploring-sierra-1220f6dbcf9)
- Crate documentation
    - [docs.rs: cairo_lang_sierra](https://docs.rs/cairo-lang-sierra/latest/cairo_lang_sierra/)
    - [docs.rs: cairo_lang_sierra_ap_change](https://docs.rs/cairo-lang-sierra-ap-change/latest/cairo_lang_sierra_ap_change/)
    - [docs.rs: cairo_lang_sierra_gas](https://docs.rs/cairo-lang-sierra-gas/latest/cairo_lang_sierra_gas/)
    - [docs.rs: cairo_lang_sierra_generator](https://docs.rs/cairo-lang-sierra-generator/latest/cairo_lang_sierra_generator/)
    - [docs.rs: cairo_lang_sierra_to_casm](https://docs.rs/crate/cairo-lang-sierra-to-casm/latest)

## What is a library function
Sierra uses a list of builtin functions that implement the language
functionality, those are called library functions, short: libfuncs.
Basically every statement in a sierra program is a call to a libfunc, thus
they are the core of Cairo Native.

Each libfunc takes input variables and outputs some other variables. Note
that in cairo a function that has 2 arguments may have more in sierra, due
to "implicits" / "builtins", which are arguments passed hidden from the
user, such as the `GasBuiltin`.
