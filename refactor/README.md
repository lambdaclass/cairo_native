# Compiler Refactor

This PR refactors the compiler to:
  - Simplify the code by a lot.
  - Improve component locality (libfuncs, for example).
  - And most importantly, use the C++ APIs which we'll be needed for starknet contracts.


## New design

First of all, there's the concept of types and libfunc processors. They are closures which accept a
builder and the item declaration from the compiler, and generate the compiled type and libfunc
generator respectively.

Those processors are registered into providers, which are just a mapping from a string to said
closure. The providers will be used by the compiler when processing the program's types and
libfuncs. The compiler implements those libfuncs for the standard Sierra types and libfuncs.


When compiling Sierra programs, there are four kinds of items to process: types, libfuncs,
statements and functions. In our compiler, they are to be processed in the following order:
  1. Types.
  2. Functions (only the declarations).
  3. Libfuncs.
  4. Statements (the actual function bodies).

It has to be this way because of the dependencies between them. For example, types do not depend on
anything, but the function declarations need the types for its argument and return types. Similarly,
libfuncs need both types and functions (due to the function call libfuncs, for example). Finally,
statements need everything else. Each item in their kind should also be ordered topologically. For
example, when processing types with generics, there'll be types that depend on whether other types
have already been compiled.


## Why this refactor?

With every libfunc and type we add, the design gets more and more complicated. This is due to us
having had to learn everything from scratch. Now that we have an idea of what we're doing (more or
less) it'd be nice to have an API to work with which is
  - Simpler internally.
  - Requires us to keep track of less stuff when using it.
  - Is externally extensible (that is, allows for custom user-defined types and libfuncs making it
    future-proof).

This new design, begin simpler will reduce the attack surface available for bugs and exploits, as
well as make it so that all libfuncs can be implemented in the same way (right now branching
libfuncs have to be implemented in a different way than non-branching ones).


Another reason for this refactor, probably more important, is that we need some MLIR APIs which are
not available when using the C bindings, as we are doing now. For example, when using starknet
contracts we need to support entry points, which is fine when they neither accept nor return
anything. However, when we need to pass or return stuff from an entrypoint we'll need to allocate
their data, therefore we'll need the MLIR type sizes (which can be induced from our side without
their API, so no problem here) but also the alignment used by MLIR so that our allocations are
compatible with MLIR's ABI. This API (that returns the size and align of the types) is not available
in C, and exposing it would require mixing the C and C++ APIs, then generating bindings of that to
be used from Rust. Needless to say, this starts being absurd since we can use the C++ API almost
directly (only requiring proxy functions).
