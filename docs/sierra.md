# Sierra Resources

🚧 TODO

## What is a library function
Sierra uses a list of builtin functions that implement the language
functionality, those are called library functions, short: libfuncs.
Basically every statement in a sierra program is a call to a libfunc, thus
they are the core of Cairo Native.

Each libfunc takes input variables and outputs some other variables. Note
that in cairo a function that has 2 arguments may have more in sierra, due
to "implicits" / "builtins", which are arguments passed hidden from the
user, such as the `GasBuiltin`.
