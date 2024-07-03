## The cairo native runtime

This runtime is used automatically when using JIT, but when using AOT, the library needs to be shipped and put in a path where it can be found by a linker.

### Getting the library for use in AOT

```bash
git clone https://github.com/lambdaclass/cairo_native
cd cairo_native
make runtime
ls libcairo_native_runtime.*

# copy it where you need it, such as /usr/local/lib where it will be found by the linker
```
<!-- PLT: ACK -->
