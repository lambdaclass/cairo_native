searchState.loadedDescShard("cairo_native", 0, "⚡ Cairo Native ⚡\nall elements need to be same type\nUsed as return value for Nullables that are null.\nOptimization levels.\nA JitValue is a value that can be passed to the JIT engine …\nRun the compiler on a program. The compiled program is …\nCairo Native Compiler and Execution Engine\nVarious error types used thorough the crate.\nExecutors\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCode generation metadata\nConverts a MLIR module to a compile object, that can be …\nLinks the passed object into a shared library, stored on …\nStarknet related code for <code>cairo_native</code>\nA (somewhat) usable implementation of the starknet syscall …\nVarious utilities\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nA Cache for programs with the same context.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nContext of IRs, dialects and passes for Cairo programs …\nCompiles a sierra program into MLIR and then lowers to …\nReturns the argument unchanged.\nInitialize an MLIR context.\nCalls <code>U::from(self)</code>.\nOverview\nCompilation Walkthrough\nExecution Walkthrough\nGas and Builtins Accounting\nImplementing Libfuncs\nDebugging\nSierra Resources\nMLIR Resources\nContains the error value\nContains the success value\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nStarknet contract execution result.\nThe result of the JIT execution.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nConvert an <code>ExecutionResult</code> into a <code>ContractExecutionResult</code>\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nPlease look at the module level docs.\nA MLIR JIT execution engine in the context of Cairo Native.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nUtility to convert a <code>NativeModule</code> into an <code>AotNativeExecutor</code>…\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nExecute a program with the given params.\nExecute a program with the given params.\nLoad the executor from an already compiled library with …\nCreate the executor from a sierra program with the given …\nRuns the given entry point.\nSave the library to the desired path, alongside it is …\nMetadata container.\nDebug utilities\nReturns the argument unchanged.\nRetrieve a reference to some metadata.\nRetrieve a mutable reference to some metadata.\nInsert some metadata and return a mutable reference.\nCalls <code>U::from(self)</code>.\nCreate an empty metadata container.\nMemory allocation external bindings\nRemove some metadata and return its last value.\nRuntime library bindings\nTail recursion information\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nPrints the given &amp;str.\nDump a memory region at runtime.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nHolds global gas info.\nError for metadata calculations.\nConfiguration for metadata computation.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the initial value for the gas counter. If …\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nMemory allocation <code>realloc</code> metadata.\nCalls the <code>free</code> function.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nRegister the bindings to the <code>realloc</code> C function and return …\nCalls the <code>realloc</code> function, returns a op with 1 result: an …\nRuntime library bindings metadata.\nRegister if necessary, then invoke the <code>dict_alloc_new()</code> …\nRegister if necessary, then invoke the <code>dict_alloc_new()</code> …\nRegister if necessary, then invoke the <code>dict_gas_refund()</code> …\nRegister if necessary, then invoke the <code>dict_get()</code> function.\nRegister if necessary, then invoke the <code>dict_insert()</code> …\nRegister if necessary, then invoke the <code>dict_clone()</code> …\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nRegister if necessary, then invoke the <code>debug::print()</code> …\nRegister if necessary, then invoke the <code>ec_point_from_x_nz()</code>…\nRegister if necessary, then invoke the …\nRegister if necessary, then invoke the <code>ec_state_add()</code> …\nRegister if necessary, then invoke the <code>ec_state_add_mul()</code> …\nRegister if necessary, then invoke the <code>ec_state_init()</code> …\nRegister if necessary, then invoke the <code>poseidon()</code> function.\nRegister if necessary, then invoke the <code>pedersen()</code> function.\nRegister if necessary, then invoke the <code>vtable_cheatcode()</code> …\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nThe tail recursion metadata.\nReturn the current depth counter value.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCreate the tail recursion meta.\nReturn the recursion target block.\nReturn the return target block, if set.\nSet the return target block.\nA MLIR module in the context of Cairo Native. It is …\nReturns the argument unchanged.\nRetrieve a reference to some stored metadata.\nInsert some metadata for the program execution and return …\nCalls <code>U::from(self)</code>.\nRemoves metadata\nContains the error value\nBinary representation of a <code>Felt</code> (in MLIR).\nContains the success value\nBinary representation of a <code>u256</code> (in MLIR).\nRuntime function that calls the <code>cheatcode</code> syscall\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nEvent emitted by the emit_event syscall.\nA (somewhat) usable implementation of the starknet syscall …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nThe <code>felt252</code> prime modulo.\nCompile a cairo program found at the given path to sierra.\nCreates the execution engine, with all symbols registered.\nReturn a type that calls a closure when formatted using …\nDecode an UTF-8 error message replacing invalid bytes with …\nParse any type that can be a bigint to a felt that can be …\nParse a short string into a felt that can be used in the …\nParse a numeric string into felt, wrapping negatives …\nReturns the given entry point if present.\nReturns the given entry point if present.\nGiven a string representing a function name, searches in …\nReturns the argument unchanged.\nGenerate a function name.\nReturn the layout for an integer of arbitrary width.\nCalls <code>U::from(self)</code>.\nCopied from std.\nEdit: Copied from the std lib.")