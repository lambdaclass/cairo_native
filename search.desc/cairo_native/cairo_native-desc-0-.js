searchState.loadedDescShard("cairo_native", 0, "Cairo Sierra to MLIR compiler and JIT engine\nA error from the LLVM API.\nOptimization levels.\nRun the compiler on a program. The compiled program is …\nVarious error types used thorough the crate.\nExecutors\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCompiler libfunc infrastructure\nCode generation metadata\nConverts a MLIR module to a compile object, that can be …\nLinks the passed object into a shared library, stored on …\nStarknet related code for <code>cairo_native</code>\nA (somewhat) usable implementation of the starknet syscall …\nCompiler type infrastructure\nVarious utilities\nJIT params and return values de/serialization\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nA Cache for programs with the same context.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nContext of IRs, dialects and passes for Cairo programs …\nCompiles a sierra program into MLIR and then lowers to …\nReturns the argument unchanged.\nInitialize an MLIR context.\nCalls <code>U::from(self)</code>.\nContains the error value\nContains the success value\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nStarknet contract execution result.\nThe result of the JIT execution.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nConvert a [<code>ExecuteResult</code>] to a [<code>NativeExecutionResult</code>]\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nA MLIR JIT execution engine in the context of Cairo Native.\nThe cairo native executor, either AOT or JIT based.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nUtility to convert a <code>NativeModule</code> into an <code>AotNativeExecutor</code>…\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nInvoke the given function by its function id, with the …\nExecute a program with the given params.\nInvoke the given function by its function id, with the …\nExecute a program with the given params.\nInvoke the given function by its function id, with the …\nA libfunc branching target.\nError type returned by this trait’s methods.\nA block within the current libfunc.\nGeneration of MLIR operations from their Sierra …\nHelper struct which contains logic generation for extra …\nA statement’s branch target by its index.\n<code>AP</code> tracking libfuncs\nInserts a new block after all the current libfunc’s …\nArray libfuncs\nBitwise libfuncs\nBoolean libfuncs\nBox libfuncs\nCreates an unconditional branching operation out of the …\nBranch alignment libfunc\nGenerate the MLIR operations.\nBytes31-related libfuncs\nCasting libfuncs\nCreates a conditional binary branching operation, …\nConst libfuncs\nBranch alignment libfunc\nDebug libfuncs\n<code>AP</code> tracking libfuncs\nState value duplication libfunc\nElliptic curve libfuncs\nEnum-related libfuncs\n<code>Felt</code>-related libfuncs\n<code>Felt</code> dictionary libfuncs\n<code>Felt</code> dictionary entry libfuncs\nReturns the argument unchanged.\nReturns the argument unchanged.\nFunction call libfuncs\nGas management libfuncs\nReturn the initialization block.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nReturn the target function if the statement is a function …\nMemory-related libfuncs\nNullable libfuncs\nPedersen hashing libfuncs\nPoseidon hashing libfuncs\n<code>i128</code>-related libfuncs\n<code>i16</code>-related libfuncs\n<code>i32</code>-related libfuncs\n<code>i64</code>-related libfuncs\n<code>i8</code>-related libfuncs\nSnapshot taking libfuncs\nStarknet libfuncs\nStruct-related libfuncs\nCreates a conditional multi-branching operation, …\n<code>u128</code>-related libfuncs\n<code>u16</code>-related libfuncs\n<code>u256</code>-related libfuncs\n<code>u32</code>-related libfuncs\n<code>u512</code>-related libfuncs\n<code>u64</code>-related libfuncs\n<code>u8</code>-related libfuncs\nUnconditional jump libfunc\nNon-zero unwrapping libfuncs\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>disable_ap_tracking</code> …\nGenerate MLIR operations for the <code>enable_ap_tracking</code> …\nGenerate MLIR operations for the <code>revoke_ap_tracking.</code> …\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>array_append</code> libfunc.\nGenerate MLIR operations for the <code>array_get</code> libfunc.\nGenerate MLIR operations for the <code>array_len</code> libfunc.\nGenerate MLIR operations for the <code>array_new</code> libfunc.\nGenerate MLIR operations for the <code>array_pop_front</code> libfunc.\nGenerate MLIR operations for the <code>array_pop_front_consume</code> …\nGenerate MLIR operations for the <code>array_slice</code> libfunc.\nGenerate MLIR operations for the <code>array_snapshot_pop_back</code> …\nGenerate MLIR operations for the <code>array_snapshot_pop_front</code> …\nGenerate MLIR operations for the <code>span_from_tuple</code> libfunc.\nGenerate MLIR operations for the <code>bitwise</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>bool_not_impl</code> libfunc.\nGenerate MLIR operations for the <code>unbox</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>into_box</code> libfunc.\nGenerate MLIR operations for the <code>unbox</code> libfunc.\nGenerate MLIR operations for the <code>branch_align</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>bytes31_const</code> libfunc.\nGenerate MLIR operations for the <code>u8_from_felt252</code> libfunc.\nGenerate MLIR operations for the <code>bytes31_to_felt252</code> …\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>downcast</code> libfunc.\nGenerate MLIR operations for the <code>upcast</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>const_as_box</code> libfunc.\nGenerate MLIR operations for the <code>const_as_immediate</code> …\nGenerate MLIR operations for the <code>coupon</code> libfuncs. In …\nGenerate MLIR operations for the <code>coupon</code> libfunc.\nGenerate MLIR operations for the <code>coupon</code> libfunc.\nGenerate MLIR operations for the <code>drop</code> libfunc.\nGenerate MLIR operations for the <code>dup</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>ec_point_is_zero</code> libfunc.\nGenerate MLIR operations for the <code>ec_neg</code> libfunc.\nGenerate MLIR operations for the <code>ec_point_from_x_nz</code> …\nGenerate MLIR operations for the <code>ec_state_add</code> libfunc.\nGenerate MLIR operations for the <code>ec_state_add_mul</code> libfunc.\nGenerate MLIR operations for the <code>ec_state_try_finalize_nz</code> …\nGenerate MLIR operations for the <code>ec_state_init</code> libfunc.\nGenerate MLIR operations for the <code>ec_point_try_new_nz</code> …\nGenerate MLIR operations for the <code>ec_point_unwrap</code> libfunc.\nGenerate MLIR operations for the <code>ec_point_zero</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>enum_from_bounded_int</code> …\nGenerate MLIR operations for the <code>enum_init</code> libfunc.\nGenerate MLIR operations for the <code>enum_match</code> libfunc.\nGenerate MLIR operations for the <code>enum_snapshot_match</code> …\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the following libfuncs:\nGenerate MLIR operations for the <code>felt252_const</code> libfunc.\nGenerate MLIR operations for the <code>felt252_is_zero</code> libfunc.\nSelect and call the correct libfunc builder function from …\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>function_call</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>withdraw_gas_all</code> libfunc.\nGenerate MLIR operations for the <code>get_builtin_costs</code> libfunc.\nGenerate MLIR operations for the <code>get_builtin_costs</code> libfunc.\nGenerate MLIR operations for the <code>withdraw_gas</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>alloc_local</code> libfunc.\nGenerate MLIR operations for the <code>finalize_locals</code> libfunc.\nGenerate MLIR operations for the <code>rename</code> libfunc.\nGenerate MLIR operations for the <code>store_local</code> libfunc.\nGenerate MLIR operations for the <code>store_temp</code> libfunc.\nSelect and call the correct libfunc builder function from …\nSelect and call the correct libfunc builder function from …\nSelect and call the correct libfunc builder function from …\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>i128_const</code> libfunc.\nGenerate MLIR operations for the <code>i128_diff</code> libfunc.\nGenerate MLIR operations for the <code>i128_eq</code> libfunc.\nGenerate MLIR operations for the <code>i128_from_felt252</code> libfunc.\nGenerate MLIR operations for the <code>i128_is_zero</code> libfunc.\nGenerate MLIR operations for the i128 operation libfunc.\nGenerate MLIR operations for the <code>i128_to_felt252</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>i16_const</code> libfunc.\nGenerate MLIR operations for the <code>i16_diff</code> libfunc.\nGenerate MLIR operations for the <code>i16_eq</code> libfunc.\nGenerate MLIR operations for the <code>i16_from_felt252</code> libfunc.\nGenerate MLIR operations for the <code>i16_is_zero</code> libfunc.\nGenerate MLIR operations for the i16 operation libfunc.\nGenerate MLIR operations for the <code>i16_to_felt252</code> libfunc.\nGenerate MLIR operations for the <code>i16_widemul</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>i32_const</code> libfunc.\nGenerate MLIR operations for the <code>i32_diff</code> libfunc.\nGenerate MLIR operations for the <code>i32_eq</code> libfunc.\nGenerate MLIR operations for the <code>i32_from_felt252</code> libfunc.\nGenerate MLIR operations for the <code>i32_is_zero</code> libfunc.\nGenerate MLIR operations for the i32 operation libfunc.\nGenerate MLIR operations for the <code>i32_to_felt252</code> libfunc.\nGenerate MLIR operations for the <code>i32_widemul</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>i64_const</code> libfunc.\nGenerate MLIR operations for the <code>i64_diff</code> libfunc.\nGenerate MLIR operations for the <code>i64_eq</code> libfunc.\nGenerate MLIR operations for the <code>i64_from_felt252</code> libfunc.\nGenerate MLIR operations for the <code>i64_is_zero</code> libfunc.\nGenerate MLIR operations for the i64 operation libfunc.\nGenerate MLIR operations for the <code>i64_to_felt252</code> libfunc.\nGenerate MLIR operations for the <code>i64_widemul</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>i8_const</code> libfunc.\nGenerate MLIR operations for the <code>i8_diff</code> libfunc.\nGenerate MLIR operations for the <code>i8_eq</code> libfunc.\nGenerate MLIR operations for the <code>i8_from_felt252</code> libfunc.\nGenerate MLIR operations for the <code>i8_is_zero</code> libfunc.\nGenerate MLIR operations for the i8 operation libfunc.\nGenerate MLIR operations for the <code>i8_to_felt252</code> libfunc.\nGenerate MLIR operations for the <code>i8_widemul</code> libfunc.\nGenerate MLIR operations for the <code>snapshot_take</code> libfunc.\nSelect and call the correct libfunc builder function from …\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>struct_construct</code> libfunc.\nGenerate MLIR operations for the <code>struct_deconstruct</code> …\nGenerate MLIR operations for the <code>struct_construct</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>u128_byte_reverse</code> libfunc.\nGenerate MLIR operations for the <code>u128_const</code> libfunc.\nGenerate MLIR operations for the <code>u128_safe_divmod</code> libfunc.\nGenerate MLIR operations for the <code>u128_equal</code> libfunc.\nGenerate MLIR operations for the <code>u128s_from_felt252</code> …\nGenerate MLIR operations for the <code>u128_guarantee_mul</code> …\nGenerate MLIR operations for the <code>u128_guarantee_verify</code> …\nGenerate MLIR operations for the <code>u128_is_zero</code> libfunc.\nGenerate MLIR operations for the <code>u128_add</code> and <code>u128_sub</code> …\nGenerate MLIR operations for the <code>u128_sqrt</code> libfunc.\nGenerate MLIR operations for the <code>u128_to_felt252</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>u16_const</code> libfunc.\nGenerate MLIR operations for the <code>u16_safe_divmod</code> libfunc.\nGenerate MLIR operations for the <code>u16_eq</code> libfunc.\nGenerate MLIR operations for the <code>u16_from_felt252</code> libfunc.\nGenerate MLIR operations for the <code>u16_is_zero</code> libfunc.\nGenerate MLIR operations for the u16 operation libfunc.\nGenerate MLIR operations for the <code>u16_sqrt</code> libfunc.\nGenerate MLIR operations for the <code>u16_to_felt252</code> libfunc.\nGenerate MLIR operations for the <code>u16_widemul</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>u256_safe_divmod</code> libfunc.\nGenerate MLIR operations for the <code>u256_is_zero</code> libfunc.\nGenerate MLIR operations for the <code>u256_sqrt</code> libfunc.\nGenerate MLIR operations for the <code>u256_guarantee_inv_mod_n</code> …\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>u32_const</code> libfunc.\nGenerate MLIR operations for the <code>u32_safe_divmod</code> libfunc.\nGenerate MLIR operations for the <code>u32_eq</code> libfunc.\nGenerate MLIR operations for the <code>u32_from_felt252</code> libfunc.\nGenerate MLIR operations for the <code>u32_is_zero</code> libfunc.\nGenerate MLIR operations for the u32 operation libfunc.\nGenerate MLIR operations for the <code>u32_sqrt</code> libfunc.\nGenerate MLIR operations for the <code>u32_to_felt252</code> libfunc.\nGenerate MLIR operations for the <code>u32_widemul</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>u512_safe_divmod_by_u256</code> …\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>u64_const</code> libfunc.\nGenerate MLIR operations for the <code>u64_safe_divmod</code> libfunc.\nGenerate MLIR operations for the <code>u64_eq</code> libfunc.\nGenerate MLIR operations for the <code>u64_from_felt252</code> libfunc.\nGenerate MLIR operations for the <code>u64_is_zero</code> libfunc.\nGenerate MLIR operations for the u64 operation libfunc.\nGenerate MLIR operations for the <code>u64_sqrt</code> libfunc.\nGenerate MLIR operations for the <code>u64_to_felt252</code> libfunc.\nGenerate MLIR operations for the <code>u64_widemul</code> libfunc.\nSelect and call the correct libfunc builder function from …\nGenerate MLIR operations for the <code>u8_const</code> libfunc.\nGenerate MLIR operations for the <code>u8_safe_divmod</code> libfunc.\nGenerate MLIR operations for the <code>u8_eq</code> libfunc.\nGenerate MLIR operations for the <code>u8_from_felt252</code> libfunc.\nGenerate MLIR operations for the <code>u8_is_zero</code> libfunc.\nGenerate MLIR operations for the u8 operation libfunc.\nGenerate MLIR operations for the <code>u8_sqrt</code> libfunc.\nGenerate MLIR operations for the <code>u8_to_felt252</code> libfunc.\nGenerate MLIR operations for the <code>u8_widemul</code> libfunc.\nGenerate MLIR operations for the <code>jump</code> libfunc.\nGenerate MLIR operations for the <code>unwrap_non_zero</code> libfunc.\nMetadata container.\nDebug utilities\nReturns the argument unchanged.\nRetrieve a reference to some metadata.\nRetrieve a mutable reference to some metadata.\nInsert some metadata and return a mutable reference.\nCalls <code>U::from(self)</code>.\nCreate an empty metadata container.\nFinite field prime modulo\nMemory allocation external bindings\nRemove some metadata and return its last value.\nRuntime library bindings\nTail recursion information\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nPrints the given &amp;str.\nDump a memory region at runtime.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nHolds global gas info.\nError for metadata calculations.\nConfiguration for metadata computation.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the initial value for the gas counter. If …\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nPrime modulo number metadata.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCreate the metadata from the prime number.\nReturn the stored prime number.\nMemory allocation <code>realloc</code> metadata.\nCalls the <code>free</code> function.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nRegister the bindings to the <code>realloc</code> C function and return …\nCalls the <code>realloc</code> function, returns a op with 1 result: an …\nRuntime library bindings metadata.\nRegister if necessary, then invoke the <code>dict_alloc_new()</code> …\nRegister if necessary, then invoke the <code>dict_alloc_new()</code> …\nRegister if necessary, then invoke the <code>dict_gas_refund()</code> …\nRegister if necessary, then invoke the <code>dict_get()</code> function.\nRegister if necessary, then invoke the <code>dict_insert()</code> …\nRegister if necessary, then invoke the <code>dict_clone()</code> …\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nRegister if necessary, then invoke the <code>debug::print()</code> …\nRegister if necessary, then invoke the <code>ec_point_from_x_nz()</code>…\nRegister if necessary, then invoke the …\nRegister if necessary, then invoke the <code>ec_state_add()</code> …\nRegister if necessary, then invoke the <code>ec_state_add_mul()</code> …\nRegister if necessary, then invoke the <code>poseidon()</code> function.\nRegister if necessary, then invoke the <code>pedersen()</code> function.\nRegister if necessary, then invoke the <code>vtable_cheatcode()</code> …\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nThe tail recursion metadata.\nReturn the current depth counter value.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCreate the tail recursion meta.\nReturn the recursion target block.\nReturn the return target block, if set.\nSet the return target block.\nA MLIR module in the context of Cairo Native. It is …\nReturns the argument unchanged.\nRetrieve a reference to some stored metadata.\nInsert some metadata for the program execution and return …\nCalls <code>U::from(self)</code>.\nRemoves metadata\nContains the error value\nBinary representation of a <code>Felt</code> (in MLIR).\nContains the success value\nBinary representation of a <code>u256</code> (in MLIR).\nRuntime function that calls the <code>cheatcode</code> syscall\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nEvent emitted by the emit_event syscall.\nA (somewhat) usable implementation of the starknet syscall …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nError type returned by this trait’s methods.\nGeneration of MLIR types from their Sierra counterparts.\nArray type\nBitwise type\n<code>BoundedInt</code> type\nBox type\nBuild the MLIR type.\nBuiltin costs type\n<code>bytes31</code> type\nCoupon type.\nElliptic curve operation type\nElliptic curve point type\nElliptic curve state type\nEnum type\n<code>felt252</code> type\n<code>Felt</code> dictionary type\n<code>Felt</code> dictionary entry type\nReturns the argument unchanged.\nGas builtin type\nIf the type is an integer type, return its width in bits.\nCalls <code>U::from(self)</code>.\nReturn whether the type is a builtin.\nReturn whether the type requires a return pointer when …\nIf the type is an integer type, return if its signed.\nWhether the layout should be allocated in memory (either …\nReturn whether the Sierra type resolves to a zero-sized …\nGenerate the layout of the MLIR type.\nNon-zero type\nNullable type\nPedersen type\nPoseidon type\nBuiltin costs type\nSegment arena type\nSnapshot type\nSquashed <code>Felt</code> dictionary type\nStarknet types\nStruct type\nUnsigned 128-bit integer type\nUnsigned 128-bit multiplication guarantee type\nUnsigned 16-bit integer type\nUnsigned 32-bit integer type\nUnsigned 64-bit integer type\nUnsigned 8-bit integer type\nUninitialized type\nIf the type is a enum type, return all possible variants.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nAn MLIR type with its memory layout.\nBuild the MLIR type.\nExtract layout for the default enum representation, its …\nExtract the type and layout for the default enum …\nThe <code>felt252</code> prime modulo.\nBuild the MLIR type.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nBuild the MLIR type.\nCompile a cairo program found at the given path to sierra.\nCreates the execution engine, with all symbols registered.\nReturn a type that calls a closure when formatted using …\nParse any type that can be a bigint to a felt that can be …\nParse a short string into a felt that can be used in the …\nParse a numeric string into felt, wrapping negatives …\nReturns the given entry point if present.\nReturns the given entry point if present.\nGiven a string representing a function name, searches in …\nReturns the argument unchanged.\nGenerate a function name.\nReturn the layout for an integer of arbitrary width.\nCalls <code>U::from(self)</code>.\nCopied from std.\nEdit: Copied from the std lib.\nall elements need to be same type\nA JitValue is a value that can be passed to the JIT engine …\nUsed as return value for Nullables that are null.\nString to felt\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.")