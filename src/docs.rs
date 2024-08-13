//! # Cairo Native Compiler and Execution Engine

#[cfg_attr(doc, aquamarine::aquamarine)]
/// # Overview
///
/// This crate is a compiler and JIT engine that transforms Sierra (or Cairo) sources into MLIR,
/// which can be [JIT-executed](https://en.wikipedia.org/wiki/Just-in-time_compilation) or further
/// compiled into a binary
/// [ahead of time](https://en.wikipedia.org/wiki/Ahead-of-time_compilation).
///
/// ## Common definitions
/// Within this project there are lots of functions with the same signature. As their arguments have
/// all the same meaning, they are documented here:
///
/// - `context: NativeContext`: The MLIR context.
/// - `module: &NativeModule`: The compiled MLIR program, with other relevant information such as program registry and metadata.
/// - `program: &Program`: The Sierra input program.
/// - `registry: &ProgramRegistry<TType, TLibfunc>`: The registry extracted from the program.
/// - `metadata: &mut MetadataStorage`: Current compiler metadata.
///
/// ## Project layout
/// The code is laid out in the following sections:
///
/// ```txt
///  src
///  ├─ context.rs - The MLIR context wrapper, provides the compile method.
///  ├─ utils.rs - Internal utilities.
///  ├─ metadata/ - Metadata injector to use within the compilation process
///  ├─ executor/ - Code related to the executor of programs.
///  ├─ module.rs - The MLIR module wrapper.
///  ├─ arch/ - Trampoline assembly for calling functions with dynamic signatures.
///  ├─ executor.rs - The executor code.
///  ├─ ffi.cpp - Missing FFI C wrappers
///  ├─ libfuncs - Cairo Sierra libfunc implementations
///  ├─ libfuncs.rs - Cairo Sierra libfunc glue code
///  ├─ starknet.rs - Starknet syscall handler glue code.
///  ├─ ffi.rs - Missing FFI C wrappers, rust side.
///  ├─ block_ext.rs - A melior (MLIR) block trait extension to write less code.
///  ├─ lib.rs - The main lib file.
///  ├─ execution_result.rs - Program result parsing.
///  ├─ values.rs - JIT serialization.
///  ├─ metadata.rs - Metadata injector to use within the compilation process.
///  ├─ compiler.rs - The glue code of the compiler, has the codegen for the function signatures
///  and calls the libfunc codegen implementations.
///  ├─ error.rs - Error handling
///  ├─ bin - Binary programs
///  ├─ types - Cairo to MLIR type information
/// ```
///
/// ### Library functions
/// Path: `src/libfuncs`
///
/// Here are stored all the library function implementations in MLIR, this contains the majority of the code.
///
/// To store information about the different types of library functions sierra has, we divide them into the following using the enum `SierraLibFunc`:
/// - **Branching**: These functions are implemented inline, adding blocks and jumping as necessary based on given conditions.
/// - **Constant**: A constant value, this isn't represented as a function and is inserted inline.
/// - **Function**: Any other function.
/// - **InlineDataFlow**: Functions that can be implemented inline without much problem. For example: dup, store_temp
///
/// ### Statements
/// Path: `src/statements`
///
/// Here is the code that processes the statements of non-library functions. It handles dataflow, branching, function calls, variable storage and also has implementations for the inline library functions.
///
/// ### User functions
/// These are extra utility functions unrelated to sierra that aid in the development, such as wrapping return values and printing them.
///
/// ## Basic API usage example
///
/// The API contains two structs, `NativeContext` and `NativeExecutor`.
/// The main purpose of `NativeContext` is MLIR initialization, compilation and lowering to LLVM.
/// `NativeExecutor` in the other hand is responsible of executing MLIR compiled sierra programs
/// from an entrypoint.
/// Programs and JIT states can be cached in contexts where their execution will be done multiple
/// times.
///
/// ```rust,noexecute
/// use starknet_types_core::felt::Felt;
/// use cairo_native::context::NativeContext;
/// use cairo_native::executor::JitNativeExecutor;
/// use cairo_native::values::JitValue;
/// use std::path::Path;
///
/// let program_path = Path::new("programs/examples/hello.cairo");
/// // Compile the cairo program to sierra.
/// let sierra_program = cairo_native::utils::cairo_to_sierra(program_path);
///
/// // Instantiate a Cairo Native MLIR context. This data structure is responsible for the MLIR
/// // initialization and compilation of sierra programs into a MLIR module.
/// let native_context = NativeContext::new();
///
/// // Compile the sierra program into a MLIR module.
/// let native_program = native_context.compile(&sierra_program, None).unwrap();
///
/// // The parameters of the entry point.
/// let params = &[JitValue::Felt252(Felt::from_bytes_be_slice(b"user"))];
///
/// // Find the entry point id by its name.
/// let entry_point = "hello::hello::greet";
/// let entry_point_id = cairo_native::utils::find_function_id(&sierra_program, entry_point);
///
/// // Instantiate the executor.
/// let native_executor = JitNativeExecutor::from_native_module(native_program, Default::default());
///
/// // Execute the program.
/// let result = native_executor
///     .invoke_dynamic(entry_point_id, params, None)
///     .unwrap();
///
/// println!("Cairo program was compiled and executed successfully.");
/// println!("{:?}", result);
/// ```
///
/// ## Running a Cairo program
///
/// This is a usage example using the API for an easy Cairo program that requires the least setup to get running. It allows you to compile and execute a program using the JIT.
///
/// Example code to run a program:
///
/// ```rust,noexecute
/// use starknet_types_core::felt::Felt;
/// use cairo_native::context::NativeContext;
/// use cairo_native::executor::NativeExecutor;
/// use cairo_native::values::JitValue;
/// use std::path::Path;
///
/// fn main() {
///     let program_path = Path::new("programs/examples/hello.cairo");
///     // Compile the cairo program to sierra.
///     let sierra_program = cairo_native::utils::cairo_to_sierra(program_path);
///
///     // Instantiate a Cairo Native MLIR context. This data structure is responsible for the MLIR
///     // initialization and compilation of sierra programs into a MLIR module.
///     let native_context = NativeContext::new();
///
///     // Compile the sierra program into a MLIR module.
///     let native_program = native_context.compile(&sierra_program).unwrap();
///
///     // The parameters of the entry point.
///     let params = &[JitValue::Felt252(Felt::from_bytes_be_slice(b"user"))];
///
///     // Find the entry point id by its name.
///     let entry_point = "hello::hello::greet";
///     let entry_point_id = cairo_native::utils::find_function_id(&sierra_program, entry_point);
///
///     // Instantiate the executor.
///     let native_executor = NativeExecutor::new(native_program);
///
///     // Execute the program.
///     let result = native_executor
///         .execute(entry_point_id, params, None)
///         .unwrap();
///
///     println!("Cairo program was compiled and executed successfully.");
///     println!("{:?}", result);
/// }
/// ```
///
/// ## Running a Starknet contract
///
/// Example code to run a Starknet contract:
///
/// ```rust,noexecute
/// use starknet_types_core::felt::Felt;
/// use cairo_lang_compiler::CompilerConfig;
/// use cairo_lang_starknet::contract_class::compile_path;
/// use cairo_native::context::NativeContext;
/// use cairo_native::executor::NativeExecutor;
/// use cairo_native::utils::find_entry_point_by_idx;
/// use cairo_native::values::JitValue;
/// use cairo_native::{
///     metadata::syscall_handler::SyscallHandlerMeta,
///     starknet::{BlockInfo, ExecutionInfo, StarkNetSyscallHandler, SyscallResult, TxInfo, U256},
/// };
/// use std::path::Path;
///
/// /// To run a starknet contract, we need to use a syscall handler, here we show how to implement one (at the end).
/// #[derive(Debug)]
/// struct SyscallHandler;
///
/// fn main() {
///     let path = Path::new("programs/examples/hello_starknet.cairo");
///
///     let contract = compile_path(
///         path,
///         None,
///         CompilerConfig {
///             replace_ids: true,
///             ..Default::default()
///         },
///     )
///     .unwrap();
///
///     let entry_point = contract.entry_points_by_type.constructor.get(0).unwrap();
///     let sierra_program = contract.extract_sierra_program().unwrap();
///
///     let native_context = NativeContext::new();
///
///     let mut native_program = native_context.compile(&sierra_program).unwrap();
///     native_program
///         .insert_metadata(SyscallHandlerMeta::new(&mut SyscallHandler))
///         .unwrap();
///
///     // Call the echo function from the contract using the generated wrapper.
///     let entry_point_fn =
///         find_entry_point_by_idx(&sierra_program, entry_point.function_idx).unwrap();
///
///     let fn_id = &entry_point_fn.id;
///
///     let native_executor = NativeExecutor::new(native_program);
///
///     let result = native_executor
///         .execute_contract(
///             fn_id,
///             // The calldata
///             &[JitValue::Felt252(Felt::ONE)],
///             u64::MAX.into(),
///         )
///         .expect("failed to execute the given contract");
///
///     println!();
///     println!("Cairo program was compiled and executed successfully.");
///     println!("{result:#?}");
/// }
///
/// // Implement an example syscall handler.
/// impl StarkNetSyscallHandler for SyscallHandler {
///     fn get_block_hash(
///         &mut self,
///         block_number: u64,
///         _gas: &mut u128,
///     ) -> SyscallResult<Felt> {
///         println!("Called `get_block_hash({block_number})` from MLIR.");
///         Ok(Felt::from_bytes_be_slice(b"get_block_hash ok"))
///     }
///
///     fn get_execution_info(
///         &mut self,
///         _gas: &mut u128,
///     ) -> SyscallResult<cairo_native::starknet::ExecutionInfo> {
///         println!("Called `get_execution_info()` from MLIR.");
///         Ok(ExecutionInfo {
///             block_info: BlockInfo {
///                 block_number: 1234,
///                 block_timestamp: 2345,
///                 sequencer_address: 3456.into(),
///             },
///             tx_info: TxInfo {
///                 version: 4567.into(),
///                 account_contract_address: 5678.into(),
///                 max_fee: 6789,
///                 signature: vec![1248.into(), 2486.into()],
///                 transaction_hash: 9876.into(),
///                 chain_id: 8765.into(),
///                 nonce: 7654.into(),
///             },
///             caller_address: 6543.into(),
///             contract_address: 5432.into(),
///             entry_point_selector: 4321.into(),
///         })
///     }
///
///     fn deploy(
///         &mut self,
///         class_hash: Felt,
///         contract_address_salt: Felt,
///         calldata: &[Felt],
///         deploy_from_zero: bool,
///         _gas: &mut u128,
///     ) -> SyscallResult<(Felt, Vec<Felt>)> {
///         println!("Called `deploy({class_hash}, {contract_address_salt}, {calldata:?}, {deploy_from_zero})` from MLIR.");
///         Ok((
///             class_hash + contract_address_salt,
///             calldata.iter().map(|x| x + &Felt::ONE).collect(),
///         ))
///     }
///
///     fn replace_class(
///         &mut self,
///         class_hash: Felt,
///         _gas: &mut u128,
///     ) -> SyscallResult<()> {
///         println!("Called `replace_class({class_hash})` from MLIR.");
///         Ok(())
///     }
///
///     fn library_call(
///         &mut self,
///         class_hash: Felt,
///         function_selector: Felt,
///         calldata: &[Felt],
///         _gas: &mut u128,
///     ) -> SyscallResult<Vec<Felt>> {
///         println!(
///             "Called `library_call({class_hash}, {function_selector}, {calldata:?})` from MLIR."
///         );
///         Ok(calldata.iter().map(|x| x * Felt::from(3)).collect())
///     }
///
///     fn call_contract(
///         &mut self,
///         address: Felt,
///         entry_point_selector: Felt,
///         calldata: &[Felt],
///         _gas: &mut u128,
///     ) -> SyscallResult<Vec<Felt>> {
///         println!(
///             "Called `call_contract({address}, {entry_point_selector}, {calldata:?})` from MLIR."
///         );
///         Ok(calldata.iter().map(|x| x * Felt::from(3)).collect())
///     }
///
///     fn storage_read(
///         &mut self,
///         address_domain: u32,
///         address: Felt,
///         _gas: &mut u128,
///     ) -> SyscallResult<Felt> {
///         println!("Called `storage_read({address_domain}, {address})` from MLIR.");
///         Ok(address * Felt::from(3))
///     }
///
///     fn storage_write(
///         &mut self,
///         address_domain: u32,
///         address: Felt,
///         value: Felt,
///         _gas: &mut u128,
///     ) -> SyscallResult<()> {
///         println!("Called `storage_write({address_domain}, {address}, {value})` from MLIR.");
///         Ok(())
///     }
///
///     fn emit_event(
///         &mut self,
///         keys: &[Felt],
///         data: &[Felt],
///         _gas: &mut u128,
///     ) -> SyscallResult<()> {
///         println!("Called `emit_event({keys:?}, {data:?})` from MLIR.");
///         Ok(())
///     }
///
///     fn send_message_to_l1(
///         &mut self,
///         to_address: Felt,
///         payload: &[Felt],
///         _gas: &mut u128,
///     ) -> SyscallResult<()> {
///         println!("Called `send_message_to_l1({to_address}, {payload:?})` from MLIR.");
///         Ok(())
///     }
///
///     fn keccak(
///         &mut self,
///         input: &[u64],
///         _gas: &mut u128,
///     ) -> SyscallResult<cairo_native::starknet::U256> {
///         println!("Called `keccak({input:?})` from MLIR.");
///         Ok(U256(Felt::from(1234567890).to_le_bytes()))
///     }
///
///     /*
///     ... more code here, check out the full example in examples/starknet.rsd
///     */
/// }
///
/// ```
///
/// For more examples, check out the `examples/` directory.
pub mod section01 {}

#[cfg_attr(doc, aquamarine::aquamarine)]
/// # Compilation Walkthrough
/// This section describes the entire process Cairo Native goes through to
/// compile a Cairo program to either a shared library (and how to use it) or a
/// MLIR module for use in the JIT engine.
///
/// ## General flow
/// If you check `lib.rs` you will see the high level modules of the project.
///
/// The compiler module is what glues together everything.
/// You should read its module level documentation.
/// But the basic flow is like this:
/// - We take a sierra `Program` and iterate over its functions.
/// - On each function, we create a MLIR region and a block for each statement (a.k.a library function call), taking into account possible branches.
/// - On each statement we call the library function implementation, which appends MLIR code to the given block, and with helper methods, it handles possible branches and input/output variables.
///
/// ```mermaid
/// stateDiagram-v2
///     state "Load sierra program" as sierra
///     state "Initialize compiler" as init
///     state "Initialize execution engine" as engine
///     state if_skip_jit <<choice>>
///     state "Load MLIR dialects" as dialects
///     state "Create builtin module" as module
///     state "Create libc wrappers" as libc
///     state "Process Types" as types
///     state "Process Library functions" as libfuncs
///     state "Save non-flow function info" as func_info
///     state "Process functions" as funcs
///     state "Calculate block ranges per function" as blocks
///     state "Process statements" as statements
///     state "Apply MLIR passes" as passes
///     [*] --> Initialize
///     state Initialize {
///         sierra --> init
///         init --> if_skip_jit
///         if_skip_jit --> engine: if JIT
///         if_skip_jit --> dialects: if Compile
///         engine --> dialects
///     }
///     Initialize --> Compile
///     state Compile {
///         module --> libc
///         libc --> types
///         types --> libfuncs
///         types --> func_info
///         func_info --> libfuncs
///         libfuncs --> funcs
///         funcs --> blocks
///         blocks --> statements
///     }
///     Compile --> passes
///     passes --> Output
///     Output --> [*]
/// ```
///
/// ## Loading a Cairo Program
/// The first step is to get the sierra code from the given cairo program, this
/// is done using the relevant methods from the `cairo_lang_compiler` crate.
///
/// This gives us a `cairo_lang_sierra::program::Program` which has the following
/// structure:
///
/// ```rust,noexecute
/// pub struct Program {
///     pub type_declarations: Vec<TypeDeclaration, Global>,
///     pub libfunc_declarations: Vec<LibfuncDeclaration, Global>,
///     pub statements: Vec<GenStatement<StatementIdx>, Global>,
///     pub funcs: Vec<GenFunction<StatementIdx>, Global>,
/// }
/// ```
///
/// The compilation process consists in parsing these fields to produce the
/// relevant MLIR IR code.
///
/// To do all this we will need a MLIR Context and a module created with that
/// context, the module describes a compilation unit, in this case, the cairo
/// program.
///
/// ## Initialization
///
/// In Cairo Native we provide a API around initializing the context, namely
/// `NativeContext` which does the following when
/// [created](https://github.com/lambdaclass/cairo_native/blob/ca6549a68c1b4266a7f9ea41dc196bf4433a2ee8/src/context.rs#L52-L53):
///
/// - Create the context
/// - Register all relevant MLIR dialects into the context
/// - Load the dialects
/// - Register all passes into the context
/// - Register all translations to LLVM IR into the context.
///
/// <aside>
/// 💡 Registering doesn’t mean using, it means that later in the compilation
/// process we will use these registered features, such as the translations to
/// LLVM IR to create a shared library.
/// </aside>
///
/// ## Compiling a Sierra Program to MLIR
///
/// The `NativeContext` has a method called
/// [compile](https://github.com/lambdaclass/cairo_native/blob/ca6549a68c1b4266a7f9ea41dc196bf4433a2ee8/src/context.rs#L62-L63),
/// which does the heavy lifting and returns a `NativeModule`.
/// This module contains the generated MLIR IR code.
///
/// The compile method does the following:
/// - Create a Module
/// - Create the Metadata storage (check the relevant section for more information).
/// - Check if the Sierra program has a gas builtin in it, if it has it will
///   insert the gas metadata into the storage.
/// - Create the Sierra program registry, which allows type and function lookups.
/// - Call a internal `compile` method.
///
/// This internal `compile` method then loops on the program function
/// declarations calling the `compile_func` method on each of them.
///
/// ### Compiling a function (`compile_func`)
///
/// This method generates the structure of the function in MLIR, meaning it will
/// create the region the body of the function will live on, and then a block
/// for each statement, each with it’s relevant arguments and return values. It
/// will also check each statement whether it is branching, and store the
/// predecessors of each block, to handle jumps.
///
/// While handling each statement on the function, it will build the types it
/// finds from the arguments and return values as it encounters them, this is
/// done using the trait `TypeBuilder`.
///
/// After having the function structure created, we proceed to creating the
/// initial state, which is a Hash map holding the local variables we currently
/// have, the parameters.
///
/// Using this initial state, it builds the entry block, which is the first
/// block the function enters when it’s called, it has the function arguments
/// as parameters.
///
/// Then it loops on the statements of the function, on each statement it does
/// the following:
///
/// - Check if there is a gas metadata, and if the statement has a gas cost,
///   insert the gas cost metadata that lives on only during this statement.
/// - Get the block and possible landing block of this statement.
/// - If there is a landing block, create it. A landing block is the target
///   block of a previous jump that simply forwards to the current block.
///
/// ## Metadata Storage
/// This storage is shared everywhere in the compilation process and allows to
/// easily share data to the relevant places, for example the Gas Metadata
/// allows getting the gas cost for a given statement, or the enum snapshot
/// metadata to get the relevant variants in the libfunc builder.
///
/// # Compiling to native code
///
/// We part from the point where the program has been compiled into MLIR IR,
/// and we hold the MLIR Context and Module.
///
/// From this point, we convert all the dialects within this IR into the LLVM
/// MLIR Dialect, which is a needed precondition to transform the MLIR IR into
/// LLVM IR. This is done through passes which are the basis of how LLVM works.
///
/// <aside>
/// ℹ️ This translation does canonicalization, so some optimizations are done.
/// </aside>
///
/// Given a MLIR Module with only the LLVM dialect, we can translate it,
/// currently the LLVM MLIR API for this is only available in C++, so we had
/// to make our temporary C API wrapper (which we contributed to upstream LLVM,
/// coming soon to LLVM 18 maybe). After that we also need to use the `llvm-sys`
/// crate which provides the C API bindings in Rust.
///
/// The required method is `mlirTranslateModuleToLLVMIR` which takes a MLIR
/// Module and a LLVM Context (not a MLIR one!). The LLVM Context will be used
/// to create a LLVM Module, which we can then compile to machine code.
///
/// The process is a bit verbose but interesting, LLVM itself is a target
/// independent code generator, but to compile down we need an actual target,
/// to do so we initialize the required target and utilities (in this case we
/// initialize all targets the current compiled LLVM supports):
///
/// ```rust,noexecute
/// LLVM_InitializeAllTargets();
/// LLVM_InitializeAllTargetInfos();
/// LLVM_InitializeAllTargetMCs();
/// LLVM_InitializeAllAsmPrinters();
/// LLVM_InitializeAllAsmParsers();
/// ```
///
/// After that we create a LLVM context, and pass it along the module to the
/// `mlirTranslateModuleToLLVMIR` method:
///
/// ```rust,noexecute
/// let llvm_module = mlirTranslateModuleToLLVMIR(mlir_module_op, llvm_context);
/// ```
///
/// Then we need to create the target machine, which needs a target triple, the
/// CPU name and CPU features. After creating the target machine, we can emit
/// the object file either to a memory buffer or a file.
///
/// ```rust,noexecute
/// let machine = LLVMCreateTargetMachine(
///             target,
///             target_triple.cast(),
///             target_cpu.cast(),
///             target_cpu_features.cast(),
///             LLVMCodeGenOptLevel::LLVMCodeGenLevelNone, // opt level
///             LLVMRelocMode::LLVMRelocDynamicNoPic,
///             LLVMCodeModel::LLVMCodeModelDefault,
/// );
///
/// let mut out_buf: MaybeUninit<LLVMMemoryBufferRef> = MaybeUninit::uninit();
///
/// LLVMTargetMachineEmitToMemoryBuffer(
///             machine,
///             llvm_module,
///             LLVMCodeGenFileType::LLVMObjectFile,
///             error_buffer,
///             out_buf.as_mut_ptr(),
/// );
/// ```
///
/// After emitting the object file, we need to pass it to a linker to get our
/// shared library. This is currently done by executing `ld`, with the proper
/// flags to create a shared library on each platform, as a process using a
/// temporary file, because it can’t be piped.
///
/// ```mermaid
/// graph TD
///   A[MLIR Module using all available dialects]
/// 		--> |Passes| B[Canonicalized MLIR module using only the LLVM dialect]
/// 	B --> BB[mlirTranslateModuleToLLVMIR]
/// 	BB --> C[LLVM IR Module]
/// 	D[Create LLVM Context] --> BB
/// 	C --> E[Emit Binary Object]
/// 	F[Create Target LLVM Machine] --> E
/// 	E --> |Link the object file| G[Shared Library]
/// ```
///
/// ## Loading the native library and using it
/// To load the library, we use the crate `libloading`, passing it the path to
/// our shared library.
///
/// Then we initialize the `AotNativeExecutor` with the loaded library and the
/// program registry.
///
/// This initialization internally does the following:
/// - Constructs the symbol of the function to be called, which is always the
///   function name but wrapped with a prefix to instead target the C API
///   wrapper, it looks like the following:
///
/// ```rust,noexecute
/// let function_name = format!("_mlir_ciface_{function_name}");
/// ```
///
/// - Using the registry we get the function signature, although `libloading`
///   allows us to have a function signature at compile time to make sure we
///   call it properly, but we need to ignore this as we want to call any
///   function given the library and the registry.
///
/// <aside>
/// 🚧 TODO: Explain how we call the function here, the custom assembly trampoline, etc
/// </aside>
///
/// ```mermaid
/// graph TD
///   A[Load library with libloading] --> B[Get the function pointer]
/// 	C[Generate the target symbol's name] --> B
/// 	D[Extract the function signature from the program's registry] --> E[Setup the arguments]
/// 	E --> F[Call the trampoline function]
/// 	B --> F
/// 	F --> G[Interpret the results]
/// 	G --> H[Cleanup and deallocate]
/// ```
///
/// ## Addendum
///
/// ### About canonicalization in MLIR:
/// MLIR has a single canonicalization pass, which iteratively applies the
/// canonicalization patterns of all loaded dialects in a greedy way.
/// Canonicalization is best-effort and not guaranteed to bring the entire IR in
/// a canonical form. It applies patterns until either fix point is reached or
/// the maximum number of iterations/rewrites (as specified via pass options) is
/// exhausted. This is for efficiency reasons and to ensure that faulty patterns
/// cannot cause infinite looping.
///
/// Good read about this: [https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html](https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html)
pub mod section02 {}

#[cfg_attr(doc, aquamarine::aquamarine)]
/// # Execution Walkthrough
/// 🚧 TODO
/// Given the following Cairo program:
///
/// ```rust,noexecute
/// // This is the cairo program. It just adds two numbers together and returns the
/// // result in an enum whose variant is selected using the result's parity.
/// enum Parity<T> {
///   Even: T,
/// Odd: T, }
/// /// Add `lhs` and `rhs` together and return the result in `Parity::Even` if it's
/// /// even or `Parity::Odd` otherwise.
/// fn run(lhs: u128, rhs: u128) -> Parity<u128> {
///   let res = lhs + rhs;
///   if (res & 1) == 0 {
///     Parity::Even(res)
///   } else {
///     Parity::Odd(res)
/// } }
/// ```
pub mod section03 {}

#[cfg_attr(doc, aquamarine::aquamarine)]
/// # Gas and Builtins Accounting
///
/// This section documents how programs generated by Cairo Native keep track of
/// gas and builtins during execution.
///
/// ## Gas
///
/// ### Introduction
/// Gas management in a blockchain environment involves accounting for the amount
/// of computation performed during the execution of a transaction. This is used
/// to accurately charge the user at the end of the execution or to revert early
/// if the transaction consumes more gas than provided by the sender.
///
/// This documentation assumes prior knowledge about Sierra and about the way
/// gas accounting is performed in Sierra. For those seeking to deepen their
/// understanding, refer to Enitrat’s Medium post about
/// [Sierra](https://medium.com/nethermind-eth/under-the-hood-of-cairo-1-0-exploring-sierra-7f32808421f5)
/// and greged’s about
/// [gas accounting in Sierra](https://blog.kakarot.org/understanding-sierra-gas-accounting-19d6141d28b9).
///
/// ### Gas builtin
/// The gas builtin is used in Sierra in order to perform gas accounting. It is
/// passed as an input to all function calls and holds the current remaining
/// gas. It is represented in MLIR by a simple `u128`.
///
/// ### Gas metadata
/// The process of calculating gas begins at the very outset of the compilation
/// process. During the initial setup of the Sierra program, metadata about the
/// program, including gas information, is extracted. Using gas helper functions
/// from the [Cairo compiler](https://github.com/starkware-libs/cairo/tree/main),
/// the consumed cost (steps, memory holes, builtins usage) for each statement
/// in the Sierra code is stored in a HashMap.
///
/// ### Withdrawing gas
/// The action of withdrawing gas can be split in two steps:
///
/// - **Calculating Total Gas Cost**: Using the previously constructed HashMap,
///   we iterate over the various cost tokens (including steps, built-in usage,
///   and memory holes) for the statement, convert them into a
///   [common gas unit](https://github.com/starkware-libs/cairo/blob/v2.7.0-dev.0/crates/cairo-lang-runner/src/lib.rs#L136),
///   and sum them up to get the total gas cost for the statement.
/// - **Executing Gas Withdrawal**: The previously calculated gas cost is used
///   when the current statement is a `withdraw_gas` libfunc call.
///
/// The `withdraw_gas` libfunc takes the current leftover gas as input and uses
/// the calculated gas cost for the statement to deduct the appropriate amount
/// from the gas builtin. In the compiled IR, gas withdrawal appears as the
/// total gas being reduced by a predefined constant. Additionally, the libfunc
/// branches based on whether the remaining gas is greater than or equal to the
/// amount being withdrawn.
///
/// ### Example
/// Let's illustrate this with a simple example using the following Cairo 1 code:
///
/// ```rust,noexecute
/// fn run_test() {
///     let mut i: u8 = 0;
///     let mut val = 0;
///     while i < 5 {
///         val = val + i;
///         i = i + 1;
///     }
/// }
/// ```
///
/// As noted earlier, gas usage is initially computed by the Cairo compiler for
/// each state. A snippet of the resulting HashMap shows the cost for each
/// statement:
///
/// ```json
/// ...
/// (
///     StatementIdx(26),
///     Const,
/// ): 2680,
/// (
///     StatementIdx(26),
///     Pedersen,
/// ): 0,
/// (
///     StatementIdx(26),
///     Poseidon,
/// ): 0,
/// ...
/// ```
///
/// For statement 26, the cost of the `Const` token type (a combination of step,
/// memory hole, and range check costs) is 2680, while other costs are 0.
/// Let's see which libfunc is called at statement 26:
///
/// ```assembly
/// ...
/// disable_ap_tracking() -> (); // 25
/// withdraw_gas([0], [1]) { fallthrough([4], [5]) 84([6], [7]) }; // 26
/// branch_align() -> (); // 27
/// const_as_immediate<Const<u8, 5>>() -> ([8]); // 28
/// ...
/// ```
///
/// When the Cairo native compiler reaches statement 26, it combines all costs
/// into gas using the Cairo compiler code. In this example, the total cost is
/// 2680 gas. This value is used in the `withdraw_gas` libfunc and the compiled
/// corresponding IR to withdraw the gas and determine whether execution should
/// revert or continue. This can be observed in the following MLIR dump:
///
/// ```assembly
/// llvm.func @"test::test::run_test[expr16](f0)"(%arg0: i64 loc(unknown), %arg1: i128 loc(unknown), %arg2: i8 loc(unknown), %arg3: i8 loc(unknown)) -> !llvm.struct<(i64, i128, struct<(i64, array<24 x i8>)>)> attributes {llvm.emit_c_interface} {
///   ...
///   %12 = llvm.mlir.constant(5 : i8) : i8 loc(#loc1)
///   %13 = llvm.mlir.constant(2680 : i128) : i128 loc(#loc1)
///   %14 = llvm.mlir.constant(1 : i64) : i64 loc(#loc1)
///   ...
///   ^bb1(%27: i64 loc(unknown), %28: i128 loc(unknown), %29: i8 loc(unknown), %30: i8 loc(unknown)):  // 2 preds: ^bb0, ^bb6
///     %31 = llvm.add %27, %14  : i64 loc(#loc13)
///     %32 = llvm.icmp "uge" %28, %13 : i128 loc(#loc13)
///     %33 = llvm.intr.usub.sat(%28, %13)  : (i128, i128) -> i128 loc(#loc13)
///     llvm.cond_br %32, ^bb2(%29 : i8), ^bb7(%5, %23, %23, %31 : i252, !llvm.ptr, !llvm.ptr, i64) loc(#loc13)
///     ...
/// ```
///
/// Here, we see the constant `2680` defined at the begining of the function.
/// In basic block 1, the withdraw_gas operations are performed: by comparing
/// `%28` (remaining gas) and `%13` (gas cost), the result stored in `%32`
/// determines the conditional branching. A saturating subtraction between the
/// remaining gas and the gas cost is then performed, updating the remaining gas
/// in the IR.
///
/// ### Final gas usage
/// The final gas usage can be easily retrieved from the gas builtin value
/// returned by the function. This is accomplished when
/// [parsing the return values](https://github.com/lambdaclass/cairo_native/blob/65face8194054b7ed396a34a60e7b1595197543a/src/executor.rs#L286)
/// from the function call:
///
/// ```rust,noexecute
/// ...
/// for type_id in &function_signature.ret_types {
///     let type_info = registry.get_type(type_id).unwrap();
///     match type_info {
///         CoreTypeConcrete::GasBuiltin(_) => {
///             remaining_gas = Some(match &mut return_ptr {
///                 Some(return_ptr) => unsafe { *read_value::<u128>(return_ptr) },
///                 None => {
///                     // If there's no return ptr then the function only returned the gas. We don't
///                     // need to bother with the syscall handler builtin.
///                     ((ret_registers[1] as u128) << 64) | ret_registers[0] as u128
///                 }
///             });
///         }
///         ...
///     }
///     ...
/// }
/// ...
/// ```
///
/// This code snippet extracts the remaining gas from the return pointer based
/// on the function's signature. If the function only returns the gas value,
/// the absence of a return pointer is handled appropriately, ensuring accurate
/// gas accounting.
///
/// ## Builtins Counter
///
/// ### Introduction
/// The Cairo Native compiler records the usage of each builtins in order to
/// provide information about the program's builtins consumption.
/// This information is NOT used for the gas calculation, as the gas cost of
/// builtins is already taken into account during the [gas accounting process](./gas.md).
/// The builtins counter types can each be found in the [types folder](../src/types/).
/// Taking the [Pedersen hash](../src/types/pedersen.rs) as an example, we see
/// that the counters will be represented as i64 integers in MLIR.
/// Counters are then simply incremented by one each time the builtins are
/// called from within the program.
///
/// ### Example
/// Let us consider the following Cairo program which uses the `pedersen` builtin:
///
/// ```rust,noexecute
/// use core::integer::bitwise;
/// use core::pedersen::pedersen;
///
/// fn run_test() {
///     let mut hash = pedersen(1.into(), 2.into());
///     hash += 1;
/// }
/// ```
///
/// We expect Native to increment the `pedersen` counter by 1 given the above code.
/// Let's first check how this compiles to Sierra:
///
/// ```sierra
/// const_as_immediate<Const<felt252, 1>>() -> ([1]); // 0
/// const_as_immediate<Const<felt252, 2>>() -> ([2]); // 1
/// store_temp<felt252>([1]) -> ([1]); // 2
/// store_temp<felt252>([2]) -> ([2]); // 3
/// pedersen([0], [1], [2]) -> ([3], [4]); // 4
/// drop<felt252>([4]) -> (); // 5
/// store_temp<Pedersen>([3]) -> ([3]); // 6
/// return([3]); // 7
///
/// contracts::run_test@0([0]: Pedersen) -> (Pedersen);
/// ```
///
/// In the compiled Sierra, we can see that the `pedersen` builtin is passed
/// with the call to the `run_test` which starts at statement `0`. It is then
/// used in the call to the `pedersen` libfunc. We would expect to see the
/// `pedersen` counter incremented by 1 in the Native compiler. Below is the
/// compiled MLIR dump for the same program:
///
/// ```assembly
/// ...
/// llvm.func @"test::test::run_test(f0)"(%arg0: i64 loc(unknown)) -> i64 attributes {llvm.emit_c_interface} {
///     %0 = llvm.mlir.constant(2 : i256) : i256 loc(#loc1)
///     %1 = llvm.mlir.constant(1 : i256) : i256 loc(#loc1)
///     %2 = llvm.mlir.constant(1 : i64) : i64 loc(#loc1)
///     %3 = llvm.alloca %2 x i256 {alignment = 16 : i64} : (i64) -> !llvm.ptr loc(#loc2)
///     %4 = llvm.alloca %2 x i256 {alignment = 16 : i64} : (i64) -> !llvm.ptr loc(#loc2)
///     %5 = llvm.alloca %2 x i256 {alignment = 16 : i64} : (i64) -> !llvm.ptr loc(#loc2)
///     %6 = llvm.add %arg0, %2  : i64 loc(#loc2)
///     %7 = llvm.intr.bswap(%1)  : (i256) -> i256 loc(#loc2)
///     %8 = llvm.intr.bswap(%0)  : (i256) -> i256 loc(#loc2)
///     llvm.store %7, %3 {alignment = 16 : i64} : i256, !llvm.ptr loc(#loc2)
///     llvm.store %8, %4 {alignment = 16 : i64} : i256, !llvm.ptr loc(#loc2)
///     llvm.call @cairo_native__libfunc__pedersen(%5, %3, %4) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> () loc(#loc2)
///     llvm.return %6 : i64 loc(#loc3)
///   } loc(#loc1)
///   ...
/// ```
///
/// The compiled MLIR function `run_test` takes a single argument as input, the
/// `pedersen` counter and returns the incremented counter at the end of the call.
/// The counter is incremented by 1 in the MLIR code, in the statement
/// `%6 = llvm.add %arg0, %2  : i64 loc(#loc2)`, which takes the `%arg0` input
/// and adds `%2` to it. We can see from statement
/// `%2 = llvm.mlir.constant(1 : i64) : i64 loc(#loc1)` that `%2` holds the
/// constant 1.
/// When this compiled MLIR code is called, the initial value of all builtin
/// counters is set to `0` as can be seen in the
/// [`invoke_dynamic` function](../src/executor.rs#L240).
pub mod section04 {}

#[cfg_attr(doc, aquamarine::aquamarine)]
/// # Implementing Libfuncs
/// 🚧 WIP
///
/// ## A libfunc implementation.
///
/// A libfunc usually works with a `type`, such as `felt252`. The compiler
/// needs to have information on this type, such as its layout and size.
/// This is defined in `src/types.rs` and `src/types/{typename}.rs`.
///
/// On each `src/types/{typename}.rs` such as `src/types/felt252.rs` you will
/// find a `build` function, this has all the necessary arguments to generate
/// the proper type and return a MLIR type, such as
/// `IntegerType::new(context, 252)` (a 252 bit integer).
///
/// In `src/types.rs` we need to declare the type layout, for example the
/// `felt252` would have the layout returned by `get_integer_layout(252)`.
/// A type that doesn't have size would be `Layout::new::<()>()`, or if the
/// type is a pointer like box: `Layout::new::<*mut ()>()`
///
/// When adding a type, we also need to add the **serialization** and
/// **deserialization** functionality, so we can use it with the JIT runner.
///
/// You can find this functionality under `src/values.rs` and
/// `src/values/{typename}.rs`. As you can see, the project is quite organized
/// if you have a feel of its layout.
///
/// Serialization is done using `Serde`, and each type provides a `deserialize`
/// and `serialize` function. The inner workings of such functions can be a bit
/// complex due to how the JIT runner works. You need to work with pointers and
/// unsafe rust.
///
/// In `values.rs` we should also declare whether the type is complex under
/// `is_complex` in the `ValueBuilder` trait implementation.
///
/// > Complex types are always passed by pointer (both as params and return
/// > values) and require a stack allocation. Examples of complex values include
/// > structs and enums, but not felts since LLVM considers them integers.
///
/// ### Deserializing a type
/// When **deserializing** (a.k.a converting the inputs so the JIT runner
/// accepts them), you are passed a bump allocator arena from `Bumpalo`, the
/// general idea is to get the layout and size of the type, allocate it under
/// the arena, get a pointer, and return it. Which will later be passed to the
/// MLIR JIT runner. It is important the pointers passed are allocated by the
/// arena and not Rust itself.
///
/// Then we need to hookup de `deserialize` method in `values.rs` `deserialize`
/// method.
///
/// ### Serializing a type
/// When **serializing** a type, you will get a `ptr: NonNull<()>` (non null
/// pointer), which you will have to cast, dereference and then deserialize.
///
/// For a simple type to learn how it works, we recommend checking
/// `src/values/uint8.rs`, for more complex types, check `src/values/felt252.rs`.
/// The hardest types to understand are the enums, dictionaries and arrays,
/// since they are complex types.
///
/// Then we need to hookup de `serialize` method in `values.rs` `serialize` method.
///
/// ### Implementing the library function
/// Libfuncs are implemented under `src/libfuncs.rs` and
/// `src/libfuncs/{libfunc_name}.rs`. Just like types.
///
/// Using the `src/libfuncs/felt252.rs` libfuncs as a aid:
///
/// ```rust,noexecute
/// /// Select and call the correct libfunc builder function from the selector.
/// pub fn build<'ctx, 'this, TType, TLibfunc>(
///     context: &'ctx Context,
///     registry: &ProgramRegistry<TType, TLibfunc>,
///     entry: &'this Block<'ctx>,
///     location: Location<'ctx>,
///     helper: &LibfuncHelper<'ctx, 'this>,
///     metadata: &mut MetadataStorage,
///     selector: &Felt252Concrete,
/// ) -> Result<()>
/// where
///     TType: GenericType,
///     TLibfunc: GenericLibfunc,
///     <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
///     <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
/// {
///     match selector {
///         Felt252Concrete::BinaryOperation(info) => {
///             build_binary_operation(context, registry, entry, location, helper, metadata, info)
///         }
///         Felt252Concrete::Const(info) => {
///             build_const(context, registry, entry, location, helper, metadata, info)
///         }
///         Felt252Concrete::IsZero(info) => {
///             build_is_zero(context, registry, entry, location, helper, metadata, info)
///         }
///     }
/// }
/// ```
///
/// You can see it also defines a build function, in this case the last method
/// is a selector, this means in this case we have a group of related libfuncs,
/// which we will implement in this same file. This is where calls to melior
/// (MLIR) and most MLIR code is located, where one gets their hands dirty.
///
/// After implementing the libfuncs, we need to hookup the `build` method in
/// the `src/libfuncs.rs` match statement.
///
/// ### Example libfunc implementation: u8_to_felt252
/// An example libfunc, converting a u8 to a felt252, extensively commented:
///
/// ```rust,noexecute
/// /// Generate MLIR operations for the `u8_to_felt252` libfunc.
/// pub fn build_to_felt252<'ctx, 'this, TType, TLibfunc>(
///     // The Context from MLIR, this is like the heart of the MLIR API, its required to create most stuff like types.
///     context: &'ctx Context,
///     // This is the sierra program registry, it aids us at finding types, functions, etc.
///     registry: &ProgramRegistry<TType, TLibfunc>,
///     // This is the MLIR entry block for this libfunc. Remember we append operations to blocks.
///     entry: &'this Block<'ctx>,
///     // The already created MLIR location for this libfunc, we need to pass this to all the MLIR operations.
///     location: Location<'ctx>,
///     // A helper, which also works as a MLIR Module, it has useful functions for stuff like branching to other libfuncs.
///     helper: &LibfuncHelper<'ctx, 'this>,
///     // The metadata storage, contains extra information needed on some libfuncs. Check out `src/metadata.rs` to learn how it works.
///     metadata: &mut MetadataStorage,
///     // The sierra information for this specific library function. This libfunc only contains signature information, but
///     // others which are generic over a type will contain information about that type, for example array related libfuncs.
///     info: &SignatureOnlyConcreteLibfunc,
/// ) -> Result<()>
/// where
///     TType: GenericType,
///     TLibfunc: GenericLibfunc,
///     <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
///     <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
/// {
///     // We retrieve the felt252 type from the registry and call the "build" method to create the MLIR type.
///     // We could also just call get_type() to hold on to the sierra type, and then `.layout(registry)` to get the type layout,
///     // which is needed in some libfuncs doing more complex stuff.
///     let felt252_ty = registry
///         .get_type(&info.branch_signatures()[0].vars[0].ty)?
///         .build(context, helper, registry, metadata)?;
///
///     // Retrieve the first argument passed to this library function, in this case its the u8 value we need to convert.
///     let value: Value = entry.argument(0)?.into();
///
///     // We create a "extui" operation from the "arith" dialect, which basically zero extends the value to have the same bits as the given type.
///     let op = entry.append_operation(arith::extui(value, felt252_ty, location));
///
///     // Get  the result from the operation, in this case it's the extended value
///     let result = op.result(0)?.into();
///
///     // Using the helper argument, append the branching operation to the next statement, passing result as our output variable.
///     entry.append_operation(helper.br(0, &[result], location));
///
///     Ok(())
/// }
/// ```
///
/// More info on the `extui` operation: <https://mlir.llvm.org/docs/Dialects/ArithOps/#arithextui-arithextuiop>
pub mod section05 {}

#[cfg_attr(doc, aquamarine::aquamarine)]
/// # Debugging
///
/// ## Useful environment variables
///
/// These 2 env vars will dump the generated MLIR code from any compilation on the current working directory as:
///
/// - `dump.mlir`: The MLIR code after passes without locations.
/// - `dump-debug.mlir`: The MLIR code after passes with locations.
/// - `dump-prepass.mlir`: The MLIR code before without locations.
/// - `dump-prepass-debug.mlir`: The MLIR code before passes with locations.
///
/// Do note that the MLIR with locations is in pretty form and thus not suitable to pass to `mlir-opt`.
///
/// ```bash
/// export NATIVE_DEBUG_DUMP_PREPASS=1
/// export NATIVE_DEBUG_DUMP=1
/// ```
///
/// Enable logging to see the compilation process:
///
/// ```bash
/// export RUST_LOG="cairo_native=trace"
/// ```
///
/// ## Other tips:
///
/// - Try to find the minimal program to reproduce an issue, the more isolated the easier to test.
/// - Use the `debug_utils` print utilities, more info [here](https://lambdaclass.github.io/cairo_native/cairo_native/metadata/debug_utils/struct.DebugUtils.html):
///
/// ```rust,noexecute
/// #[cfg(feature = "with-debug-utils")]
/// {
///     metadata.get_mut::<DebugUtils>()
///         .unwrap()
///         .print_pointer(context, helper, entry, ptr, location)?;
/// }
/// ```
///
/// ## Debugging Contracts
///
/// Contracts are difficult to debug for various reasons, including:
/// - They are external to the project.
/// - We don’t have their source code.
/// - They run autogenerated code (the wrapper).
/// - They have a limited number of allowed libfuncs (ex. cannot use the print libfunc).
/// - Usually it’s not a single contract but multiple that
///
/// Some of them have workarounds:
///
/// ### Obtaining the contract
/// There are various options for obtaining the contract, which include:
///
/// - Manually invoking the a Starknet API using `curl` with the contract class.
///
/// Example:
///
/// ```bash
/// curl --location --request POST 'https://mainnet.juno.internal.lambdaclass.com' \
/// --header 'Content-Type: application/json' \
/// --data-raw '{
///   "jsonrpc": "2.0",
///   "method": "starknet_getClass",
///   "id": 0,
///   "params": {
///     "class_hash": "0x036078334509b514626504edc9fb252328d1a240e4e948bef8d0c08dff45927f",
///     "block_id": 657887
/// }
/// }'
/// ```
///
/// - Running the replay with some code to write all the executed contracts on disk.
///
/// Both should provide us with the contract, but if we’re manually invoking the API we’ll need to process the JSON a bit to:
///
/// - Remove the JsonRPC overhead, and
/// - Convert the ABI from a string of JSON into a JSON object.
///
/// ### Interpreting the contract
/// The contract JSON contains the Sierra program in a useless form (in the sense
/// that we cannot understand anything), as well as some information about the
/// entry points and some ABI types. We’ll need the Sierra program (in Sierra
/// format, not the JSON) to be able to understand what should be happening.
///
/// We can use the `starknet-sierra-extract-code` binary, which can be found in
/// the cairo project when compiled from source (not in the binary distribution).
/// That binary will extract the Sierra program without any debug information,
/// which is still not very useful.
///
/// Once we have the Sierra we can run the
/// [Sierra mapper](https://github.com/azteca1998/sierra-mapper) to autogenerate
/// some type, libfunc and function names so that we know what we’re looking at
/// without losing our mind. The Sierra mapper can be run multiple times, adding
/// more names manually as the user sees fit.
///
/// ### How to actually debug
///
/// First of all we need to **know which contract is actually failing**. Most
/// of the time the contract where it crashes isn’t the transaction’s class
/// hash, but a chain of contract/library calls.
///
/// To know which contract is being called we can add some debugging prints in
/// the replay that logs contract executions. For example:
///
/// ```rust,noexecute
/// impl StarknetSyscallHandler for ReplaySyscallHandler {
///     // ...
///
///     fn library_call(
///         &mut self,
///         class_hash: Felt,
///         function_selector: Felt,
///         calldata: &[Felt],
///         remaining_gas: &mut u128,
///     ) -> SyscallResult<Vec<Felt>> {
///         // ...
///
///         println!("Starting execution of contract {class_hash} on selector {function_selector} with calldata {calldata:?}.");
///         let result = executor.invoke_contract_dynamic(...);
///         println!("Finished execution of contract {class_hash}.");
///         if result.failure_flag {
///             println!("Execution of contract {class_hash} failed.");
///         }
///
///         // ...
///     }
///
///     fn call_contract(
///         &mut self,
///         address: Felt,
///         entry_point_selector: Felt,
///         calldata: &[Felt],
///         remaining_gas: &mut u128,
///     ) -> SyscallResult<Vec<Felt>> {
/// 			  // ...
///
/// 			  println!("Starting execution of contract {class_hash} on selector {function_selector} with calldata {calldata:?}.");
/// 			  let result = executor.invoke_contract_dynamic(...);
/// 			  println!("Finished execution of contract {class_hash}.");
/// 			  if result.failure_flag {
/// 					  println!("Execution of contract {class_hash} failed.");
/// 				}
///
/// 				// ...
/// 		}
/// }
/// ```
///
/// If we run something like the above then the
/// [replay](https://github.com/lambdaclass/starknet-replay) should start
/// printing the log of what’s actually being executed and where it crashes.
/// It may print multiple times the error message, but **only the first one is
/// the relevant one** (the others should be the contract call chain in reverse
/// order). Once we know which contract is being called and its calldata we can
/// download and extract its Sierra as detailed above.
///
/// We then need to know **where it fails within the contract**. To do that we
/// can look at the error message and deduce where it’s used based on the Sierra
/// program. For example, the error message `u256_mul overflow` is felt-encoded
/// as `0x753235365f6d756c206f766572666c6f77`, or
/// `39879774624083218221774975706286902767479` in decimal. If we look for
/// usages of that specific value we’ll most likely find all the **places where
/// that error can be thrown**. Now we just need to narrow them down to a single
/// one and we’ll be able to actually start debugging.
///
/// An idea on how to do that is modifying Cairo native so that it adds a
/// breakpoint every time a constant with that error message is generated.
/// For example:
///
/// ```rust,noexecute
/// /// Generate MLIR operations for the `felt252_const` libfunc.
/// pub fn build_const<'ctx, 'this>(
///     context: &'ctx Context,
///     registry: &ProgramRegistry<CoreType, CoreLibfunc>,
///     entry: &'this Block<'ctx>,
///     location: Location<'ctx>,
///     helper: &LibfuncHelper<'ctx, 'this>,
///     metadata: &mut MetadataStorage,
///     info: &Felt252ConstConcreteLibfunc,
/// ) -> Result<()> {
///     let value = match info.c.sign() {
///         Sign::Minus => {
///             let prime = metadata
///                 .get::<PrimeModuloMeta<Felt>>()
///                 .ok_or(Error::MissingMetadata)?
///                 .prime();
///             (&info.c + prime.to_bigint().expect("always is Some"))
///                 .to_biguint()
///                 .expect("always is positive")
///         }
///         _ => info.c.to_biguint().expect("sign already checked"),
///     };
///     let felt252_ty = registry.build_type(
///         context,
///         helper,
///         registry,
///         metadata,
///         &info.branch_signatures()[0].vars[0].ty,
///     )?;
///     if value == "39879774624083218221774975706286902767479".parse().unwrap() {
/// 		    // If using the debugger:
///         metadata
///             .get_mut::<crate::metadata::debug_utils::DebugUtils>()
///             .unwrap()
///             .debug_breakpoint_trap(entry, location)
///             .unwrap();
///         // If not using the debugger (not tested, may not provide useful information).
///         metadata
///             .get_mut::<crate::metadata::debug_utils::DebugUtils>()
///             .unwrap()
///             .debug_print(
///                 context,
///                 helper,
///                 entry,
///                 &format!("Invoked felt252_const<error_msg> at {location}."),
///                 location,
///             )
///             .unwrap();
///     }
///     let value = entry.const_int_from_type(context, location, value, felt252_ty)?;
///     entry.append_operation(helper.br(0, &[value], location));
///     Ok(())
/// }
/// ```
///
/// Using the debugger will also provide the internal call backtrace (of the
/// contract) and register values, so it’s the recommended way, but depending on
/// the contract it may not be feasible (ex. the contract is too big and running
/// the debugger is not practical due to the amount of time it takes to get to
/// the crash).
///
/// Once we know exactly where it crashes we can follow the control flow of the
/// Sierra program backwards and discover how it reached that point.
///
/// In some cases the **problem may be somewhere completely different from where
/// the error is thrown**. In other words, the error we’re seeing may be a side
/// effect of a completely different bug. For example, in a `u256_mul overflow`,
/// the bug may be found in the mul operation implementation, or alternatively it
/// may just be that the values passed to it are not what they should be. That’s
/// why it’s important to check for those cases and keep following the control
/// flow backwards as required.
///
/// ### Fixing the bug
/// Before fixing the bug it’s really important to know:
///
/// - **Where** it happens (in our compiler, not so much in the contract at this point)
/// - **Why** it happens (as in, what caused this bug to be in our codebase in the first place)
/// - **How** to fix it properly (not the actual code but to know what steps to take to fix it).
/// - Could the **same bug** happen in **different places**? (for example, if it was the implementation of `u64_sqrt`, could the same bug happen in `u32_sqrt` and others?)
/// - What **side-effects** will the bug fix trigger? (for example, if the fix implies changing the layout of some type, will the new layout make something completely unrelated fail later on?)
///
/// The last one is really important since we don’t want to cause more bugs
/// fixing the ones we already have. To understand the side effects we need to
/// have a full understanding of the bug, which implies having an answer to (at
/// least) all the other things to know before fixing it.
///
/// Once we know all that we can:
///
/// 1. Add tests that reproduce the bug (including all the variants that we may discover).
/// 2. Implement the fix in code.
///
/// > Note: Those steps must be done in that order. Otherwise we risk
/// unconsciously avoiding bugs in our tests for our bug fix implementation by
/// building our tests from our implementation instead of the correct
/// behaviour.
pub mod section06 {}

#[cfg_attr(doc, aquamarine::aquamarine)]
/// # Sierra Resources
/// 🚧 TODO
/// ## What is a library function
/// Sierra uses a list of builtin functions that implement the language
/// functionality, those are called library functions, short: libfuncs.
/// Basically every statement in a sierra program is a call to a libfunc, thus
/// they are the core of Cairo Native.
///
/// Each libfunc takes input variables and outputs some other variables. Note
/// that in cairo a function that has 2 arguments may have more in sierra, due
/// to "implicits" / "builtins", which are arguments passed hidden from the
/// user, such as the `GasBuiltin`.
pub mod section07 {}

#[cfg_attr(doc, aquamarine::aquamarine)]
/// # MLIR Resources
///
/// ## How MLIR Works
/// MLIR is composed of **dialects**, which is like a IR of it's own, and this
/// IR can be converted to another dialect IR (if the functionality exists).
/// This is what makes MLIR shine.
///
/// Some commonly used dialects in this project:
/// - The arith dialect: It contains arithmetic operations, such as `addi`, `subi` for addition and subtraction.
/// - The cf dialect: It contains basic control flow operations, such as the `br` and `cond_br`, which are unconditional and conditional jumps.
///
/// ### The IR
/// The MLIR IR is composed recursively like this: `Operation -> Region -> Block -> Operations`
///
/// Each operation has 1 or more region, each region has 1 or more blocks, each
/// block has 1 or more operations.
///
/// This way a MLIR program can be composed.
///
/// ### Transformations and passes
/// MLIR provides a set of transformations that can optimize the IR.
/// Such as `canonicalize`.
///
/// Check out <https://mlir.llvm.org/docs/Canonicalization/> and <https://mlir.llvm.org/docs/Passes/>.
///
/// ### Translating
/// In our case, llvm is our target, so we end up translating all dialects down
/// to the LLVM dialect, which then gets converted to LLVM IR.
///
/// ## Learning Resources
/// Resources marked with **→** are best.
///
/// - Introduction
///     - **→** [2019 EuroLLVM Developers’ Meeting: MLIR: Multi-Level Intermediate Representation Compiler Infrastructure](https://www.youtube.com/watch?v=qzljG6DKgic)
///     - → [MLIR: A Compiler Infrastructure for the End of Moore’s Law](https://arxiv.org/pdf/2002.11054.pdf)
///     The paper introducing the MLIR framework
///         - 7-minute video summary of paper:
///         [Read a paper: Multi-level Intermediate Representation (MLIR)](https://www.youtube.com/watch?v=6BwqK6E8v3g)
///         - Another version of the paper:
///         [MLIR: Scaling Compiler Infrastructure for Domain Specific Computation](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/85bf23fe88bd5c7ff60365bd0c6882928562cbeb.pdf)
/// - MLIR Tutorial
///     - **→** (slides) [MLIR Tutorial (LLVM Dev Mtg, 2020)](https://llvm.org/devmtg/2020-09/slides/MLIR_Tutorial.pdf)
///     - **→** (video) [2020 LLVM Developers’ Meeting: M. Amini & R. Riddle “MLIR Tutorial”](https://www.youtube.com/watch?v=Y4SvqTtOIDk)
///     - (older slides) [MLIR Tutorial (LLVM Developers Meeting, Euro-LLVM 2019)](https://llvm.org/devmtg/2019-04/slides/Tutorial-AminiVasilacheZinenko-MLIR.pdf)
///     - (older slides) [MLIR Tutorial (MLIR 4 HPC, 2019)](https://users.cs.utah.edu/~mhall/mlir4hpc/pienaar-MLIR-Tutorial.pdf)
///     - (older video) [2019 EuroLLVM Developers’ Meeting: Mehdi & Vasilache & Zinenko “Building a Compiler with MLIR”](https://www.youtube.com/watch?v=cyICUIZ56wQ)
/// - **→** Another MLIR Tutorial
/// [https://github.com/j2kun/mlir-tutorial](https://github.com/j2kun/mlir-tutorial)
/// - **→** [How to build a compiler with LLVM and MLIR](https://www.youtube.com/playlist?list=PLlONLmJCfHTo9WYfsoQvwjsa5ZB6hjOG5)
/// - Other articles, posts
///     - **→** [Intro to LLVM and MLIR with Rust and Melior](https://edgarluque.com/blog/mlir-with-rust/)
///     - **→** [MLIR Notes](http://lastweek.io/notes/MLIR/)
///     - **→** [Compilers and IRs: LLVM IR, SPIR-V, and MLIR](https://www.lei.chat/posts/compilers-and-irs-llvm-ir-spirv-and-mlir/) [[HN]](https://news.ycombinator.com/item?id=33387149)
///     - [MLIR: Redefining the compiler infrastructure](https://iq.opengenus.org/mlir-compiler-infrastructure/)
///     - [Pinch: Implementing a borrow-checked language with MLIR](https://badland.io/pinch.md)
/// - [Official Documentation](https://mlir.llvm.org/docs/)
///     - [MLIR Homepage](https://mlir.llvm.org/)
///     - [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
///     - [MLIR Compiler](https://www.youtube.com/MLIRCompiler) Youtube Channel
///
/// ### Talks, Presentations, & Videos
/// - [2020 LLVM in HPC Workshop: Keynote: MLIR: an Agile Infrastructure for Building a Compiler Ecosystem](https://www.youtube.com/watch?v=0bxyZDGs-aA)
/// - [2021 LLVM Dev Mtg “Representing Concurrency with Graph Regions in MLIR”](https://www.youtube.com/watch?v=Vfk9n3ir_5s)
/// - [2022 LLVM Dev Mtg: Paths towards unifying LLVM and MLIR](https://www.youtube.com/watch?v=VbFqA9rvxPs)
/// - [2022 LLVM Dev Mtg: VAST: MLIR for program analysis of C/C++](https://www.youtube.com/watch?v=YFqWa4pxXzM)
/// - [2022 LLVM Dev Mtg: MLIR for Functional Programming](https://www.youtube.com/watch?v=cyMQbZ0B84Q)
/// - [2022 EuroLLVM Dev Mtg “Prototyping a Compiler for Homomorphic Encryption Using MLIR”](https://www.youtube.com/watch?v=QyxiqmO6_qQ)
/// - [cirgen: MLIR based compiler for zk-STARK circuit generation - Frank Laub (RISC Zero)](https://www.youtube.com/watch?v=TsP14-hI_W0)
/// - [Prototyping a compiler for homomorphic encryption using MLIR](https://www.youtube.com/watch?v=F9qXBuSkQFY)
///     - [Slides](https://llvm.org/devmtg/2022-04-03/slides/Prototyping.a.compiler.for.homomorphic.encryption.in.MLIR.pdf)
/// - [The HEIR Compiler w/ Jeremy Kun](https://www.youtube.com/watch?v=ne5D_kqlxYg)
///
/// ### Useful code
/// - [`raviqqe/melior`](https://github.com/raviqqe/melior)
/// - [`femtomc/mlir-sys`](https://github.com/femtomc/mlir-sys) Rust bindings to the MLIR C API.
/// - [`GetFirefly/firefly`](https://github.com/GetFirefly/firefly) An alternative BEAM implementation, designed for WebAssembly.
pub mod section08 {}
