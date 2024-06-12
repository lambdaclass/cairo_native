////! # Compilation process
//! # Compilation process
////!
//!
////! A Sierra program is compiled one function at a time. Each function has a pre-entry block that
//! A Sierra program is compiled one function at a time. Each function has a pre-entry block that
////! will be ran only once, even in tail-recursive functions. All libfuncs are intended to place
//! will be ran only once, even in tail-recursive functions. All libfuncs are intended to place
////! their stack-allocating operations there so as to not grow the stack when recursing.
//! their stack-allocating operations there so as to not grow the stack when recursing.
////!
//!
////! After the pre-entry block, there is an entry block, which is in charge of preparing the first
//! After the pre-entry block, there is an entry block, which is in charge of preparing the first
////! statement's arguments and jumping into it. From here on, all the statements's
//! statement's arguments and jumping into it. From here on, all the statements's
////! [builders](crate::libfuncs::LibfuncBuilder) are invoked. Every libfunc builder must end its
//! [builders](crate::libfuncs::LibfuncBuilder) are invoked. Every libfunc builder must end its
////! execution calling a branch function from the helper, which will generate the operations required
//! execution calling a branch function from the helper, which will generate the operations required
////! to jump to next statements. This simplifies the branching design, especially when a libfunc has
//! to jump to next statements. This simplifies the branching design, especially when a libfunc has
////! multiple target branches.
//! multiple target branches.
////!
//!
////! > Note: Libfunc builders must have a branching operation out into each possible branch, even if
//! > Note: Libfunc builders must have a branching operation out into each possible branch, even if
////!     it's unreachable. This is required to keep the state consistent. More on that later.
//!     it's unreachable. This is required to keep the state consistent. More on that later.
////!
//!
////! Some statements do require a special landing block. Those are the ones which are the branching
//! Some statements do require a special landing block. Those are the ones which are the branching
////! target of more than a single statement. In other words, if a statement can be reached (directly)
//! target of more than a single statement. In other words, if a statement can be reached (directly)
////! from more than a single place, it needs a landing block.
//! from more than a single place, it needs a landing block.
////!
//!
////! The landing blocks are in charge of synchronizing the Sierra state. The state is just a
//! The landing blocks are in charge of synchronizing the Sierra state. The state is just a
////! dictionary mapping variable ids to their values. Since the values can come from a single branch,
//! dictionary mapping variable ids to their values. Since the values can come from a single branch,
////! this landing block is required.
//! this landing block is required.
////!
//!
////! In order to generate the libfuncs's blocks, all the libfunc's entry blocks are required. That is
//! In order to generate the libfuncs's blocks, all the libfunc's entry blocks are required. That is
////! why they are generated all beforehand. The order in which they are generated follows a
//! why they are generated all beforehand. The order in which they are generated follows a
////! breadth-first ordering; that is, the compiler uses a [BFS algorithm]. This algorithm should
//! breadth-first ordering; that is, the compiler uses a [BFS algorithm]. This algorithm should
////! generate the libfuncs in the same order as they appear in Sierra. As expected, the algorithm
//! generate the libfuncs in the same order as they appear in Sierra. As expected, the algorithm
////! forks the path each time a branching libfunc is found, which dies once a return statement is
//! forks the path each time a branching libfunc is found, which dies once a return statement is
////! detected.
//! detected.
////!
//!
////! ## Function nomenclature transforms
//! ## Function nomenclature transforms
////!
//!
////! When compiling from Cairo, or from a Sierra source with debug information (the `-r` flag on
//! When compiling from Cairo, or from a Sierra source with debug information (the `-r` flag on
////! `cairo-compile`), those identifiers are the function's exported symbol. However, Sierra programs
//! `cairo-compile`), those identifiers are the function's exported symbol. However, Sierra programs
////! are not required to contain that information. In those cases, the
//! are not required to contain that information. In those cases, the
////! (`generate_function_name`)[generate_function_name] will generate a new symbol name based on its
//! (`generate_function_name`)[generate_function_name] will generate a new symbol name based on its
////! function id.
//! function id.
////!
//!
////! ## Tail-recursive functions
//! ## Tail-recursive functions
////!
//!
////! Part of the tail-recursion handling algorithm is implemented here, but tail-recursive functions
//! Part of the tail-recursion handling algorithm is implemented here, but tail-recursive functions
////! are better explained in (their metadata section)[crate::metadata::tail_recursion].
//! are better explained in (their metadata section)[crate::metadata::tail_recursion].
////!
//!
////! [BFS algorithm]: https://en.wikipedia.org/wiki/Breadth-first_search
//! [BFS algorithm]: https://en.wikipedia.org/wiki/Breadth-first_search
//

//use crate::{
use crate::{
//    debug_info::DebugLocations,
    debug_info::DebugLocations,
//    error::Error,
    error::Error,
//    libfuncs::{BranchArg, LibfuncBuilder, LibfuncHelper},
    libfuncs::{BranchArg, LibfuncBuilder, LibfuncHelper},
//    metadata::{
    metadata::{
//        gas::{GasCost, GasMetadata},
        gas::{GasCost, GasMetadata},
//        tail_recursion::TailRecursionMeta,
        tail_recursion::TailRecursionMeta,
//        MetadataStorage,
        MetadataStorage,
//    },
    },
//    types::TypeBuilder,
    types::TypeBuilder,
//    utils::generate_function_name,
    utils::generate_function_name,
//};
};
//use bumpalo::Bump;
use bumpalo::Bump;
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    edit_state,
    edit_state,
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        ConcreteLibfunc,
        ConcreteLibfunc,
//    },
    },
//    ids::{ConcreteTypeId, VarId},
    ids::{ConcreteTypeId, VarId},
//    program::{Function, Invocation, Program, Statement, StatementIdx},
    program::{Function, Invocation, Program, Statement, StatementIdx},
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use itertools::Itertools;
use itertools::Itertools;
//use melior::{
use melior::{
//    dialect::{
    dialect::{
//        arith::CmpiPredicate,
        arith::CmpiPredicate,
//        cf, func, index,
        cf, func, index,
//        llvm::{self, LoadStoreOptions},
        llvm::{self, LoadStoreOptions},
//        memref,
        memref,
//    },
    },
//    ir::{
    ir::{
//        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
//        r#type::{FunctionType, IntegerType, MemRefType},
        r#type::{FunctionType, IntegerType, MemRefType},
//        Attribute, Block, BlockRef, Identifier, Location, Module, Region, Type, Value,
        Attribute, Block, BlockRef, Identifier, Location, Module, Region, Type, Value,
//    },
    },
//    Context,
    Context,
//};
};
//use std::{
use std::{
//    cell::Cell,
    cell::Cell,
//    collections::{hash_map::Entry, BTreeMap, HashMap, HashSet},
    collections::{hash_map::Entry, BTreeMap, HashMap, HashSet},
//    ops::Deref,
    ops::Deref,
//};
};
//

///// The [BlockStorage] type is used to map each statement into its own entry block (on the right),
/// The [BlockStorage] type is used to map each statement into its own entry block (on the right),
///// and its landing block (on the left) if required.
/// and its landing block (on the left) if required.
/////
///
///// The landing block contains also the variable ids that must be present when jumping into it,
/// The landing block contains also the variable ids that must be present when jumping into it,
///// otherwise it's a compiler error due to an inconsistent variable state.
/// otherwise it's a compiler error due to an inconsistent variable state.
//type BlockStorage<'c, 'a> =
type BlockStorage<'c, 'a> =
//    HashMap<StatementIdx, (Option<(BlockRef<'c, 'a>, Vec<VarId>)>, BlockRef<'c, 'a>)>;
    HashMap<StatementIdx, (Option<(BlockRef<'c, 'a>, Vec<VarId>)>, BlockRef<'c, 'a>)>;
//

///// Run the compiler on a program. The compiled program is stored in the MLIR module.
/// Run the compiler on a program. The compiled program is stored in the MLIR module.
/////
///
///// The generics `TType` and `TLibfunc` contain the information required to generate the MLIR types
/// The generics `TType` and `TLibfunc` contain the information required to generate the MLIR types
///// and statement operations. Most of the time you'll want to use the default ones, which are
/// and statement operations. Most of the time you'll want to use the default ones, which are
///// [CoreType](cairo_lang_sierra::extensions::core::CoreType) and
/// [CoreType](cairo_lang_sierra::extensions::core::CoreType) and
///// [CoreLibfunc](cairo_lang_sierra::extensions::core::CoreLibfunc) respectively.
/// [CoreLibfunc](cairo_lang_sierra::extensions::core::CoreLibfunc) respectively.
/////
///
///// This function needs the program and the program's registry, which doesn't need to have AP
/// This function needs the program and the program's registry, which doesn't need to have AP
///// tracking information.
/// tracking information.
/////
///
///// Additionally, it needs a reference to the MLIR context, the output module and the metadata
/// Additionally, it needs a reference to the MLIR context, the output module and the metadata
///// storage. The last one is passed externally so that stuff can be initialized if necessary.
/// storage. The last one is passed externally so that stuff can be initialized if necessary.
//pub fn compile(
pub fn compile(
//    context: &Context,
    context: &Context,
//    module: &Module,
    module: &Module,
//    program: &Program,
    program: &Program,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    debug_info: Option<&DebugLocations>,
    debug_info: Option<&DebugLocations>,
//) -> Result<(), Error> {
) -> Result<(), Error> {
//    for function in &program.funcs {
    for function in &program.funcs {
//        tracing::info!("Compiling function `{}`.", function.id);
        tracing::info!("Compiling function `{}`.", function.id);
//        compile_func(
        compile_func(
//            context,
            context,
//            module,
            module,
//            registry,
            registry,
//            function,
            function,
//            &program.statements,
            &program.statements,
//            metadata,
            metadata,
//            debug_info,
            debug_info,
//        )?;
        )?;
//    }
    }
//

//    tracing::info!("The program was compiled successfully.");
    tracing::info!("The program was compiled successfully.");
//    Ok(())
    Ok(())
//}
}
//

///// Compile a single Sierra function.
/// Compile a single Sierra function.
/////
///
///// The function accepts a `Function` argument, which provides the function's entry point, signature
/// The function accepts a `Function` argument, which provides the function's entry point, signature
///// and name. Check out [compile](self::compile) for a description of the other arguments.
/// and name. Check out [compile](self::compile) for a description of the other arguments.
/////
///
///// The [module docs](self) contain more information about the compilation process.
/// The [module docs](self) contain more information about the compilation process.
//fn compile_func(
fn compile_func(
//    context: &Context,
    context: &Context,
//    module: &Module,
    module: &Module,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    function: &Function,
    function: &Function,
//    statements: &[Statement],
    statements: &[Statement],
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    debug_info: Option<&DebugLocations>,
    debug_info: Option<&DebugLocations>,
//) -> Result<(), Error> {
) -> Result<(), Error> {
//    let region = Region::new();
    let region = Region::new();
//    let blocks_arena = Bump::new();
    let blocks_arena = Bump::new();
//

//    let mut arg_types = extract_types(
    let mut arg_types = extract_types(
//        context,
        context,
//        module,
        module,
//        &function.signature.param_types,
        &function.signature.param_types,
//        registry,
        registry,
//        metadata,
        metadata,
//    )
    )
//    .collect::<Result<Vec<_>, _>>()?;
    .collect::<Result<Vec<_>, _>>()?;
//    let mut return_types = extract_types(
    let mut return_types = extract_types(
//        context,
        context,
//        module,
        module,
//        &function.signature.ret_types,
        &function.signature.ret_types,
//        registry,
        registry,
//        metadata,
        metadata,
//    )
    )
//    .collect::<Result<Vec<_>, _>>()?;
    .collect::<Result<Vec<_>, _>>()?;
//

//    // Replace memory-allocated arguments with pointers.
    // Replace memory-allocated arguments with pointers.
//    for (ty, type_info) in
    for (ty, type_info) in
//        arg_types
        arg_types
//            .iter_mut()
            .iter_mut()
//            .zip(function.signature.param_types.iter().filter_map(|type_id| {
            .zip(function.signature.param_types.iter().filter_map(|type_id| {
//                let type_info = registry.get_type(type_id).unwrap();
                let type_info = registry.get_type(type_id).unwrap();
//                if type_info.is_builtin() && type_info.is_zst(registry) {
                if type_info.is_builtin() && type_info.is_zst(registry) {
//                    None
                    None
//                } else {
                } else {
//                    Some(type_info)
                    Some(type_info)
//                }
                }
//            }))
            }))
//    {
    {
//        if type_info.is_memory_allocated(registry) {
        if type_info.is_memory_allocated(registry) {
//            *ty = llvm::r#type::pointer(context, 0);
            *ty = llvm::r#type::pointer(context, 0);
//        }
        }
//    }
    }
//

//    // Extract memory-allocated return types from return_types and insert them in arg_types as a
    // Extract memory-allocated return types from return_types and insert them in arg_types as a
//    // pointer.
    // pointer.
//    let return_type_infos = function
    let return_type_infos = function
//        .signature
        .signature
//        .ret_types
        .ret_types
//        .iter()
        .iter()
//        .filter_map(|type_id| {
        .filter_map(|type_id| {
//            let type_info = registry.get_type(type_id).unwrap();
            let type_info = registry.get_type(type_id).unwrap();
//            if type_info.is_builtin() && type_info.is_zst(registry) {
            if type_info.is_builtin() && type_info.is_zst(registry) {
//                None
                None
//            } else {
            } else {
//                Some((type_id, type_info))
                Some((type_id, type_info))
//            }
            }
//        })
        })
//        .collect::<Vec<_>>();
        .collect::<Vec<_>>();
//    // Possible values:
    // Possible values:
//    //   None        => Doesn't return anything.
    //   None        => Doesn't return anything.
//    //   Some(false) => Has a complex return type.
    //   Some(false) => Has a complex return type.
//    //   Some(true)  => Has a manual return type which is in `arg_types[0]`.
    //   Some(true)  => Has a manual return type which is in `arg_types[0]`.
//    let has_return_ptr = if return_type_infos.len() > 1 {
    let has_return_ptr = if return_type_infos.len() > 1 {
//        Some(false)
        Some(false)
//    } else if return_type_infos
    } else if return_type_infos
//        .first()
        .first()
//        .is_some_and(|(_, type_info)| type_info.is_memory_allocated(registry))
        .is_some_and(|(_, type_info)| type_info.is_memory_allocated(registry))
//    {
    {
//        assert_eq!(return_types.len(), 1);
        assert_eq!(return_types.len(), 1);
//

//        return_types.remove(0);
        return_types.remove(0);
//        arg_types.insert(0, llvm::r#type::pointer(context, 0));
        arg_types.insert(0, llvm::r#type::pointer(context, 0));
//

//        Some(true)
        Some(true)
//    } else {
    } else {
//        None
        None
//    };
    };
//

//    tracing::debug!("Generating function structure (region with blocks).");
    tracing::debug!("Generating function structure (region with blocks).");
//    let (entry_block, blocks) = generate_function_structure(
    let (entry_block, blocks) = generate_function_structure(
//        context, module, &region, registry, function, statements, metadata,
        context, module, &region, registry, function, statements, metadata,
//    )?;
    )?;
//

//    tracing::debug!("Generating the function implementation.");
    tracing::debug!("Generating the function implementation.");
//    // Workaround for the `entry block of region may not have predecessors` error:
    // Workaround for the `entry block of region may not have predecessors` error:
//    let pre_entry_block = region.insert_block_before(
    let pre_entry_block = region.insert_block_before(
//        entry_block,
        entry_block,
//        Block::new(
        Block::new(
//            &arg_types
            &arg_types
//                .iter()
                .iter()
//                .map(|ty| (*ty, Location::unknown(context)))
                .map(|ty| (*ty, Location::unknown(context)))
//                .collect::<Vec<_>>(),
                .collect::<Vec<_>>(),
//        ),
        ),
//    );
    );
//

//    let initial_state = edit_state::put_results(HashMap::<_, Value>::new(), {
    let initial_state = edit_state::put_results(HashMap::<_, Value>::new(), {
//        let mut values = Vec::new();
        let mut values = Vec::new();
//

//        let mut count = 0;
        let mut count = 0;
//        for param in &function.params {
        for param in &function.params {
//            let type_info = registry.get_type(&param.ty)?;
            let type_info = registry.get_type(&param.ty)?;
//

//            values.push((
            values.push((
//                &param.id,
                &param.id,
//                if type_info.is_builtin() && type_info.is_zst(registry) {
                if type_info.is_builtin() && type_info.is_zst(registry) {
//                    pre_entry_block
                    pre_entry_block
//                        .append_operation(llvm::undef(
                        .append_operation(llvm::undef(
//                            type_info.build(context, module, registry, metadata, &param.ty)?,
                            type_info.build(context, module, registry, metadata, &param.ty)?,
//                            Location::unknown(context),
                            Location::unknown(context),
//                        ))
                        ))
//                        .result(0)?
                        .result(0)?
//                        .into()
                        .into()
//                } else {
                } else {
//                    let value = entry_block.argument(count)?.into();
                    let value = entry_block.argument(count)?.into();
//                    count += 1;
                    count += 1;
//

//                    value
                    value
//                },
                },
//            ));
            ));
//        }
        }
//

//        values.into_iter()
        values.into_iter()
//    })?;
    })?;
//

//    tracing::trace!("Implementing the entry block.");
    tracing::trace!("Implementing the entry block.");
//    entry_block.append_operation(cf::br(
    entry_block.append_operation(cf::br(
//        &blocks[&function.entry_point].1,
        &blocks[&function.entry_point].1,
//        &match &statements[function.entry_point.0] {
        &match &statements[function.entry_point.0] {
//            Statement::Invocation(x) => &x.args,
            Statement::Invocation(x) => &x.args,
//            Statement::Return(x) => x,
            Statement::Return(x) => x,
//        }
        }
//        .iter()
        .iter()
//        .map(|x| initial_state[x])
        .map(|x| initial_state[x])
//        .collect::<Vec<_>>(),
        .collect::<Vec<_>>(),
//        Location::unknown(context),
        Location::unknown(context),
//    ));
    ));
//

//    let mut tailrec_storage = Vec::<(Value, BlockRef)>::new();
    let mut tailrec_storage = Vec::<(Value, BlockRef)>::new();
//    foreach_statement_in_function::<_, Error>(
    foreach_statement_in_function::<_, Error>(
//        statements,
        statements,
//        function.entry_point,
        function.entry_point,
//        (initial_state, BTreeMap::<usize, usize>::new()),
        (initial_state, BTreeMap::<usize, usize>::new()),
//        |statement_idx, (mut state, mut tailrec_state)| {
        |statement_idx, (mut state, mut tailrec_state)| {
//            if let Some(gas_metadata) = metadata.get::<GasMetadata>() {
            if let Some(gas_metadata) = metadata.get::<GasMetadata>() {
//                let gas_cost = gas_metadata.get_gas_cost_for_statement(statement_idx);
                let gas_cost = gas_metadata.get_gas_cost_for_statement(statement_idx);
//                metadata.remove::<GasCost>();
                metadata.remove::<GasCost>();
//                metadata.insert(GasCost(gas_cost));
                metadata.insert(GasCost(gas_cost));
//            }
            }
//

//            let (landing_block, block) = &blocks[&statement_idx];
            let (landing_block, block) = &blocks[&statement_idx];
//

//            if let Some((landing_block, _)) = landing_block {
            if let Some((landing_block, _)) = landing_block {
//                tracing::trace!("Implementing the statement {statement_idx}'s landing block.");
                tracing::trace!("Implementing the statement {statement_idx}'s landing block.");
//

//                state = edit_state::put_results(
                state = edit_state::put_results(
//                    HashMap::default(),
                    HashMap::default(),
//                    state
                    state
//                        .keys()
                        .keys()
//                        .sorted_by_key(|x| x.id)
                        .sorted_by_key(|x| x.id)
//                        .enumerate()
                        .enumerate()
//                        .map(|(idx, var_id)| Ok((var_id, landing_block.argument(idx)?.into())))
                        .map(|(idx, var_id)| Ok((var_id, landing_block.argument(idx)?.into())))
//                        .collect::<Result<Vec<_>, Error>>()?
                        .collect::<Result<Vec<_>, Error>>()?
//                        .into_iter(),
                        .into_iter(),
//                )?;
                )?;
//

//                landing_block.append_operation(cf::br(
                landing_block.append_operation(cf::br(
//                    block,
                    block,
//                    &edit_state::take_args(
                    &edit_state::take_args(
//                        state.clone(),
                        state.clone(),
//                        match &statements[statement_idx.0] {
                        match &statements[statement_idx.0] {
//                            Statement::Invocation(x) => &x.args,
                            Statement::Invocation(x) => &x.args,
//                            Statement::Return(x) => x,
                            Statement::Return(x) => x,
//                        }
                        }
//                        .iter(),
                        .iter(),
//                    )?
                    )?
//                    .1,
                    .1,
//                    Location::name(
                    Location::name(
//                        context,
                        context,
//                        &format!("landing_block(stmt_idx={})", statement_idx),
                        &format!("landing_block(stmt_idx={})", statement_idx),
//                        Location::unknown(context),
                        Location::unknown(context),
//                    ),
                    ),
//                ));
                ));
//            }
            }
//

//            Ok(match &statements[statement_idx.0] {
            Ok(match &statements[statement_idx.0] {
//                Statement::Invocation(invocation) => {
                Statement::Invocation(invocation) => {
//                    tracing::trace!(
                    tracing::trace!(
//                        "Implementing the invocation statement at {statement_idx}: {}.",
                        "Implementing the invocation statement at {statement_idx}: {}.",
//                        invocation.libfunc_id
                        invocation.libfunc_id
//                    );
                    );
//                    let libfunc_name =
                    let libfunc_name =
//                        format!("{}(stmt_idx={})", invocation.libfunc_id, statement_idx);
                        format!("{}(stmt_idx={})", invocation.libfunc_id, statement_idx);
//

//                    let (state, _) = edit_state::take_args(state, invocation.args.iter())?;
                    let (state, _) = edit_state::take_args(state, invocation.args.iter())?;
//

//                    let helper = LibfuncHelper {
                    let helper = LibfuncHelper {
//                        module,
                        module,
//                        init_block: &pre_entry_block,
                        init_block: &pre_entry_block,
//                        region: &region,
                        region: &region,
//                        blocks_arena: &blocks_arena,
                        blocks_arena: &blocks_arena,
//                        last_block: Cell::new(block),
                        last_block: Cell::new(block),
//                        branches: generate_branching_targets(
                        branches: generate_branching_targets(
//                            &blocks,
                            &blocks,
//                            statements,
                            statements,
//                            statement_idx,
                            statement_idx,
//                            invocation,
                            invocation,
//                            &state,
                            &state,
//                        ),
                        ),
//                        results: invocation
                        results: invocation
//                            .branches
                            .branches
//                            .iter()
                            .iter()
//                            .map(|x| vec![Cell::new(None); x.results.len()])
                            .map(|x| vec![Cell::new(None); x.results.len()])
//                            .collect::<Vec<_>>(),
                            .collect::<Vec<_>>(),
//                    };
                    };
//

//                    let concrete_libfunc = registry.get_libfunc(&invocation.libfunc_id)?;
                    let concrete_libfunc = registry.get_libfunc(&invocation.libfunc_id)?;
//                    if let Some(target) = concrete_libfunc.is_function_call() {
                    if let Some(target) = concrete_libfunc.is_function_call() {
//                        if target == &function.id && state.is_empty() {
                        if target == &function.id && state.is_empty() {
//                            // TODO: Defer insertions until after the recursion has been confirmed
                            // TODO: Defer insertions until after the recursion has been confirmed
//                            //   (when removing the meta, if a return target is set).
                            //   (when removing the meta, if a return target is set).
//                            // TODO: Explore replacing the `memref` counter with a normal variable.
                            // TODO: Explore replacing the `memref` counter with a normal variable.
//                            let location = Location::name(
                            let location = Location::name(
//                                context,
                                context,
//                                &format!("recursion_counter({})", libfunc_name),
                                &format!("recursion_counter({})", libfunc_name),
//                                Location::unknown(context),
                                Location::unknown(context),
//                            );
                            );
//                            let op0 = pre_entry_block.insert_operation(
                            let op0 = pre_entry_block.insert_operation(
//                                0,
                                0,
//                                memref::alloca(
                                memref::alloca(
//                                    context,
                                    context,
//                                    MemRefType::new(Type::index(context), &[], None, None),
                                    MemRefType::new(Type::index(context), &[], None, None),
//                                    &[],
                                    &[],
//                                    &[],
                                    &[],
//                                    None,
                                    None,
//                                    location,
                                    location,
//                                ),
                                ),
//                            );
                            );
//                            let op1 = pre_entry_block.insert_operation_after(
                            let op1 = pre_entry_block.insert_operation_after(
//                                op0,
                                op0,
//                                index::constant(
                                index::constant(
//                                    context,
                                    context,
//                                    IntegerAttribute::new(Type::index(context), 0),
                                    IntegerAttribute::new(Type::index(context), 0),
//                                    location,
                                    location,
//                                ),
                                ),
//                            );
                            );
//                            pre_entry_block.insert_operation_after(
                            pre_entry_block.insert_operation_after(
//                                op1,
                                op1,
//                                memref::store(
                                memref::store(
//                                    op1.result(0)?.into(),
                                    op1.result(0)?.into(),
//                                    op0.result(0)?.into(),
                                    op0.result(0)?.into(),
//                                    &[],
                                    &[],
//                                    location,
                                    location,
//                                ),
                                ),
//                            );
                            );
//

//                            metadata
                            metadata
//                                .insert(TailRecursionMeta::new(op0.result(0)?.into(), &entry_block))
                                .insert(TailRecursionMeta::new(op0.result(0)?.into(), &entry_block))
//                                .expect("should not have this metadata inserted yet");
                                .expect("should not have this metadata inserted yet");
//                        }
                        }
//                    }
                    }
//

//                    concrete_libfunc.build(
                    concrete_libfunc.build(
//                        context,
                        context,
//                        registry,
                        registry,
//                        block,
                        block,
//                        Location::name(
                        Location::name(
//                            context,
                            context,
//                            &libfunc_name,
                            &libfunc_name,
//                            debug_info
                            debug_info
//                                .and_then(|debug_info| {
                                .and_then(|debug_info| {
//                                    debug_info.statements.get(&statement_idx).copied()
                                    debug_info.statements.get(&statement_idx).copied()
//                                })
                                })
//                                .unwrap_or_else(|| Location::unknown(context)),
                                .unwrap_or_else(|| Location::unknown(context)),
//                        ),
                        ),
//                        &helper,
                        &helper,
//                        metadata,
                        metadata,
//                    )?;
                    )?;
//                    assert!(block.terminator().is_some());
                    assert!(block.terminator().is_some());
//

//                    if let Some(tailrec_meta) = metadata.remove::<TailRecursionMeta>() {
                    if let Some(tailrec_meta) = metadata.remove::<TailRecursionMeta>() {
//                        if let Some(return_block) = tailrec_meta.return_target() {
                        if let Some(return_block) = tailrec_meta.return_target() {
//                            tailrec_state.insert(statement_idx.0, tailrec_storage.len());
                            tailrec_state.insert(statement_idx.0, tailrec_storage.len());
//                            tailrec_storage.push((tailrec_meta.depth_counter(), return_block));
                            tailrec_storage.push((tailrec_meta.depth_counter(), return_block));
//                        }
                        }
//                    }
                    }
//

//                    invocation
                    invocation
//                        .branches
                        .branches
//                        .iter()
                        .iter()
//                        .zip(helper.results())
                        .zip(helper.results())
//                        .map(|(branch_info, result_values)| {
                        .map(|(branch_info, result_values)| {
//                            assert_eq!(
                            assert_eq!(
//                                branch_info.results.len(),
                                branch_info.results.len(),
//                                result_values.len(),
                                result_values.len(),
//                                "Mismatched number of returned values from branch."
                                "Mismatched number of returned values from branch."
//                            );
                            );
//

//                            Ok((
                            Ok((
//                                edit_state::put_results(
                                edit_state::put_results(
//                                    state.clone(),
                                    state.clone(),
//                                    branch_info
                                    branch_info
//                                        .results
                                        .results
//                                        .iter()
                                        .iter()
//                                        .zip(result_values.iter().copied()),
                                        .zip(result_values.iter().copied()),
//                                )?,
                                )?,
//                                tailrec_state.clone(),
                                tailrec_state.clone(),
//                            ))
                            ))
//                        })
                        })
//                        .collect::<Result<_, Error>>()?
                        .collect::<Result<_, Error>>()?
//                }
                }
//                Statement::Return(var_ids) => {
                Statement::Return(var_ids) => {
//                    tracing::trace!("Implementing the return statement at {statement_idx}");
                    tracing::trace!("Implementing the return statement at {statement_idx}");
//

//                    let location = Location::name(
                    let location = Location::name(
//                        context,
                        context,
//                        &format!("return(stmt_idx={})", statement_idx),
                        &format!("return(stmt_idx={})", statement_idx),
//                        Location::unknown(context),
                        Location::unknown(context),
//                    );
                    );
//

//                    let (_, mut values) = edit_state::take_args(state, var_ids.iter())?;
                    let (_, mut values) = edit_state::take_args(state, var_ids.iter())?;
//

//                    let mut block = *block;
                    let mut block = *block;
//                    if !tailrec_state.is_empty() {
                    if !tailrec_state.is_empty() {
//                        let location = Location::name(
                        let location = Location::name(
//                            context,
                            context,
//                            &format!("return(stmt_idx={}, tail_recursion)", statement_idx),
                            &format!("return(stmt_idx={}, tail_recursion)", statement_idx),
//                            Location::unknown(context),
                            Location::unknown(context),
//                        );
                        );
//                        // Perform tail recursion.
                        // Perform tail recursion.
//                        for counter_idx in tailrec_state.into_values() {
                        for counter_idx in tailrec_state.into_values() {
//                            let cont_block = region.insert_block_after(block, Block::new(&[]));
                            let cont_block = region.insert_block_after(block, Block::new(&[]));
//

//                            let (depth_counter, return_target) = tailrec_storage[counter_idx];
                            let (depth_counter, return_target) = tailrec_storage[counter_idx];
//                            let depth_counter_value = block
                            let depth_counter_value = block
//                                .append_operation(memref::load(depth_counter, &[], location))
                                .append_operation(memref::load(depth_counter, &[], location))
//                                .result(0)?
                                .result(0)?
//                                .into();
                                .into();
//                            let k0 = block
                            let k0 = block
//                                .append_operation(index::constant(
                                .append_operation(index::constant(
//                                    context,
                                    context,
//                                    IntegerAttribute::new(Type::index(context), 0),
                                    IntegerAttribute::new(Type::index(context), 0),
//                                    location,
                                    location,
//                                ))
                                ))
//                                .result(0)?
                                .result(0)?
//                                .into();
                                .into();
//                            let is_zero_depth = block
                            let is_zero_depth = block
//                                .append_operation(index::cmp(
                                .append_operation(index::cmp(
//                                    context,
                                    context,
//                                    CmpiPredicate::Eq,
                                    CmpiPredicate::Eq,
//                                    depth_counter_value,
                                    depth_counter_value,
//                                    k0,
                                    k0,
//                                    location,
                                    location,
//                                ))
                                ))
//                                .result(0)?
                                .result(0)?
//                                .into();
                                .into();
//

//                            let k1 = block
                            let k1 = block
//                                .append_operation(index::constant(
                                .append_operation(index::constant(
//                                    context,
                                    context,
//                                    IntegerAttribute::new(Type::index(context), 1),
                                    IntegerAttribute::new(Type::index(context), 1),
//                                    location,
                                    location,
//                                ))
                                ))
//                                .result(0)?
                                .result(0)?
//                                .into();
                                .into();
//                            let depth_counter_value = block
                            let depth_counter_value = block
//                                .append_operation(index::sub(depth_counter_value, k1, location))
                                .append_operation(index::sub(depth_counter_value, k1, location))
//                                .result(0)?
                                .result(0)?
//                                .into();
                                .into();
//                            block.append_operation(memref::store(
                            block.append_operation(memref::store(
//                                depth_counter_value,
                                depth_counter_value,
//                                depth_counter,
                                depth_counter,
//                                &[],
                                &[],
//                                location,
                                location,
//                            ));
                            ));
//

//                            let recursive_values = match has_return_ptr {
                            let recursive_values = match has_return_ptr {
//                                Some(true) => function
                                Some(true) => function
//                                    .signature
                                    .signature
//                                    .ret_types
                                    .ret_types
//                                    .iter()
                                    .iter()
//                                    .zip(&values)
                                    .zip(&values)
//                                    .filter_map(|(type_id, value)| {
                                    .filter_map(|(type_id, value)| {
//                                        let type_info = registry.get_type(type_id).unwrap();
                                        let type_info = registry.get_type(type_id).unwrap();
//                                        if type_info.is_zst(registry)
                                        if type_info.is_zst(registry)
//                                            || type_info.is_memory_allocated(registry)
                                            || type_info.is_memory_allocated(registry)
//                                        {
                                        {
//                                            None
                                            None
//                                        } else {
                                        } else {
//                                            Some(*value)
                                            Some(*value)
//                                        }
                                        }
//                                    })
                                    })
//                                    .collect::<Vec<_>>(),
                                    .collect::<Vec<_>>(),
//                                Some(false) => function
                                Some(false) => function
//                                    .signature
                                    .signature
//                                    .ret_types
                                    .ret_types
//                                    .iter()
                                    .iter()
//                                    .zip(&values)
                                    .zip(&values)
//                                    .filter_map(|(type_id, value)| {
                                    .filter_map(|(type_id, value)| {
//                                        let type_info = registry.get_type(type_id).unwrap();
                                        let type_info = registry.get_type(type_id).unwrap();
//                                        if type_info.is_zst(registry) {
                                        if type_info.is_zst(registry) {
//                                            None
                                            None
//                                        } else {
                                        } else {
//                                            Some(*value)
                                            Some(*value)
//                                        }
                                        }
//                                    })
                                    })
//                                    .collect::<Vec<_>>(),
                                    .collect::<Vec<_>>(),
//                                None => todo!(),
                                None => todo!(),
//                            };
                            };
//

//                            block.append_operation(cf::cond_br(
                            block.append_operation(cf::cond_br(
//                                context,
                                context,
//                                is_zero_depth,
                                is_zero_depth,
//                                &cont_block,
                                &cont_block,
//                                &return_target,
                                &return_target,
//                                &[],
                                &[],
//                                &recursive_values,
                                &recursive_values,
//                                location,
                                location,
//                            ));
                            ));
//

//                            block = cont_block;
                            block = cont_block;
//                        }
                        }
//                    }
                    }
//

//                    // Remove ZST builtins from the return values.
                    // Remove ZST builtins from the return values.
//                    for (idx, type_id) in function.signature.ret_types.iter().enumerate().rev() {
                    for (idx, type_id) in function.signature.ret_types.iter().enumerate().rev() {
//                        let type_info = registry.get_type(type_id)?;
                        let type_info = registry.get_type(type_id)?;
//                        if type_info.is_builtin() && type_info.is_zst(registry) {
                        if type_info.is_builtin() && type_info.is_zst(registry) {
//                            values.remove(idx);
                            values.remove(idx);
//                        }
                        }
//                    }
                    }
//

//                    // Store the return value in the return pointer, if there's one.
                    // Store the return value in the return pointer, if there's one.
//                    if let Some(true) = has_return_ptr {
                    if let Some(true) = has_return_ptr {
//                        let (_ret_type_id, ret_type_info) = return_type_infos[0];
                        let (_ret_type_id, ret_type_info) = return_type_infos[0];
//                        let ret_layout = ret_type_info.layout(registry)?;
                        let ret_layout = ret_type_info.layout(registry)?;
//

//                        let ptr = values.remove(0);
                        let ptr = values.remove(0);
//                        block.append_operation(llvm::store(
                        block.append_operation(llvm::store(
//                            context,
                            context,
//                            ptr,
                            ptr,
//                            pre_entry_block.argument(0)?.into(),
                            pre_entry_block.argument(0)?.into(),
//                            location,
                            location,
//                            LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                            LoadStoreOptions::new().align(Some(IntegerAttribute::new(
//                                IntegerType::new(context, 64).into(),
                                IntegerType::new(context, 64).into(),
//                                ret_layout.align() as i64,
                                ret_layout.align() as i64,
//                            ))),
                            ))),
//                        ));
                        ));
//                    }
                    }
//

//                    block.append_operation(func::r#return(&values, location));
                    block.append_operation(func::r#return(&values, location));
//

//                    Vec::new()
                    Vec::new()
//                }
                }
//            })
            })
//        },
        },
//    )?;
    )?;
//

//    // Load arguments and jump to the entry block.
    // Load arguments and jump to the entry block.
//    {
    {
//        let mut arg_values = Vec::with_capacity(function.signature.param_types.len());
        let mut arg_values = Vec::with_capacity(function.signature.param_types.len());
//        for (i, type_id_and_info) in function
        for (i, type_id_and_info) in function
//            .signature
            .signature
//            .param_types
            .param_types
//            .iter()
            .iter()
//            .filter_map(|type_id| {
            .filter_map(|type_id| {
//                registry
                registry
//                    .get_type(type_id)
                    .get_type(type_id)
//                    .map(|type_info| {
                    .map(|type_info| {
//                        if type_info.is_builtin() && type_info.is_zst(registry) {
                        if type_info.is_builtin() && type_info.is_zst(registry) {
//                            None
                            None
//                        } else {
                        } else {
//                            Some((type_id, type_info))
                            Some((type_id, type_info))
//                        }
                        }
//                    })
                    })
//                    .transpose()
                    .transpose()
//            })
            })
//            .enumerate()
            .enumerate()
//        {
        {
//            let (type_id, type_info) = type_id_and_info?;
            let (type_id, type_info) = type_id_and_info?;
//

//            let mut value = pre_entry_block
            let mut value = pre_entry_block
//                .argument((has_return_ptr == Some(true)) as usize + i)?
                .argument((has_return_ptr == Some(true)) as usize + i)?
//                .into();
                .into();
//            if type_info.is_memory_allocated(registry) {
            if type_info.is_memory_allocated(registry) {
//                value = pre_entry_block
                value = pre_entry_block
//                    .append_operation(llvm::load(
                    .append_operation(llvm::load(
//                        context,
                        context,
//                        value,
                        value,
//                        type_info.build(context, module, registry, metadata, type_id)?,
                        type_info.build(context, module, registry, metadata, type_id)?,
//                        Location::unknown(context),
                        Location::unknown(context),
//                        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            type_info.layout(registry)?.align() as i64,
                            type_info.layout(registry)?.align() as i64,
//                        ))),
                        ))),
//                    ))
                    ))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//            }
            }
//

//            arg_values.push(value);
            arg_values.push(value);
//        }
        }
//

//        pre_entry_block.append_operation(cf::br(
        pre_entry_block.append_operation(cf::br(
//            &entry_block,
            &entry_block,
//            &arg_values,
            &arg_values,
//            Location::unknown(context),
            Location::unknown(context),
//        ));
        ));
//    }
    }
//

//    let function_name = generate_function_name(&function.id);
    let function_name = generate_function_name(&function.id);
//    tracing::debug!("Creating the actual function, named `{function_name}`.");
    tracing::debug!("Creating the actual function, named `{function_name}`.");
//

//    module.body().append_operation(func::func(
    module.body().append_operation(func::func(
//        context,
        context,
//        StringAttribute::new(context, &function_name),
        StringAttribute::new(context, &function_name),
//        TypeAttribute::new(FunctionType::new(context, &arg_types, &return_types).into()),
        TypeAttribute::new(FunctionType::new(context, &arg_types, &return_types).into()),
//        region,
        region,
//        &[
        &[
//            (
            (
//                Identifier::new(context, "sym_visibility"),
                Identifier::new(context, "sym_visibility"),
//                StringAttribute::new(context, "public").into(),
                StringAttribute::new(context, "public").into(),
//            ),
            ),
//            (
            (
//                Identifier::new(context, "llvm.emit_c_interface"),
                Identifier::new(context, "llvm.emit_c_interface"),
//                Attribute::unit(context),
                Attribute::unit(context),
//            ),
            ),
//        ],
        ],
//        Location::unknown(context),
        Location::unknown(context),
//    ));
    ));
//

//    tracing::debug!("Done generating function {}.", function.id);
    tracing::debug!("Done generating function {}.", function.id);
//    Ok(())
    Ok(())
//}
}
//

//fn generate_function_structure<'c, 'a>(
fn generate_function_structure<'c, 'a>(
//    context: &'c Context,
    context: &'c Context,
//    module: &'a Module<'c>,
    module: &'a Module<'c>,
//    region: &'a Region<'c>,
    region: &'a Region<'c>,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    function: &Function,
    function: &Function,
//    statements: &[Statement],
    statements: &[Statement],
//    metadata_storage: &mut MetadataStorage,
    metadata_storage: &mut MetadataStorage,
//) -> Result<(BlockRef<'c, 'a>, BlockStorage<'c, 'a>), Error> {
) -> Result<(BlockRef<'c, 'a>, BlockStorage<'c, 'a>), Error> {
//    let initial_state = edit_state::put_results::<Type>(
    let initial_state = edit_state::put_results::<Type>(
//        HashMap::new(),
        HashMap::new(),
//        function
        function
//            .params
            .params
//            .iter()
            .iter()
//            .zip(&function.signature.param_types)
            .zip(&function.signature.param_types)
//            .map(|(param, ty)| {
            .map(|(param, ty)| {
//                let type_info = registry.get_type(ty)?;
                let type_info = registry.get_type(ty)?;
//                Ok((
                Ok((
//                    &param.id,
                    &param.id,
//                    type_info.build(context, module, registry, metadata_storage, ty)?,
                    type_info.build(context, module, registry, metadata_storage, ty)?,
//                ))
                ))
//            })
            })
//            .collect::<Result<Vec<_>, Error>>()?
            .collect::<Result<Vec<_>, Error>>()?
//            .into_iter(),
            .into_iter(),
//    )?;
    )?;
//

//    let mut blocks = BTreeMap::new();
    let mut blocks = BTreeMap::new();
//    let mut predecessors = HashMap::from([(function.entry_point, (initial_state.clone(), 0))]);
    let mut predecessors = HashMap::from([(function.entry_point, (initial_state.clone(), 0))]);
//

//    foreach_statement_in_function::<_, Error>(
    foreach_statement_in_function::<_, Error>(
//        statements,
        statements,
//        function.entry_point,
        function.entry_point,
//        initial_state,
        initial_state,
//        |statement_idx, state| {
        |statement_idx, state| {
//            let block = {
            let block = {
//                if let std::collections::btree_map::Entry::Vacant(e) = blocks.entry(statement_idx.0)
                if let std::collections::btree_map::Entry::Vacant(e) = blocks.entry(statement_idx.0)
//                {
                {
//                    e.insert(Block::new(&[]));
                    e.insert(Block::new(&[]));
//                    blocks
                    blocks
//                        .get_mut(&statement_idx.0)
                        .get_mut(&statement_idx.0)
//                        .expect("the block should exist")
                        .expect("the block should exist")
//                } else {
                } else {
//                    panic!("statement index already present in block");
                    panic!("statement index already present in block");
//                }
                }
//            };
            };
//

//            Ok(match &statements[statement_idx.0] {
            Ok(match &statements[statement_idx.0] {
//                Statement::Invocation(invocation) => {
                Statement::Invocation(invocation) => {
//                    tracing::trace!(
                    tracing::trace!(
//                        "Creating block for invocation statement at index {statement_idx}: {}",
                        "Creating block for invocation statement at index {statement_idx}: {}",
//                        invocation.libfunc_id
                        invocation.libfunc_id
//                    );
                    );
//

//                    let (state, types) =
                    let (state, types) =
//                        edit_state::take_args(state.clone(), invocation.args.iter())?;
                        edit_state::take_args(state.clone(), invocation.args.iter())?;
//

//                    for ty in types {
                    for ty in types {
//                        block.add_argument(ty, Location::unknown(context));
                        block.add_argument(ty, Location::unknown(context));
//                    }
                    }
//

//                    let libfunc = registry.get_libfunc(&invocation.libfunc_id)?;
                    let libfunc = registry.get_libfunc(&invocation.libfunc_id)?;
//                    invocation
                    invocation
//                        .branches
                        .branches
//                        .iter()
                        .iter()
//                        .zip(libfunc.branch_signatures())
                        .zip(libfunc.branch_signatures())
//                        .map(|(branch, branch_signature)| {
                        .map(|(branch, branch_signature)| {
//                            let state = edit_state::put_results(
                            let state = edit_state::put_results(
//                                state.clone(),
                                state.clone(),
//                                branch.results.iter().zip(
                                branch.results.iter().zip(
//                                    branch_signature
                                    branch_signature
//                                        .vars
                                        .vars
//                                        .iter()
                                        .iter()
//                                        .map(|var_info| -> Result<_, Error> {
                                        .map(|var_info| -> Result<_, Error> {
//                                            registry.get_type(&var_info.ty)?.build(
                                            registry.get_type(&var_info.ty)?.build(
//                                                context,
                                                context,
//                                                module,
                                                module,
//                                                registry,
                                                registry,
//                                                metadata_storage,
                                                metadata_storage,
//                                                &var_info.ty,
                                                &var_info.ty,
//                                            )
                                            )
//                                        })
                                        })
//                                        .collect::<Result<Vec<_>, _>>()?,
                                        .collect::<Result<Vec<_>, _>>()?,
//                                ),
                                ),
//                            )?;
                            )?;
//

//                            let (prev_state, pred_count) =
                            let (prev_state, pred_count) =
//                                match predecessors.entry(statement_idx.next(&branch.target)) {
                                match predecessors.entry(statement_idx.next(&branch.target)) {
//                                    Entry::Occupied(entry) => entry.into_mut(),
                                    Entry::Occupied(entry) => entry.into_mut(),
//                                    Entry::Vacant(entry) => entry.insert((state.clone(), 0)),
                                    Entry::Vacant(entry) => entry.insert((state.clone(), 0)),
//                                };
                                };
//                            assert_eq!(prev_state, &state, "Branch target states do not match.");
                            assert_eq!(prev_state, &state, "Branch target states do not match.");
//                            *pred_count += 1;
                            *pred_count += 1;
//

//                            Ok(state)
                            Ok(state)
//                        })
                        })
//                        .collect::<Result<_, Error>>()?
                        .collect::<Result<_, Error>>()?
//                }
                }
//                Statement::Return(var_ids) => {
                Statement::Return(var_ids) => {
//                    tracing::trace!(
                    tracing::trace!(
//                        "Creating block for return statement at index {statement_idx}."
                        "Creating block for return statement at index {statement_idx}."
//                    );
                    );
//

//                    let (state, types) = edit_state::take_args(state.clone(), var_ids.iter())?;
                    let (state, types) = edit_state::take_args(state.clone(), var_ids.iter())?;
//                    assert!(
                    assert!(
//                        state.is_empty(),
                        state.is_empty(),
//                        "State must be empty after a return statement."
                        "State must be empty after a return statement."
//                    );
                    );
//

//                    for ty in types {
                    for ty in types {
//                        block.add_argument(ty, Location::unknown(context));
                        block.add_argument(ty, Location::unknown(context));
//                    }
                    }
//

//                    Vec::new()
                    Vec::new()
//                }
                }
//            })
            })
//        },
        },
//    )?;
    )?;
//

//    tracing::trace!("Generating function entry block.");
    tracing::trace!("Generating function entry block.");
//    let entry_block = region.append_block(Block::new(&{
    let entry_block = region.append_block(Block::new(&{
//        extract_types(
        extract_types(
//            context,
            context,
//            module,
            module,
//            &function.signature.param_types,
            &function.signature.param_types,
//            registry,
            registry,
//            metadata_storage,
            metadata_storage,
//        )
        )
//        .map(|ty| Ok((ty?, Location::unknown(context))))
        .map(|ty| Ok((ty?, Location::unknown(context))))
//        .collect::<Result<Vec<_>, Error>>()?
        .collect::<Result<Vec<_>, Error>>()?
//    }));
    }));
//

//    let blocks = blocks
    let blocks = blocks
//        .into_iter()
        .into_iter()
//        .map(|(i, block)| {
        .map(|(i, block)| {
//            let statement_idx = StatementIdx(i);
            let statement_idx = StatementIdx(i);
//

//            tracing::trace!("Inserting block for statement at index {statement_idx}.");
            tracing::trace!("Inserting block for statement at index {statement_idx}.");
//            let libfunc_block = region.append_block(block);
            let libfunc_block = region.append_block(block);
//            let landing_block = (predecessors[&statement_idx].1 > 1).then(|| {
            let landing_block = (predecessors[&statement_idx].1 > 1).then(|| {
//                tracing::trace!(
                tracing::trace!(
//                    "Generating a landing block for the statement at index {statement_idx}."
                    "Generating a landing block for the statement at index {statement_idx}."
//                );
                );
//

//                (
                (
//                    region.insert_block_before(
                    region.insert_block_before(
//                        libfunc_block,
                        libfunc_block,
//                        Block::new(
                        Block::new(
//                            &predecessors[&statement_idx]
                            &predecessors[&statement_idx]
//                                .0
                                .0
//                                .iter()
                                .iter()
//                                .map(|(var_id, ty)| (var_id.id, *ty))
                                .map(|(var_id, ty)| (var_id.id, *ty))
//                                .collect::<BTreeMap<_, _>>()
                                .collect::<BTreeMap<_, _>>()
//                                .into_values()
                                .into_values()
//                                .map(|ty| (ty, Location::unknown(context)))
                                .map(|ty| (ty, Location::unknown(context)))
//                                .collect::<Vec<_>>(),
                                .collect::<Vec<_>>(),
//                        ),
                        ),
//                    ),
                    ),
//                    predecessors[&statement_idx]
                    predecessors[&statement_idx]
//                        .0
                        .0
//                        .clone()
                        .clone()
//                        .into_iter()
                        .into_iter()
//                        .sorted_by_key(|(k, _)| k.id)
                        .sorted_by_key(|(k, _)| k.id)
//                        .collect::<Vec<_>>(),
                        .collect::<Vec<_>>(),
//                )
                )
//            });
            });
//

//            (statement_idx, (landing_block, libfunc_block))
            (statement_idx, (landing_block, libfunc_block))
//        })
        })
//        .collect::<HashMap<_, _>>();
        .collect::<HashMap<_, _>>();
//

//    Ok((
    Ok((
//        entry_block,
        entry_block,
//        blocks
        blocks
//            .into_iter()
            .into_iter()
//            .map(|(k, v)| {
            .map(|(k, v)| {
//                (
                (
//                    k,
                    k,
//                    (
                    (
//                        v.0.map(|x| (x.0, x.1.into_iter().map(|x| x.0).collect::<Vec<_>>())),
                        v.0.map(|x| (x.0, x.1.into_iter().map(|x| x.0).collect::<Vec<_>>())),
//                        v.1,
                        v.1,
//                    ),
                    ),
//                )
                )
//            })
            })
//            .collect(),
            .collect(),
//    ))
    ))
//}
}
//

//fn extract_types<'c: 'a, 'a>(
fn extract_types<'c: 'a, 'a>(
//    context: &'c Context,
    context: &'c Context,
//    module: &'a Module<'c>,
    module: &'a Module<'c>,
//    type_ids: &'a [ConcreteTypeId],
    type_ids: &'a [ConcreteTypeId],
//    registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,
//    metadata_storage: &'a mut MetadataStorage,
    metadata_storage: &'a mut MetadataStorage,
//) -> impl 'a + Iterator<Item = Result<Type<'c>, Error>> {
) -> impl 'a + Iterator<Item = Result<Type<'c>, Error>> {
//    type_ids.iter().filter_map(|id| {
    type_ids.iter().filter_map(|id| {
//        let type_info = match registry.get_type(id) {
        let type_info = match registry.get_type(id) {
//            Ok(x) => x,
            Ok(x) => x,
//            Err(e) => return Some(Err(e.into())),
            Err(e) => return Some(Err(e.into())),
//        };
        };
//

//        if type_info.is_builtin() && type_info.is_zst(registry) {
        if type_info.is_builtin() && type_info.is_zst(registry) {
//            None
            None
//        } else {
        } else {
//            Some(type_info.build(context, module, registry, metadata_storage, id))
            Some(type_info.build(context, module, registry, metadata_storage, id))
//        }
        }
//    })
    })
//}
}
//

//fn foreach_statement_in_function<S, E>(
fn foreach_statement_in_function<S, E>(
//    statements: &[Statement],
    statements: &[Statement],
//    entry_point: StatementIdx,
    entry_point: StatementIdx,
//    initial_state: S,
    initial_state: S,
//    mut closure: impl FnMut(StatementIdx, S) -> Result<Vec<S>, E>,
    mut closure: impl FnMut(StatementIdx, S) -> Result<Vec<S>, E>,
//) -> Result<(), E>
) -> Result<(), E>
//where
where
//    S: Clone,
    S: Clone,
//{
{
//    let mut queue = vec![(entry_point, initial_state)];
    let mut queue = vec![(entry_point, initial_state)];
//    let mut visited = HashSet::new();
    let mut visited = HashSet::new();
//

//    while let Some((statement_idx, state)) = queue.pop() {
    while let Some((statement_idx, state)) = queue.pop() {
//        if !visited.insert(statement_idx) {
        if !visited.insert(statement_idx) {
//            continue;
            continue;
//        }
        }
//

//        let branch_states = closure(statement_idx, state)?;
        let branch_states = closure(statement_idx, state)?;
//

//        let branches = match &statements[statement_idx.0] {
        let branches = match &statements[statement_idx.0] {
//            Statement::Invocation(x) => x.branches.as_slice(),
            Statement::Invocation(x) => x.branches.as_slice(),
//            Statement::Return(_) => &[],
            Statement::Return(_) => &[],
//        };
        };
//        assert_eq!(
        assert_eq!(
//            branches.len(),
            branches.len(),
//            branch_states.len(),
            branch_states.len(),
//            "Returned number of states must match the number of branches."
            "Returned number of states must match the number of branches."
//        );
        );
//

//        queue.extend(
        queue.extend(
//            branches
            branches
//                .iter()
                .iter()
//                .map(|branch| statement_idx.next(&branch.target))
                .map(|branch| statement_idx.next(&branch.target))
//                .zip(branch_states),
                .zip(branch_states),
//        );
        );
//    }
    }
//

//    Ok(())
    Ok(())
//}
}
//

//fn generate_branching_targets<'ctx, 'this, 'a>(
fn generate_branching_targets<'ctx, 'this, 'a>(
//    blocks: &'this BlockStorage<'ctx, 'this>,
    blocks: &'this BlockStorage<'ctx, 'this>,
//    statements: &'this [Statement],
    statements: &'this [Statement],
//    statement_idx: StatementIdx,
    statement_idx: StatementIdx,
//    invocation: &'this Invocation,
    invocation: &'this Invocation,
//    state: &HashMap<VarId, Value<'ctx, 'this>>,
    state: &HashMap<VarId, Value<'ctx, 'this>>,
//) -> Vec<(&'this Block<'ctx>, Vec<BranchArg<'ctx, 'this>>)>
) -> Vec<(&'this Block<'ctx>, Vec<BranchArg<'ctx, 'this>>)>
//where
where
//    'this: 'ctx,
    'this: 'ctx,
//{
{
//    invocation
    invocation
//        .branches
        .branches
//        .iter()
        .iter()
//        .map(move |branch| {
        .map(move |branch| {
//            let target_idx = statement_idx.next(&branch.target);
            let target_idx = statement_idx.next(&branch.target);
//            let (landing_block, block) = &blocks[&target_idx];
            let (landing_block, block) = &blocks[&target_idx];
//

//            match landing_block {
            match landing_block {
//                Some((landing_block, state_vars)) => {
                Some((landing_block, state_vars)) => {
//                    let target_vars = state_vars
                    let target_vars = state_vars
//                        .iter()
                        .iter()
//                        .map(|var_id| {
                        .map(|var_id| {
//                            match branch.results.iter().find_position(|id| *id == var_id) {
                            match branch.results.iter().find_position(|id| *id == var_id) {
//                                Some((i, _)) => BranchArg::Returned(i),
                                Some((i, _)) => BranchArg::Returned(i),
//                                None => BranchArg::External(state[var_id]),
                                None => BranchArg::External(state[var_id]),
//                            }
                            }
//                        })
                        })
//                        .collect::<Vec<_>>();
                        .collect::<Vec<_>>();
//

//                    (landing_block.deref(), target_vars)
                    (landing_block.deref(), target_vars)
//                }
                }
//                None => {
                None => {
//                    let target_vars = match &statements[target_idx.0] {
                    let target_vars = match &statements[target_idx.0] {
//                        Statement::Invocation(x) => &x.args,
                        Statement::Invocation(x) => &x.args,
//                        Statement::Return(x) => x,
                        Statement::Return(x) => x,
//                    }
                    }
//                    .iter()
                    .iter()
//                    .map(|var_id| {
                    .map(|var_id| {
//                        match branch
                        match branch
//                            .results
                            .results
//                            .iter()
                            .iter()
//                            .enumerate()
                            .enumerate()
//                            .find_map(|(i, id)| (id == var_id).then_some(i))
                            .find_map(|(i, id)| (id == var_id).then_some(i))
//                        {
                        {
//                            Some(i) => BranchArg::Returned(i),
                            Some(i) => BranchArg::Returned(i),
//                            None => BranchArg::External(state[var_id]),
                            None => BranchArg::External(state[var_id]),
//                        }
                        }
//                    })
                    })
//                    .collect::<Vec<_>>();
                    .collect::<Vec<_>>();
//

//                    (block.deref(), target_vars)
                    (block.deref(), target_vars)
//                }
                }
//            }
            }
//        })
        })
//        .collect()
        .collect()
//}
}
