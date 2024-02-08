//! # Compilation process
//!
//! A Sierra program is compiled one function at a time. Each function has a pre-entry block that
//! will be ran only once, even in tail-recursive functions. All libfuncs are intended to place
//! their stack-allocating operations there so as to not grow the stack when recursing.
//!
//! After the pre-entry block, there is an entry block, which is in charge of preparing the first
//! statement's arguments and jumping into it. From here on, all the statements's
//! [builders](crate::libfuncs::LibfuncBuilder) are invoked. Every libfunc builder must end its
//! execution calling a branch function from the helper, which will generate the operations required
//! to jump to next statements. This simplifies the branching design, especially when a libfunc has
//! multiple target branches.
//!
//! > Note: Libfunc builders must have a branching operation out into each possible branch, even if
//!     it's unreachable. This is required to keep the state consistent. More on that later.
//!
//! Some statements do require a special landing block. Those are the ones which are the branching
//! target of more than a single statement. In other words, if a statement can be reached (directly)
//! from more than a single place, it needs a landing block.
//!
//! The landing blocks are in charge of synchronizing the Sierra state. The state is just a
//! dictionary mapping variable ids to their values. Since the values can come from a single branch,
//! this landing block is required.
//!
//! In order to generate the libfuncs's blocks, all the libfunc's entry blocks are required. That is
//! why they are generated all beforehand. The order in which they are generated follows a
//! breadth-first ordering; that is, the compiler uses a [BFS algorithm]. This algorithm should
//! generate the libfuncs in the same order as they appear in Sierra. As expected, the algorithm
//! forks the path each time a branching libfunc is found, which dies once a return statement is
//! detected.
//!
//! ## Function nomenclature transforms
//!
//! When compiling from Cairo, or from a Sierra source with debug information (the `-r` flag on
//! `cairo-compile`), those identifiers are the function's exported symbol. However, Sierra programs
//! are not required to contain that information. In those cases, the
//! (`generate_function_name`)[generate_function_name] will generate a new symbol name based on its
//! function id.
//!
//! ## Tail-recursive functions
//!
//! Part of the tail-recursion handling algorithm is implemented here, but tail-recursive functions
//! are better explained in (their metadata section)[crate::metadata::tail_recursion].
//!
//! [BFS algorithm]: https://en.wikipedia.org/wiki/Breadth-first_search

use crate::{
    debug_info::DebugLocations,
    error::{
        compile::{make_libfunc_builder_error, make_type_builder_error},
        CompileError,
    },
    libfuncs::{BranchArg, LibfuncBuilder, LibfuncHelper},
    metadata::{
        gas::{GasCost, GasMetadata},
        tail_recursion::TailRecursionMeta,
        MetadataStorage,
    },
    types::TypeBuilder,
    utils::generate_function_name,
};
use bumpalo::Bump;
use cairo_lang_sierra::{
    edit_state,
    extensions::{
        core::{CoreLibfunc, CoreType},
        gas::CostTokenType,
        ConcreteLibfunc,
    },
    ids::{ConcreteTypeId, VarId},
    program::{Function, Invocation, Program, Statement, StatementIdx},
    program_registry::ProgramRegistry,
};
use itertools::Itertools;
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf, func, index,
        llvm::{self, LoadStoreOptions},
        memref,
    },
    ir::{
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType, MemRefType},
        Attribute, Block, BlockRef, Identifier, Location, Module, Region, Type, Value,
    },
    Context,
};
use std::{
    cell::Cell,
    collections::{hash_map::Entry, BTreeMap, HashMap, HashSet},
    ops::Deref,
};

/// The [BlockStorage] type is used to map each statement into its own entry block (on the right),
/// and its landing block (on the left) if required.
///
/// The landing block contains also the variable ids that must be present when jumping into it,
/// otherwise it's a compiler error due to an inconsistent variable state.
type BlockStorage<'c, 'a> =
    HashMap<StatementIdx, (Option<(BlockRef<'c, 'a>, Vec<VarId>)>, BlockRef<'c, 'a>)>;

/// Run the compiler on a program. The compiled program is stored in the MLIR module.
///
/// The generics `TType` and `TLibfunc` contain the information required to generate the MLIR types
/// and statement operations. Most of the time you'll want to use the default ones, which are
/// [CoreType](cairo_lang_sierra::extensions::core::CoreType) and
/// [CoreLibfunc](cairo_lang_sierra::extensions::core::CoreLibfunc) respectively.
///
/// This function needs the program and the program's registry, which doesn't need to have AP
/// tracking information.
///
/// Additionally, it needs a reference to the MLIR context, the output module and the metadata
/// storage. The last one is passed externally so that stuff can be initialized if necessary.
pub fn compile(
    context: &Context,
    module: &Module,
    program: &Program,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    debug_info: Option<&DebugLocations>,
) -> Result<(), CompileError> {
    for function in &program.funcs {
        tracing::info!("Compiling function `{}`.", function.id);
        compile_func(
            context,
            module,
            registry,
            function,
            &program.statements,
            metadata,
            debug_info,
        )?;
    }

    tracing::info!("The program was compiled successfully.");
    Ok(())
}

/// Compile a single Sierra function.
///
/// The function accepts a `Function` argument, which provides the function's entry point, signature
/// and name. Check out [compile](self::compile) for a description of the other arguments.
///
/// The [module docs](self) contain more information about the compilation process.
fn compile_func(
    context: &Context,
    module: &Module,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    function: &Function,
    statements: &[Statement],
    metadata: &mut MetadataStorage,
    debug_info: Option<&DebugLocations>,
) -> Result<(), CompileError> {
    let region = Region::new();
    let blocks_arena = Bump::new();

    let mut arg_types = extract_types(
        context,
        module,
        &function.signature.param_types,
        registry,
        metadata,
    )
    .collect::<Result<Vec<_>, _>>()?;
    let mut ret_types = extract_types(
        context,
        module,
        &function.signature.ret_types,
        registry,
        metadata,
    )
    .collect::<Result<Vec<_>, _>>()?;

    // Replace memory-allocated arguments with pointers.
    for (ty, type_info) in
        arg_types
            .iter_mut()
            .zip(function.signature.param_types.iter().filter_map(|type_id| {
                let type_info = registry.get_type(type_id).unwrap();
                if type_info.is_builtin() && type_info.is_zst(registry) {
                    None
                } else {
                    Some(type_info)
                }
            }))
    {
        if type_info.is_memory_allocated(registry) {
            *ty = llvm::r#type::opaque_pointer(context);
        }
    }

    // Extract memory-allocated return types from ret_types and insert them in arg_types as a
    // pointer.
    let return_types = function
        .signature
        .ret_types
        .iter()
        .filter_map(|type_id| {
            let type_info = registry.get_type(type_id).unwrap();
            if type_info.is_builtin() && type_info.is_zst(registry) {
                None
            } else {
                Some((type_id, type_info))
            }
        })
        .collect::<Vec<_>>();
    // Possible values:
    //   None        => Doesn't return anything.
    //   Some(false) => Has a complex return type.
    //   Some(true)  => Has a manual return type which is in `arg_types[0]`.
    let has_return_ptr = if return_types.len() > 1 {
        Some(false)
    } else if return_types
        .first()
        .is_some_and(|(_, type_info)| type_info.is_memory_allocated(registry))
    {
        assert_eq!(ret_types.len(), 1);

        ret_types.remove(0);
        arg_types.insert(0, llvm::r#type::opaque_pointer(context));

        Some(true)
    } else {
        None
    };

    tracing::debug!("Generating function structure (region with blocks).");
    let (entry_block, blocks) = generate_function_structure(
        context, module, &region, registry, function, statements, metadata,
    )?;

    tracing::debug!("Generating the function implementation.");
    // Workaround for the `entry block of region may not have predecessors` error:
    let pre_entry_block = region.insert_block_before(
        entry_block,
        Block::new(
            &arg_types
                .iter()
                .map(|ty| (*ty, Location::unknown(context)))
                .collect::<Vec<_>>(),
        ),
    );

    let initial_state = edit_state::put_results(HashMap::<_, Value>::new(), {
        let mut values = Vec::new();

        let mut count = 0;
        for param in &function.params {
            let type_info = registry.get_type(&param.ty)?;

            values.push((
                &param.id,
                if type_info.is_builtin() && type_info.is_zst(registry) {
                    pre_entry_block
                        .append_operation(llvm::undef(
                            type_info
                                .build(context, module, registry, metadata, &param.ty)
                                .map_err(make_type_builder_error(&param.ty))?,
                            Location::unknown(context),
                        ))
                        .result(0)?
                        .into()
                } else {
                    let value = entry_block.argument(count)?.into();
                    count += 1;
                    value
                },
            ));
        }

        values.into_iter()
    })?;

    tracing::trace!("Implementing the entry block.");
    entry_block.append_operation(cf::br(
        &blocks[&function.entry_point].1,
        &match &statements[function.entry_point.0] {
            Statement::Invocation(x) => &x.args,
            Statement::Return(x) => x,
        }
        .iter()
        .map(|x| initial_state[x])
        .collect::<Vec<_>>(),
        Location::unknown(context),
    ));

    let mut tailrec_storage = Vec::<(Value, BlockRef)>::new();
    foreach_statement_in_function::<_, CompileError>(
        statements,
        function.entry_point,
        (initial_state, BTreeMap::<usize, usize>::new()),
        |statement_idx, (mut state, mut tailrec_state)| {
            if let Some(gas_metadata) = metadata.get::<GasMetadata>() {
                let gas_cost =
                    gas_metadata.get_gas_cost_for_statement(statement_idx, CostTokenType::Const);
                metadata.remove::<GasCost>();
                metadata.insert(GasCost(gas_cost));
            }

            let (landing_block, block) = &blocks[&statement_idx];

            if let Some((landing_block, _)) = landing_block {
                tracing::trace!("Implementing the statement {statement_idx}'s landing block.");

                state = edit_state::put_results(
                    HashMap::default(),
                    state
                        .keys()
                        .sorted_by_key(|x| x.id)
                        .enumerate()
                        .map(|(idx, var_id)| Ok((var_id, landing_block.argument(idx)?.into())))
                        .collect::<Result<Vec<_>, CompileError>>()?
                        .into_iter(),
                )?;

                landing_block.append_operation(cf::br(
                    block,
                    &edit_state::take_args(
                        state.clone(),
                        match &statements[statement_idx.0] {
                            Statement::Invocation(x) => &x.args,
                            Statement::Return(x) => x,
                        }
                        .iter(),
                    )?
                    .1,
                    Location::unknown(context),
                ));
            }

            Ok(match &statements[statement_idx.0] {
                Statement::Invocation(invocation) => {
                    tracing::trace!(
                        "Implementing the invocation statement at {statement_idx}: {}.",
                        invocation.libfunc_id
                    );

                    let (state, _) = edit_state::take_args(state, invocation.args.iter())?;

                    let helper = LibfuncHelper {
                        module,
                        init_block: &pre_entry_block,
                        region: &region,
                        blocks_arena: &blocks_arena,
                        last_block: Cell::new(block),
                        branches: generate_branching_targets(
                            &blocks,
                            statements,
                            statement_idx,
                            invocation,
                            &state,
                        ),
                        results: invocation
                            .branches
                            .iter()
                            .map(|x| vec![Cell::new(None); x.results.len()])
                            .collect::<Vec<_>>(),
                    };

                    let concrete_libfunc = registry.get_libfunc(&invocation.libfunc_id)?;
                    if let Some(target) = concrete_libfunc.is_function_call() {
                        if target == &function.id && state.is_empty() {
                            // TODO: Defer insertions until after the recursion has been confirmed
                            //   (when removing the meta, if a return target is set).
                            // TODO: Explore replacing the `memref` counter with a normal variable.
                            let op0 = pre_entry_block.insert_operation(
                                0,
                                memref::alloca(
                                    context,
                                    MemRefType::new(Type::index(context), &[], None, None),
                                    &[],
                                    &[],
                                    None,
                                    Location::unknown(context),
                                ),
                            );
                            let op1 = pre_entry_block.insert_operation_after(
                                op0,
                                index::constant(
                                    context,
                                    IntegerAttribute::new(0, Type::index(context)),
                                    Location::unknown(context),
                                ),
                            );
                            pre_entry_block.insert_operation_after(
                                op1,
                                memref::store(
                                    op1.result(0)?.into(),
                                    op0.result(0)?.into(),
                                    &[],
                                    Location::unknown(context),
                                ),
                            );

                            metadata
                                .insert(TailRecursionMeta::new(op0.result(0)?.into(), &entry_block))
                                .expect("should not have this metadata inserted yet");
                        }
                    }

                    concrete_libfunc
                        .build(
                            context,
                            registry,
                            block,
                            debug_info
                                .and_then(|debug_info| {
                                    debug_info.statements.get(&statement_idx).copied()
                                })
                                .unwrap_or_else(|| Location::unknown(context)),
                            &helper,
                            metadata,
                        )
                        .map_err(make_libfunc_builder_error(&invocation.libfunc_id))?;
                    assert!(block.terminator().is_some());

                    if let Some(tailrec_meta) = metadata.remove::<TailRecursionMeta>() {
                        if let Some(return_block) = tailrec_meta.return_target() {
                            tailrec_state.insert(statement_idx.0, tailrec_storage.len());
                            tailrec_storage.push((tailrec_meta.depth_counter(), return_block));
                        }
                    }

                    invocation
                        .branches
                        .iter()
                        .zip(helper.results())
                        .map(|(branch_info, result_values)| {
                            assert_eq!(
                                branch_info.results.len(),
                                result_values.len(),
                                "Mismatched number of returned values from branch."
                            );

                            Ok((
                                edit_state::put_results(
                                    state.clone(),
                                    branch_info
                                        .results
                                        .iter()
                                        .zip(result_values.iter().copied()),
                                )?,
                                tailrec_state.clone(),
                            ))
                        })
                        .collect::<Result<_, CompileError>>()?
                }
                Statement::Return(var_ids) => {
                    tracing::trace!("Implementing the return statement at {statement_idx}");

                    let (_, mut values) = edit_state::take_args(state, var_ids.iter())?;

                    let mut block = *block;
                    if !tailrec_state.is_empty() {
                        // Perform tail recursion.
                        for counter_idx in tailrec_state.into_values() {
                            let cont_block = region.insert_block_after(block, Block::new(&[]));

                            let (depth_counter, return_target) = tailrec_storage[counter_idx];
                            let op0 = block.append_operation(memref::load(
                                depth_counter,
                                &[],
                                Location::unknown(context),
                            ));
                            let op1 = block.append_operation(index::constant(
                                context,
                                IntegerAttribute::new(0, Type::index(context)),
                                Location::unknown(context),
                            ));
                            let op2 = block.append_operation(index::cmp(
                                context,
                                CmpiPredicate::Eq,
                                op0.result(0)?.into(),
                                op1.result(0)?.into(),
                                Location::unknown(context),
                            ));

                            let op3 = block.append_operation(index::constant(
                                context,
                                IntegerAttribute::new(1, Type::index(context)),
                                Location::unknown(context),
                            ));
                            let op4 = block.append_operation(index::sub(
                                op0.result(0)?.into(),
                                op3.result(0)?.into(),
                                Location::unknown(context),
                            ));
                            block.append_operation(memref::store(
                                op4.result(0)?.into(),
                                depth_counter,
                                &[],
                                Location::unknown(context),
                            ));

                            let recursive_values = match has_return_ptr {
                                Some(true) => function
                                    .signature
                                    .ret_types
                                    .iter()
                                    .zip(&values)
                                    .filter_map(|(type_id, value)| {
                                        let type_info = registry.get_type(type_id).unwrap();
                                        if type_info.is_zst(registry)
                                            || type_info.is_memory_allocated(registry)
                                        {
                                            None
                                        } else {
                                            Some(*value)
                                        }
                                    })
                                    .collect::<Vec<_>>(),
                                Some(false) => function
                                    .signature
                                    .ret_types
                                    .iter()
                                    .zip(&values)
                                    .filter_map(|(type_id, value)| {
                                        let type_info = registry.get_type(type_id).unwrap();
                                        if type_info.is_zst(registry) {
                                            None
                                        } else {
                                            let value = *value;

                                            Some(if type_info.is_memory_allocated(registry) {
                                                let ty = type_info
                                                    .build(
                                                        context, module, registry, metadata,
                                                        type_id,
                                                    )
                                                    .unwrap();
                                                let layout = type_info.layout(registry).unwrap();

                                                block
                                                    .append_operation(llvm::load(
                                                        context,
                                                        value,
                                                        ty,
                                                        Location::unknown(context),
                                                        LoadStoreOptions::new().align(Some(
                                                            IntegerAttribute::new(
                                                                layout.align() as i64,
                                                                IntegerType::new(context, 64)
                                                                    .into(),
                                                            ),
                                                        )),
                                                    ))
                                                    .result(0)
                                                    .unwrap()
                                                    .into()
                                            } else {
                                                value
                                            })
                                        }
                                    })
                                    .collect::<Vec<_>>(),
                                None => todo!(),
                            };

                            block.append_operation(cf::cond_br(
                                context,
                                op2.result(0)?.into(),
                                &cont_block,
                                &return_target,
                                &[],
                                &recursive_values,
                                Location::unknown(context),
                            ));

                            block = cont_block;
                        }
                    }

                    // Remove ZST builtins from the return values.
                    for (idx, type_id) in function.signature.ret_types.iter().enumerate().rev() {
                        let type_info = registry.get_type(type_id)?;
                        if type_info.is_builtin() && type_info.is_zst(registry) {
                            values.remove(idx);
                        }
                    }

                    match has_return_ptr {
                        Some(true) => {
                            let (ret_type_id, ret_type_info) = return_types[0];
                            let ret_layout = ret_type_info
                                .layout(registry)
                                .map_err(make_type_builder_error(ret_type_id))?;

                            let ptr = values.remove(0);

                            let num_bytes = block
                                .append_operation(arith::constant(
                                    context,
                                    IntegerAttribute::new(
                                        ret_layout.size() as i64,
                                        IntegerType::new(context, 64).into(),
                                    )
                                    .into(),
                                    Location::unknown(context),
                                ))
                                .result(0)?
                                .into();
                            let is_volatile = block
                                .append_operation(arith::constant(
                                    context,
                                    IntegerAttribute::new(0, IntegerType::new(context, 1).into())
                                        .into(),
                                    Location::unknown(context),
                                ))
                                .result(0)?
                                .into();

                            block.append_operation(llvm::call_intrinsic(
                                context,
                                StringAttribute::new(context, "llvm.memcpy.inline"),
                                &[
                                    pre_entry_block.argument(0)?.into(),
                                    ptr,
                                    num_bytes,
                                    is_volatile,
                                ],
                                &[],
                                Location::unknown(context),
                            ));
                        }
                        Some(false) => {
                            for (value, (type_id, type_info)) in
                                values.iter_mut().zip(&return_types)
                            {
                                if type_info.is_memory_allocated(registry) {
                                    let layout = type_info
                                        .layout(registry)
                                        .map_err(make_type_builder_error(type_id))?;

                                    *value = block
                                        .append_operation(llvm::load(
                                            context,
                                            *value,
                                            type_info
                                                .build(context, module, registry, metadata, type_id)
                                                .map_err(make_type_builder_error(type_id))?,
                                            Location::unknown(context),
                                            LoadStoreOptions::new().align(Some(
                                                IntegerAttribute::new(
                                                    layout.align() as i64,
                                                    IntegerType::new(context, 64).into(),
                                                ),
                                            )),
                                        ))
                                        .result(0)?
                                        .into();
                                }
                            }
                        }
                        None => {}
                    }

                    block.append_operation(func::r#return(&values, Location::unknown(context)));

                    Vec::new()
                }
            })
        },
    )?;

    pre_entry_block.append_operation(cf::br(
        &entry_block,
        &(0..entry_block.argument_count())
            .map(|i| {
                Ok(pre_entry_block
                    .argument((has_return_ptr == Some(true)) as usize + i)?
                    .into())
            })
            .collect::<Result<Vec<_>, CompileError>>()?,
        Location::unknown(context),
    ));

    let function_name = generate_function_name(&function.id);
    tracing::debug!("Creating the actual function, named `{function_name}`.");

    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, &function_name),
        TypeAttribute::new(FunctionType::new(context, &arg_types, &ret_types).into()),
        region,
        &[
            (
                Identifier::new(context, "sym_visibility"),
                StringAttribute::new(context, "public").into(),
            ),
            (
                Identifier::new(context, "llvm.emit_c_interface"),
                Attribute::unit(context),
            ),
        ],
        Location::unknown(context),
    ));

    tracing::debug!("Done generating function {}.", function.id);
    Ok(())
}

fn generate_function_structure<'c, 'a>(
    context: &'c Context,
    module: &'a Module<'c>,
    region: &'a Region<'c>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    function: &Function,
    statements: &[Statement],
    metadata_storage: &mut MetadataStorage,
) -> Result<(BlockRef<'c, 'a>, BlockStorage<'c, 'a>), CompileError> {
    let initial_state = edit_state::put_results::<(Type, bool)>(
        HashMap::new(),
        function
            .params
            .iter()
            .zip(&function.signature.param_types)
            .map(|(param, ty)| {
                let type_info = registry.get_type(ty)?;
                Ok((
                    &param.id,
                    (
                        type_info
                            .build(context, module, registry, metadata_storage, ty)
                            .map_err(make_type_builder_error(ty))?,
                        type_info.is_memory_allocated(registry),
                    ),
                ))
            })
            .collect::<Result<Vec<_>, CompileError>>()?
            .into_iter(),
    )?;

    let mut blocks = BTreeMap::new();
    let mut predecessors = HashMap::from([(function.entry_point, (initial_state.clone(), 0))]);

    foreach_statement_in_function::<_, CompileError>(
        statements,
        function.entry_point,
        initial_state,
        |statement_idx, state| {
            let block = {
                if let std::collections::btree_map::Entry::Vacant(e) = blocks.entry(statement_idx.0)
                {
                    e.insert(Block::new(&[]));
                    blocks
                        .get_mut(&statement_idx.0)
                        .expect("the block should exist")
                } else {
                    panic!("statement index already present in block");
                }
            };

            Ok(match &statements[statement_idx.0] {
                Statement::Invocation(invocation) => {
                    tracing::trace!(
                        "Creating block for invocation statement at index {statement_idx}: {}",
                        invocation.libfunc_id
                    );

                    let (state, types) =
                        edit_state::take_args(state.clone(), invocation.args.iter())?;

                    for (ty, is_memory_allocated) in types {
                        block.add_argument(
                            if is_memory_allocated {
                                llvm::r#type::opaque_pointer(context)
                            } else {
                                ty
                            },
                            Location::unknown(context),
                        );
                    }

                    let libfunc = registry.get_libfunc(&invocation.libfunc_id)?;
                    invocation
                        .branches
                        .iter()
                        .zip(libfunc.branch_signatures())
                        .map(|(branch, branch_signature)| {
                            let state = edit_state::put_results(
                                state.clone(),
                                branch.results.iter().zip(
                                    branch_signature
                                        .vars
                                        .iter()
                                        .map(|var_info| -> Result<_, CompileError> {
                                            let type_info = registry.get_type(&var_info.ty)?;

                                            Ok((
                                                type_info
                                                    .build(
                                                        context,
                                                        module,
                                                        registry,
                                                        metadata_storage,
                                                        &var_info.ty,
                                                    )
                                                    .map_err(make_type_builder_error(
                                                        &var_info.ty,
                                                    ))?,
                                                type_info.is_memory_allocated(registry),
                                            ))
                                        })
                                        .collect::<Result<Vec<_>, _>>()?,
                                ),
                            )?;

                            let (prev_state, pred_count) =
                                match predecessors.entry(statement_idx.next(&branch.target)) {
                                    Entry::Occupied(entry) => entry.into_mut(),
                                    Entry::Vacant(entry) => entry.insert((state.clone(), 0)),
                                };
                            assert_eq!(prev_state, &state, "Branch target states do not match.");
                            *pred_count += 1;

                            Ok(state)
                        })
                        .collect::<Result<_, CompileError>>()?
                }
                Statement::Return(var_ids) => {
                    tracing::trace!(
                        "Creating block for return statement at index {statement_idx}."
                    );

                    let (state, types) = edit_state::take_args(state.clone(), var_ids.iter())?;
                    assert!(
                        state.is_empty(),
                        "State must be empty after a return statement."
                    );

                    for (ty, is_memory_allocated) in types {
                        block.add_argument(
                            if is_memory_allocated {
                                llvm::r#type::opaque_pointer(context)
                            } else {
                                ty
                            },
                            Location::unknown(context),
                        );
                    }

                    Vec::new()
                }
            })
        },
    )?;

    tracing::trace!("Generating function entry block.");
    let entry_block = region.append_block(Block::new(&{
        let mut args = extract_types(
            context,
            module,
            &function.signature.param_types,
            registry,
            metadata_storage,
        )
        .map(|ty| Ok((ty?, Location::unknown(context))))
        .collect::<Result<Vec<_>, CompileError>>()?;

        for (type_info, (ty, _)) in function
            .signature
            .param_types
            .iter()
            .filter_map(|type_id| {
                let type_info = match registry.get_type(type_id) {
                    Ok(x) => x,
                    Err(e) => return Some(Err(e)),
                };

                if type_info.is_builtin() && type_info.is_zst(registry) {
                    None
                } else {
                    Some(Ok(type_info))
                }
            })
            .zip(args.iter_mut())
        {
            if type_info?.is_memory_allocated(registry) {
                *ty = llvm::r#type::opaque_pointer(context);
            }
        }

        args
    }));

    let blocks = blocks
        .into_iter()
        .map(|(i, block)| {
            let statement_idx = StatementIdx(i);

            tracing::trace!("Inserting block for statement at index {statement_idx}.");
            let libfunc_block = region.append_block(block);
            let landing_block = (predecessors[&statement_idx].1 > 1).then(|| {
                tracing::trace!(
                    "Generating a landing block for the statement at index {statement_idx}."
                );

                (
                    region.insert_block_before(
                        libfunc_block,
                        Block::new(
                            &predecessors[&statement_idx]
                                .0
                                .iter()
                                .map(|(var_id, ty)| (var_id.id, *ty))
                                .collect::<BTreeMap<_, _>>()
                                .into_values()
                                .map(|(ty, is_memory_allocated)| {
                                    (
                                        if is_memory_allocated {
                                            llvm::r#type::opaque_pointer(context)
                                        } else {
                                            ty
                                        },
                                        Location::unknown(context),
                                    )
                                })
                                .collect::<Vec<_>>(),
                        ),
                    ),
                    predecessors[&statement_idx]
                        .0
                        .clone()
                        .into_iter()
                        .sorted_by_key(|(k, _)| k.id)
                        .collect::<Vec<_>>(),
                )
            });

            (statement_idx, (landing_block, libfunc_block))
        })
        .collect::<HashMap<_, _>>();

    Ok((
        entry_block,
        blocks
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    (
                        v.0.map(|x| (x.0, x.1.into_iter().map(|x| x.0).collect::<Vec<_>>())),
                        v.1,
                    ),
                )
            })
            .collect(),
    ))
}

fn extract_types<'c: 'a, 'a>(
    context: &'c Context,
    module: &'a Module<'c>,
    type_ids: &'a [ConcreteTypeId],
    registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,
    metadata_storage: &'a mut MetadataStorage,
) -> impl 'a + Iterator<Item = Result<Type<'c>, CompileError>> {
    type_ids.iter().filter_map(|id| {
        let type_info = match registry.get_type(id) {
            Ok(x) => x,
            Err(e) => return Some(Err(e.into())),
        };

        if type_info.is_builtin() && type_info.is_zst(registry) {
            None
        } else {
            Some(
                type_info
                    .build(context, module, registry, metadata_storage, id)
                    .map_err(make_type_builder_error(id)),
            )
        }
    })
}

fn foreach_statement_in_function<S, E>(
    statements: &[Statement],
    entry_point: StatementIdx,
    initial_state: S,
    mut closure: impl FnMut(StatementIdx, S) -> Result<Vec<S>, E>,
) -> Result<(), E>
where
    S: Clone,
{
    let mut queue = vec![(entry_point, initial_state)];
    let mut visited = HashSet::new();

    while let Some((statement_idx, state)) = queue.pop() {
        if !visited.insert(statement_idx) {
            continue;
        }

        let branch_states = closure(statement_idx, state)?;

        let branches = match &statements[statement_idx.0] {
            Statement::Invocation(x) => x.branches.as_slice(),
            Statement::Return(_) => &[],
        };
        assert_eq!(
            branches.len(),
            branch_states.len(),
            "Returned number of states must match the number of branches."
        );

        queue.extend(
            branches
                .iter()
                .map(|branch| statement_idx.next(&branch.target))
                .zip(branch_states),
        );
    }

    Ok(())
}

fn generate_branching_targets<'ctx, 'this, 'a>(
    blocks: &'this BlockStorage<'ctx, 'this>,
    statements: &'this [Statement],
    statement_idx: StatementIdx,
    invocation: &'this Invocation,
    state: &HashMap<VarId, Value<'ctx, 'this>>,
) -> Vec<(&'this Block<'ctx>, Vec<BranchArg<'ctx, 'this>>)>
where
    'this: 'ctx,
{
    invocation
        .branches
        .iter()
        .map(move |branch| {
            let target_idx = statement_idx.next(&branch.target);
            let (landing_block, block) = &blocks[&target_idx];

            match landing_block {
                Some((landing_block, state_vars)) => {
                    let target_vars = state_vars
                        .iter()
                        .map(|var_id| {
                            match branch.results.iter().find_position(|id| *id == var_id) {
                                Some((i, _)) => BranchArg::Returned(i),
                                None => BranchArg::External(state[var_id]),
                            }
                        })
                        .collect::<Vec<_>>();

                    (landing_block.deref(), target_vars)
                }
                None => {
                    let target_vars = match &statements[target_idx.0] {
                        Statement::Invocation(x) => &x.args,
                        Statement::Return(x) => x,
                    }
                    .iter()
                    .map(|var_id| {
                        match branch
                            .results
                            .iter()
                            .enumerate()
                            .find_map(|(i, id)| (id == var_id).then_some(i))
                        {
                            Some(i) => BranchArg::Returned(i),
                            None => BranchArg::External(state[var_id]),
                        }
                    })
                    .collect::<Vec<_>>();

                    (block.deref(), target_vars)
                }
            }
        })
        .collect()
}
