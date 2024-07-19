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
    error::Error,
    ffi::{
        mlirLLVMDIBasicTypeAttrGet, mlirLLVMDICompileUnitAttrGet, mlirLLVMDIFileAttrGet, mlirLLVMDIModuleAttrGet, mlirLLVMDIModuleAttrGetScope, mlirLLVMDISubprogramAttrGet, mlirLLVMDISubroutineTypeAttrGet, mlirLLVMDistinctAttrCreate
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
        array::ArrayConcreteLibfunc,
        boolean::BoolConcreteLibfunc,
        boxing::BoxConcreteLibfunc,
        bytes31::Bytes31ConcreteLibfunc,
        casts::CastConcreteLibfunc,
        const_type::ConstConcreteLibfunc,
        core::{CoreConcreteLibfunc, CoreLibfunc, CoreType},
        coupon::CouponConcreteLibfunc,
        debug::DebugConcreteLibfunc,
        ec::EcConcreteLibfunc,
        enm::EnumConcreteLibfunc,
        felt252::{Felt252BinaryOperationConcrete, Felt252BinaryOperator, Felt252Concrete},
        felt252_dict::{Felt252DictConcreteLibfunc, Felt252DictEntryConcreteLibfunc},
        gas::GasConcreteLibfunc,
        int::{
            signed::SintConcrete, signed128::Sint128Concrete, unsigned::UintConcrete,
            unsigned128::Uint128Concrete, unsigned256::Uint256Concrete,
            unsigned512::Uint512Concrete, IntOperator,
        },
        mem::MemConcreteLibfunc,
        nullable::NullableConcreteLibfunc,
        pedersen::PedersenConcreteLibfunc,
        poseidon::PoseidonConcreteLibfunc,
        starknet::{
            secp256::{Secp256ConcreteLibfunc, Secp256OpConcreteLibfunc},
            testing::TestingConcreteLibfunc,
            StarkNetConcreteLibfunc,
        },
        structure::StructConcreteLibfunc,
        ConcreteLibfunc,
    },
    ids::{ConcreteTypeId, VarId},
    program::{Function, Invocation, Program, Statement, StatementIdx},
    program_registry::ProgramRegistry,
};
use itertools::Itertools;
use melior::{
    dialect::{
        arith::CmpiPredicate,
        cf, func, index,
        llvm::{self, LoadStoreOptions},
        memref,
    },
    ir::{
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType, MemRefType},
        Attribute, AttributeLike, Block, BlockRef, Identifier, Location, Module, Region, Type,
        Value,
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
    di_compile_unit_id: Attribute,
) -> Result<(), Error> {
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
            di_compile_unit_id
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
    di_compile_unit_id: Attribute,
) -> Result<(), Error> {
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
    let mut return_types = extract_types(
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
            *ty = llvm::r#type::pointer(context, 0);
        }
    }

    // Extract memory-allocated return types from return_types and insert them in arg_types as a
    // pointer.
    let return_type_infos = function
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
    let has_return_ptr = if return_type_infos.len() > 1 {
        Some(false)
    } else if return_type_infos
        .first()
        .is_some_and(|(_, type_info)| type_info.is_memory_allocated(registry))
    {
        assert_eq!(return_types.len(), 1);

        return_types.remove(0);
        arg_types.insert(0, llvm::r#type::pointer(context, 0));

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
                .map(|ty| (*ty, Location::new(context, "program.sierra", 0, 0)))
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
                            type_info.build(context, module, registry, metadata, &param.ty)?,
                            Location::new(context, "program.sierra", 0, 0),
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
        Location::new(context, "program.sierra", 0, 0),
    ));

    let mut tailrec_storage = Vec::<(Value, BlockRef)>::new();
    foreach_statement_in_function::<_, Error>(
        statements,
        function.entry_point,
        (initial_state, BTreeMap::<usize, usize>::new()),
        |statement_idx, (mut state, mut tailrec_state)| {
            if let Some(gas_metadata) = metadata.get::<GasMetadata>() {
                let gas_cost = gas_metadata.get_gas_cost_for_statement(statement_idx);
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
                        .collect::<Result<Vec<_>, Error>>()?
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
                    Location::name(
                        context,
                        &format!("landing_block(stmt_idx={})", statement_idx),
                        Location::new(context, "program.sierra", 0, 0),
                    ),
                ));
            }

            Ok(match &statements[statement_idx.0] {
                Statement::Invocation(invocation) => {
                    tracing::trace!(
                        "Implementing the invocation statement at {statement_idx}: {}.",
                        invocation.libfunc_id
                    );

                    let libfunc_name = if invocation.libfunc_id.debug_name.is_some() {
                        format!("{}(stmt_idx={})", invocation.libfunc_id, statement_idx)
                    } else {
                        let libf = registry.get_libfunc(&invocation.libfunc_id)?;
                        format!("{}(stmt_idx={})", libfunc_to_name(libf), statement_idx)
                    };

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
                            let location = Location::name(
                                context,
                                &format!("recursion_counter({})", libfunc_name),
                                Location::new(context, "program.sierra", 0, 0),
                            );
                            let op0 = pre_entry_block.insert_operation(
                                0,
                                memref::alloca(
                                    context,
                                    MemRefType::new(Type::index(context), &[], None, None),
                                    &[],
                                    &[],
                                    None,
                                    location,
                                ),
                            );
                            let op1 = pre_entry_block.insert_operation_after(
                                op0,
                                index::constant(
                                    context,
                                    IntegerAttribute::new(Type::index(context), 0),
                                    location,
                                ),
                            );
                            pre_entry_block.insert_operation_after(
                                op1,
                                memref::store(
                                    op1.result(0)?.into(),
                                    op0.result(0)?.into(),
                                    &[],
                                    location,
                                ),
                            );

                            metadata
                                .insert(TailRecursionMeta::new(op0.result(0)?.into(), &entry_block))
                                .expect("should not have this metadata inserted yet");
                        }
                    }

                    concrete_libfunc.build(
                        context,
                        registry,
                        block,
                        Location::name(
                            context,
                            &libfunc_name,
                            debug_info
                                .and_then(|debug_info| {
                                    debug_info.statements.get(&statement_idx).copied()
                                })
                                .unwrap_or_else(|| Location::new(context, "program.sierra", 0, 0)),
                        ),
                        &helper,
                        metadata,
                    )?;
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
                        .collect::<Result<_, Error>>()?
                }
                Statement::Return(var_ids) => {
                    tracing::trace!("Implementing the return statement at {statement_idx}");

                    let location = Location::name(
                        context,
                        &format!("return(stmt_idx={})", statement_idx),
                        Location::new(context, "program.sierra", 0, 0),
                    );

                    let (_, mut values) = edit_state::take_args(state, var_ids.iter())?;

                    let mut block = *block;
                    if !tailrec_state.is_empty() {
                        let location = Location::name(
                            context,
                            &format!("return(stmt_idx={}, tail_recursion)", statement_idx),
                            Location::new(context, "program.sierra", 0, 0),
                        );
                        // Perform tail recursion.
                        for counter_idx in tailrec_state.into_values() {
                            let cont_block = region.insert_block_after(block, Block::new(&[]));

                            let (depth_counter, return_target) = tailrec_storage[counter_idx];
                            let depth_counter_value = block
                                .append_operation(memref::load(depth_counter, &[], location))
                                .result(0)?
                                .into();
                            let k0 = block
                                .append_operation(index::constant(
                                    context,
                                    IntegerAttribute::new(Type::index(context), 0),
                                    location,
                                ))
                                .result(0)?
                                .into();
                            let is_zero_depth = block
                                .append_operation(index::cmp(
                                    context,
                                    CmpiPredicate::Eq,
                                    depth_counter_value,
                                    k0,
                                    location,
                                ))
                                .result(0)?
                                .into();

                            let k1 = block
                                .append_operation(index::constant(
                                    context,
                                    IntegerAttribute::new(Type::index(context), 1),
                                    location,
                                ))
                                .result(0)?
                                .into();
                            let depth_counter_value = block
                                .append_operation(index::sub(depth_counter_value, k1, location))
                                .result(0)?
                                .into();
                            block.append_operation(memref::store(
                                depth_counter_value,
                                depth_counter,
                                &[],
                                location,
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
                                            Some(*value)
                                        }
                                    })
                                    .collect::<Vec<_>>(),
                                None => todo!(),
                            };

                            block.append_operation(cf::cond_br(
                                context,
                                is_zero_depth,
                                &cont_block,
                                &return_target,
                                &[],
                                &recursive_values,
                                location,
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

                    // Store the return value in the return pointer, if there's one.
                    if let Some(true) = has_return_ptr {
                        let (_ret_type_id, ret_type_info) = return_type_infos[0];
                        let ret_layout = ret_type_info.layout(registry)?;

                        let ptr = values.remove(0);
                        block.append_operation(llvm::store(
                            context,
                            ptr,
                            pre_entry_block.argument(0)?.into(),
                            location,
                            LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                                IntegerType::new(context, 64).into(),
                                ret_layout.align() as i64,
                            ))),
                        ));
                    }

                    block.append_operation(func::r#return(&values, location));

                    Vec::new()
                }
            })
        },
    )?;

    // Load arguments and jump to the entry block.
    {
        let mut arg_values = Vec::with_capacity(function.signature.param_types.len());
        for (i, type_id_and_info) in function
            .signature
            .param_types
            .iter()
            .filter_map(|type_id| {
                registry
                    .get_type(type_id)
                    .map(|type_info| {
                        if type_info.is_builtin() && type_info.is_zst(registry) {
                            None
                        } else {
                            Some((type_id, type_info))
                        }
                    })
                    .transpose()
            })
            .enumerate()
        {
            let (type_id, type_info) = type_id_and_info?;

            let mut value = pre_entry_block
                .argument((has_return_ptr == Some(true)) as usize + i)?
                .into();
            if type_info.is_memory_allocated(registry) {
                value = pre_entry_block
                    .append_operation(llvm::load(
                        context,
                        value,
                        type_info.build(context, module, registry, metadata, type_id)?,
                        Location::new(context, "program.sierra", 0, 0),
                        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                            IntegerType::new(context, 64).into(),
                            type_info.layout(registry)?.align() as i64,
                        ))),
                    ))
                    .result(0)?
                    .into();
            }

            arg_values.push(value);
        }

        pre_entry_block.append_operation(cf::br(
            &entry_block,
            &arg_values,
            Location::new(context, "program.sierra", 0, 0),
        ));
    }

    let function_name = generate_function_name(&function.id);
    tracing::debug!("Creating the actual function, named `{function_name}`.");

    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, &function_name),
        TypeAttribute::new(FunctionType::new(context, &arg_types, &return_types).into()),
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
        Location::fused(
            context,
            &[Location::new(context, "program.sierra", 0, 0)],
            {
                let file_attr = unsafe {
                    Attribute::from_raw(mlirLLVMDIFileAttrGet(
                        context.to_raw(),
                        StringAttribute::new(context, "program.sierra").to_raw(),
                        StringAttribute::new(context, "").to_raw(),
                    ))
                };
                let compile_unit = {
                    let id = unsafe {
                        let id = StringAttribute::new(context, "compile_unit_id").to_raw();
                        mlirLLVMDistinctAttrCreate(id)
                    };
                    unsafe {
                        Attribute::from_raw(mlirLLVMDICompileUnitAttrGet(
                            context.to_raw(),
                            id,
                            0x1c,
                            file_attr.to_raw(),
                            StringAttribute::new(context, "cairo-native").to_raw(),
                            false,
                            crate::ffi::DiEmissionKind::Full,
                        ))
                    }
                };

                let di_module = unsafe { mlirLLVMDIModuleAttrGet(
                    context.to_raw(),
                    file_attr.to_raw(),
                    compile_unit.to_raw(),
                    StringAttribute::new(context, "LLVMDialectModule").to_raw(),
                    StringAttribute::new(context, "").to_raw(),
                    StringAttribute::new(context, "").to_raw(),
                    StringAttribute::new(context, "").to_raw(),
                    0,
                    false,
                ) };

                let scope = unsafe {
                    mlirLLVMDIModuleAttrGetScope(di_module)
                };

                let x = unsafe {
                    let id = {
                        let id = StringAttribute::new(context, "fnid").to_raw();
                        mlirLLVMDistinctAttrCreate(id)
                    };

                    let basic_ty = mlirLLVMDIBasicTypeAttrGet(
                        context.to_raw(),
                        0x24,
                        StringAttribute::new(context, "mytype").to_raw(),
                        64,
                        0x5,
                    );

                    let ty = mlirLLVMDISubroutineTypeAttrGet(
                        context.to_raw(),
                        0x0,
                        1,
                        [basic_ty].as_ptr(),
                    );

                    mlirLLVMDISubprogramAttrGet(
                        context.to_raw(),
                        id,
                        scope,
                        file_attr.to_raw(),
                        StringAttribute::new(context, &function_name).to_raw(),
                        StringAttribute::new(context, &function_name).to_raw(),
                        file_attr.to_raw(),
                        1,
                        2,
                        8,
                        ty,
                    )
                };

                unsafe { Attribute::from_raw(x) }
            },
        ),
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
) -> Result<(BlockRef<'c, 'a>, BlockStorage<'c, 'a>), Error> {
    let initial_state = edit_state::put_results::<Type>(
        HashMap::new(),
        function
            .params
            .iter()
            .zip(&function.signature.param_types)
            .map(|(param, ty)| {
                let type_info = registry.get_type(ty)?;
                Ok((
                    &param.id,
                    type_info.build(context, module, registry, metadata_storage, ty)?,
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?
            .into_iter(),
    )?;

    let mut blocks = BTreeMap::new();
    let mut predecessors = HashMap::from([(function.entry_point, (initial_state.clone(), 0))]);

    foreach_statement_in_function::<_, Error>(
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

                    for ty in types {
                        block.add_argument(ty, Location::new(context, "program.sierra", 0, 0));
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
                                        .map(|var_info| -> Result<_, Error> {
                                            registry.get_type(&var_info.ty)?.build(
                                                context,
                                                module,
                                                registry,
                                                metadata_storage,
                                                &var_info.ty,
                                            )
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
                        .collect::<Result<_, Error>>()?
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

                    for ty in types {
                        block.add_argument(ty, Location::new(context, "program.sierra", 0, 0));
                    }

                    Vec::new()
                }
            })
        },
    )?;

    tracing::trace!("Generating function entry block.");
    let entry_block = region.append_block(Block::new(&{
        extract_types(
            context,
            module,
            &function.signature.param_types,
            registry,
            metadata_storage,
        )
        .map(|ty| Ok((ty?, Location::new(context, "program.sierra", 0, 0))))
        .collect::<Result<Vec<_>, Error>>()?
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
                                .map(|ty| (ty, Location::new(context, "program.sierra", 0, 0)))
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
) -> impl 'a + Iterator<Item = Result<Type<'c>, Error>> {
    type_ids.iter().filter_map(|id| {
        let type_info = match registry.get_type(id) {
            Ok(x) => x,
            Err(e) => return Some(Err(e.into())),
        };

        if type_info.is_builtin() && type_info.is_zst(registry) {
            None
        } else {
            Some(type_info.build(context, module, registry, metadata_storage, id))
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

fn libfunc_to_name(value: &CoreConcreteLibfunc) -> &'static str {
    match value {
        CoreConcreteLibfunc::ApTracking(value) => match value {
            cairo_lang_sierra::extensions::ap_tracking::ApTrackingConcreteLibfunc::Revoke(_) => {
                "revoke_ap_tracking"
            }
            cairo_lang_sierra::extensions::ap_tracking::ApTrackingConcreteLibfunc::Enable(_) => {
                "enable_ap_tracking"
            }
            cairo_lang_sierra::extensions::ap_tracking::ApTrackingConcreteLibfunc::Disable(_) => {
                "disable_ap_tracking"
            }
        },
        CoreConcreteLibfunc::Array(value) => match value {
            ArrayConcreteLibfunc::New(_) => "array_new",
            ArrayConcreteLibfunc::SpanFromTuple(_) => "span_from_tuple",
            ArrayConcreteLibfunc::Append(_) => "array_append",
            ArrayConcreteLibfunc::PopFront(_) => "array_pop_front",
            ArrayConcreteLibfunc::PopFrontConsume(_) => "array_pop_front_consume",
            ArrayConcreteLibfunc::Get(_) => "array_get",
            ArrayConcreteLibfunc::Slice(_) => "array_slice",
            ArrayConcreteLibfunc::Len(_) => "array_len",
            ArrayConcreteLibfunc::SnapshotPopFront(_) => "array_snapshot_pop_front",
            ArrayConcreteLibfunc::SnapshotPopBack(_) => "array_snapshot_pop_back",
        },
        CoreConcreteLibfunc::BranchAlign(_) => "branch_align",
        CoreConcreteLibfunc::Bool(value) => match value {
            BoolConcreteLibfunc::And(_) => "bool_and",
            BoolConcreteLibfunc::Not(_) => "bool_not",
            BoolConcreteLibfunc::Xor(_) => "bool_xor",
            BoolConcreteLibfunc::Or(_) => "bool_or",
            BoolConcreteLibfunc::ToFelt252(_) => "bool_to_felt252",
        },
        CoreConcreteLibfunc::Box(value) => match value {
            BoxConcreteLibfunc::Into(_) => "box_into",
            BoxConcreteLibfunc::Unbox(_) => "box_unbox",
            BoxConcreteLibfunc::ForwardSnapshot(_) => "box_forward_snapshot",
        },
        CoreConcreteLibfunc::Cast(value) => match value {
            CastConcreteLibfunc::Downcast(_) => "downcast",
            CastConcreteLibfunc::Upcast(_) => "upcast",
        },
        CoreConcreteLibfunc::Coupon(value) => match value {
            CouponConcreteLibfunc::Buy(_) => "coupon_buy",
            CouponConcreteLibfunc::Refund(_) => "coupon_refund",
        },
        CoreConcreteLibfunc::CouponCall(_) => "coupon_call",
        CoreConcreteLibfunc::Drop(_) => "drop",
        CoreConcreteLibfunc::Dup(_) => "dup",
        CoreConcreteLibfunc::Ec(value) => match value {
            EcConcreteLibfunc::IsZero(_) => "ec_is_zero",
            EcConcreteLibfunc::Neg(_) => "ec_neg",
            EcConcreteLibfunc::StateAdd(_) => "ec_state_add",
            EcConcreteLibfunc::TryNew(_) => "ec_try_new",
            EcConcreteLibfunc::StateFinalize(_) => "ec_state_finalize",
            EcConcreteLibfunc::StateInit(_) => "ec_state_init",
            EcConcreteLibfunc::StateAddMul(_) => "ec_state_add_mul",
            EcConcreteLibfunc::PointFromX(_) => "ec_point_from_x",
            EcConcreteLibfunc::UnwrapPoint(_) => "ec_unwrap_point",
            EcConcreteLibfunc::Zero(_) => "ec_zero",
        },
        CoreConcreteLibfunc::Felt252(value) => match value {
            Felt252Concrete::BinaryOperation(op) => match op {
                Felt252BinaryOperationConcrete::WithVar(op) => match &op.operator {
                    Felt252BinaryOperator::Add => "felt252_add",
                    Felt252BinaryOperator::Sub => "felt252_sub",
                    Felt252BinaryOperator::Mul => "felt252_mul",
                    Felt252BinaryOperator::Div => "felt252_div",
                },
                Felt252BinaryOperationConcrete::WithConst(op) => match &op.operator {
                    Felt252BinaryOperator::Add => "felt252_const_add",
                    Felt252BinaryOperator::Sub => "felt252_const_sub",
                    Felt252BinaryOperator::Mul => "felt252_const_mul",
                    Felt252BinaryOperator::Div => "felt252_const_div",
                },
            },
            Felt252Concrete::Const(_) => "felt252_const",
            Felt252Concrete::IsZero(_) => "felt252_is_zero",
        },
        CoreConcreteLibfunc::Const(value) => match value {
            ConstConcreteLibfunc::AsBox(_) => "const_as_box",
            ConstConcreteLibfunc::AsImmediate(_) => "const_as_immediate",
        },
        CoreConcreteLibfunc::FunctionCall(_) => "function_call",
        CoreConcreteLibfunc::Gas(value) => match value {
            GasConcreteLibfunc::WithdrawGas(_) => "withdraw_gas",
            GasConcreteLibfunc::RedepositGas(_) => "redeposit_gas",
            GasConcreteLibfunc::GetAvailableGas(_) => "get_available_gas",
            GasConcreteLibfunc::BuiltinWithdrawGas(_) => "builtin_withdraw_gas",
            GasConcreteLibfunc::GetBuiltinCosts(_) => "get_builtin_costs",
        },
        CoreConcreteLibfunc::Uint8(value) => match value {
            UintConcrete::Const(_) => "u8_const",
            UintConcrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "u8_overflowing_add",
                IntOperator::OverflowingSub => "u8_overflowing_sub",
            },
            UintConcrete::SquareRoot(_) => "u8_sqrt",
            UintConcrete::Equal(_) => "u8_eq",
            UintConcrete::ToFelt252(_) => "u8_to_felt252",
            UintConcrete::FromFelt252(_) => "u8_from_felt252",
            UintConcrete::IsZero(_) => "u8_is_zero",
            UintConcrete::Divmod(_) => "u8_divmod",
            UintConcrete::WideMul(_) => "u8_wide_mul",
            UintConcrete::Bitwise(_) => "u8_bitwise",
        },
        CoreConcreteLibfunc::Uint16(value) => match value {
            UintConcrete::Const(_) => "u16_const",
            UintConcrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "u16_overflowing_add",
                IntOperator::OverflowingSub => "u16_overflowing_sub",
            },
            UintConcrete::SquareRoot(_) => "u16_sqrt",
            UintConcrete::Equal(_) => "u16_eq",
            UintConcrete::ToFelt252(_) => "u16_to_felt252",
            UintConcrete::FromFelt252(_) => "u16_from_felt252",
            UintConcrete::IsZero(_) => "u16_is_zero",
            UintConcrete::Divmod(_) => "u16_divmod",
            UintConcrete::WideMul(_) => "u16_wide_mul",
            UintConcrete::Bitwise(_) => "u16_bitwise",
        },
        CoreConcreteLibfunc::Uint32(value) => match value {
            UintConcrete::Const(_) => "u32_const",
            UintConcrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "u32_overflowing_add",
                IntOperator::OverflowingSub => "u32_overflowing_sub",
            },
            UintConcrete::SquareRoot(_) => "u32_sqrt",
            UintConcrete::Equal(_) => "u32_eq",
            UintConcrete::ToFelt252(_) => "u32_to_felt252",
            UintConcrete::FromFelt252(_) => "u32_from_felt252",
            UintConcrete::IsZero(_) => "u32_is_zero",
            UintConcrete::Divmod(_) => "u32_divmod",
            UintConcrete::WideMul(_) => "u32_wide_mul",
            UintConcrete::Bitwise(_) => "u32_bitwise",
        },
        CoreConcreteLibfunc::Uint64(value) => match value {
            UintConcrete::Const(_) => "u64_const",
            UintConcrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "u64_overflowing_add",
                IntOperator::OverflowingSub => "u64_overflowing_sub",
            },
            UintConcrete::SquareRoot(_) => "u64_sqrt",
            UintConcrete::Equal(_) => "u64_eq",
            UintConcrete::ToFelt252(_) => "u64_to_felt252",
            UintConcrete::FromFelt252(_) => "u64_from_felt252",
            UintConcrete::IsZero(_) => "u64_is_zero",
            UintConcrete::Divmod(_) => "u64_divmod",
            UintConcrete::WideMul(_) => "u64_wide_mul",
            UintConcrete::Bitwise(_) => "u64_bitwise",
        },
        CoreConcreteLibfunc::Uint128(value) => match value {
            Uint128Concrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "u128_overflowing_add",
                IntOperator::OverflowingSub => "u128_overflowing_sub",
            },
            Uint128Concrete::Divmod(_) => "u128_divmod",
            Uint128Concrete::GuaranteeMul(_) => "u128_guarantee_mul",
            Uint128Concrete::MulGuaranteeVerify(_) => "u128_mul_guarantee_verify",
            Uint128Concrete::Equal(_) => "u128_equal",
            Uint128Concrete::SquareRoot(_) => "u128_sqrt",
            Uint128Concrete::Const(_) => "u128_const",
            Uint128Concrete::FromFelt252(_) => "u128_from_felt",
            Uint128Concrete::ToFelt252(_) => "u128_to_felt252",
            Uint128Concrete::IsZero(_) => "u128_is_zero",
            Uint128Concrete::Bitwise(_) => "u128_bitwise",
            Uint128Concrete::ByteReverse(_) => "u128_bytereverse",
        },
        CoreConcreteLibfunc::Uint256(value) => match value {
            Uint256Concrete::IsZero(_) => "u256_is_zero",
            Uint256Concrete::Divmod(_) => "u256_divmod",
            Uint256Concrete::SquareRoot(_) => "u256_sqrt",
            Uint256Concrete::InvModN(_) => "u256_inv_mod_n",
        },
        CoreConcreteLibfunc::Uint512(value) => match value {
            Uint512Concrete::DivModU256(_) => "u512_divmod_u256",
        },
        CoreConcreteLibfunc::Sint8(value) => match value {
            SintConcrete::Const(_) => "i8_const",
            SintConcrete::Equal(_) => "i8_eq",
            SintConcrete::ToFelt252(_) => "i8_to_felt252",
            SintConcrete::FromFelt252(_) => "i8_from_felt252",
            SintConcrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "i8_overflowing_add",
                IntOperator::OverflowingSub => "i8_overflowing_sub",
            },
            SintConcrete::Diff(_) => "i8_diff",
            SintConcrete::IsZero(_) => "i8_is_zero",
            SintConcrete::WideMul(_) => "i8_wide_mul",
        },
        CoreConcreteLibfunc::Sint16(value) => match value {
            SintConcrete::Const(_) => "i16_const",
            SintConcrete::Equal(_) => "i16_eq",
            SintConcrete::ToFelt252(_) => "i16_to_felt252",
            SintConcrete::FromFelt252(_) => "i16_from_felt252",
            SintConcrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "i16_overflowing_add",
                IntOperator::OverflowingSub => "i16_overflowing_sub",
            },
            SintConcrete::Diff(_) => "i16_diff",
            SintConcrete::IsZero(_) => "i16_is_zero",
            SintConcrete::WideMul(_) => "i16_wide_mul",
        },
        CoreConcreteLibfunc::Sint32(value) => match value {
            SintConcrete::Const(_) => "i32_const",
            SintConcrete::Equal(_) => "i32_eq",
            SintConcrete::ToFelt252(_) => "i32_to_felt252",
            SintConcrete::FromFelt252(_) => "i32_from_felt252",
            SintConcrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "i32_overflowing_add",
                IntOperator::OverflowingSub => "i32_overflowing_sub",
            },
            SintConcrete::Diff(_) => "i32_diff",
            SintConcrete::IsZero(_) => "i32_is_zero",
            SintConcrete::WideMul(_) => "i32_wide_mul",
        },
        CoreConcreteLibfunc::Sint64(value) => match value {
            SintConcrete::Const(_) => "i64_const",
            SintConcrete::Equal(_) => "i64_eq",
            SintConcrete::ToFelt252(_) => "i64_to_felt252",
            SintConcrete::FromFelt252(_) => "i64_from_felt252",
            SintConcrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "i64_overflowing_add",
                IntOperator::OverflowingSub => "i64_overflowing_sub",
            },
            SintConcrete::Diff(_) => "i64_diff",
            SintConcrete::IsZero(_) => "i64_is_zero",
            SintConcrete::WideMul(_) => "i64_wide_mul",
        },
        CoreConcreteLibfunc::Sint128(value) => match value {
            Sint128Concrete::Const(_) => "i128_const",
            Sint128Concrete::Equal(_) => "i128_eq",
            Sint128Concrete::ToFelt252(_) => "i128_to_felt252",
            Sint128Concrete::FromFelt252(_) => "i128_from_felt252",
            Sint128Concrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "i128_overflowing_add",
                IntOperator::OverflowingSub => "i128_overflowing_sub",
            },
            Sint128Concrete::Diff(_) => "i128_diff",
            Sint128Concrete::IsZero(_) => "i128_is_zero",
        },
        CoreConcreteLibfunc::Mem(value) => match value {
            MemConcreteLibfunc::StoreTemp(_) => "store_temp",
            MemConcreteLibfunc::StoreLocal(_) => "store_local",
            MemConcreteLibfunc::FinalizeLocals(_) => "finalize_locals",
            MemConcreteLibfunc::AllocLocal(_) => "alloc_local",
            MemConcreteLibfunc::Rename(_) => "rename",
        },
        CoreConcreteLibfunc::Nullable(value) => match value {
            NullableConcreteLibfunc::Null(_) => "nullable_null",
            NullableConcreteLibfunc::NullableFromBox(_) => "nullable_from_box",
            NullableConcreteLibfunc::MatchNullable(_) => "match_nullable",
            NullableConcreteLibfunc::ForwardSnapshot(_) => "nullable_forward_snapshot",
        },
        CoreConcreteLibfunc::UnwrapNonZero(_) => "unwrap_non_zero",
        CoreConcreteLibfunc::UnconditionalJump(_) => "jump",
        CoreConcreteLibfunc::Enum(value) => match value {
            EnumConcreteLibfunc::Init(_) => "enum_init",
            EnumConcreteLibfunc::FromBoundedInt(_) => "enum_from_bounded_int",
            EnumConcreteLibfunc::Match(_) => "enum_match",
            EnumConcreteLibfunc::SnapshotMatch(_) => "enum_snapshot_match",
        },
        CoreConcreteLibfunc::Struct(value) => match value {
            StructConcreteLibfunc::Construct(_) => "struct_construct",
            StructConcreteLibfunc::Deconstruct(_) => "struct_deconstruct",
            StructConcreteLibfunc::SnapshotDeconstruct(_) => "struct_snapshot_deconstruct",
        },
        CoreConcreteLibfunc::Felt252Dict(value) => match value {
            Felt252DictConcreteLibfunc::New(_) => "felt252dict_new",
            Felt252DictConcreteLibfunc::Squash(_) => "felt252dict_squash",
        },
        CoreConcreteLibfunc::Felt252DictEntry(value) => match value {
            Felt252DictEntryConcreteLibfunc::Get(_) => "felt252dict_get",
            Felt252DictEntryConcreteLibfunc::Finalize(_) => "felt252dict_finalize",
        },
        CoreConcreteLibfunc::Pedersen(value) => match value {
            PedersenConcreteLibfunc::PedersenHash(_) => "pedersen_hash",
        },
        CoreConcreteLibfunc::Poseidon(value) => match value {
            PoseidonConcreteLibfunc::HadesPermutation(_) => "hades_permutation",
        },
        CoreConcreteLibfunc::StarkNet(value) => match value {
            StarkNetConcreteLibfunc::CallContract(_) => "call_contract",
            StarkNetConcreteLibfunc::ClassHashConst(_) => "class_hash_const",
            StarkNetConcreteLibfunc::ClassHashTryFromFelt252(_) => "class_hash_try_from_felt252",
            StarkNetConcreteLibfunc::ClassHashToFelt252(_) => "class_hash_to_felt252",
            StarkNetConcreteLibfunc::ContractAddressConst(_) => "contract_address_const",
            StarkNetConcreteLibfunc::ContractAddressTryFromFelt252(_) => {
                "contract_address_try_from_felt252"
            }
            StarkNetConcreteLibfunc::ContractAddressToFelt252(_) => "contract_address_to_felt252",
            StarkNetConcreteLibfunc::StorageRead(_) => "storage_read",
            StarkNetConcreteLibfunc::StorageWrite(_) => "storage_write",
            StarkNetConcreteLibfunc::StorageBaseAddressConst(_) => "storage_base_address_const",
            StarkNetConcreteLibfunc::StorageBaseAddressFromFelt252(_) => {
                "storage_base_address_from_felt252"
            }
            StarkNetConcreteLibfunc::StorageAddressFromBase(_) => "storage_address_from_bas",
            StarkNetConcreteLibfunc::StorageAddressFromBaseAndOffset(_) => {
                "storage_address_from_Base_and_offset"
            }
            StarkNetConcreteLibfunc::StorageAddressToFelt252(_) => "storage_address_to_felt252",
            StarkNetConcreteLibfunc::StorageAddressTryFromFelt252(_) => {
                "storage_address_try_from_felt252"
            }
            StarkNetConcreteLibfunc::EmitEvent(_) => "emit_event",
            StarkNetConcreteLibfunc::GetBlockHash(_) => "get_block_hash",
            StarkNetConcreteLibfunc::GetExecutionInfo(_) => "get_exec_info_v1",
            StarkNetConcreteLibfunc::GetExecutionInfoV2(_) => "get_exec_info_v2",
            StarkNetConcreteLibfunc::Deploy(_) => "deploy",
            StarkNetConcreteLibfunc::Keccak(_) => "keccak",
            StarkNetConcreteLibfunc::LibraryCall(_) => "library_call",
            StarkNetConcreteLibfunc::ReplaceClass(_) => "replace_class",
            StarkNetConcreteLibfunc::SendMessageToL1(_) => "send_message_to_l1",
            StarkNetConcreteLibfunc::Testing(value) => match value {
                TestingConcreteLibfunc::Cheatcode(_) => "cheatcode",
            },
            StarkNetConcreteLibfunc::Secp256(value) => match value {
                Secp256ConcreteLibfunc::K1(value) => match value {
                    Secp256OpConcreteLibfunc::New(_) => "secp256k1_new",
                    Secp256OpConcreteLibfunc::Add(_) => "secp256k1_add",
                    Secp256OpConcreteLibfunc::Mul(_) => "secp256k1_mul",
                    Secp256OpConcreteLibfunc::GetPointFromX(_) => "secp256k1_get_point_from_x",
                    Secp256OpConcreteLibfunc::GetXy(_) => "secp256k1_get_xy",
                },
                Secp256ConcreteLibfunc::R1(value) => match value {
                    Secp256OpConcreteLibfunc::New(_) => "secp256r1_new",
                    Secp256OpConcreteLibfunc::Add(_) => "secp256r1_add",
                    Secp256OpConcreteLibfunc::Mul(_) => "secp256r1_mul",
                    Secp256OpConcreteLibfunc::GetPointFromX(_) => "secp256r1_get_point_from_x",
                    Secp256OpConcreteLibfunc::GetXy(_) => "secp256r1_get_xy",
                },
            },
        },
        CoreConcreteLibfunc::Debug(value) => match value {
            DebugConcreteLibfunc::Print(_) => "debug_print",
        },
        CoreConcreteLibfunc::SnapshotTake(_) => "snapshot_take",
        CoreConcreteLibfunc::Bytes31(value) => match value {
            Bytes31ConcreteLibfunc::Const(_) => "bytes31_const",
            Bytes31ConcreteLibfunc::ToFelt252(_) => "bytes31_to_felt252",
            Bytes31ConcreteLibfunc::TryFromFelt252(_) => "bytes31_try_from_felt252",
        },
    }
}
