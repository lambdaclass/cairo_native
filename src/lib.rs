#![feature(box_into_inner)]
#![feature(int_roundings)]
#![feature(iter_intersperse)]
#![feature(iterator_try_collect)]
#![feature(map_try_insert)]
#![feature(pointer_byte_offsets)]

pub use self::debug_info::DebugInfo;
use self::libfuncs::{BranchArg, LibfuncHelper};
use crate::{metadata::tail_recursion::TailRecursionMeta, utils::generate_function_name};
use cairo_lang_sierra::{
    edit_state,
    extensions::{ConcreteLibfunc, GenericLibfunc, GenericType},
    ids::{ConcreteTypeId, VarId},
    program::{Function, Program, Statement, StatementIdx},
    program_registry::ProgramRegistry,
};
use itertools::Itertools;
use libfuncs::LibfuncBuilder;
use melior::{
    dialect::{arith::CmpiPredicate, cf, func, index, memref},
    ir::{
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, MemRefType},
        Attribute, Block, BlockRef, Identifier, Location, Module, Region, Type, Value,
    },
    Context,
};
use metadata::MetadataStorage;
use std::{
    cell::Cell,
    collections::{hash_map::Entry, BTreeMap, HashMap, HashSet},
    ops::Deref,
};
use typed_arena::Arena;
use types::TypeBuilder;

mod debug_info;
pub mod ffi;
pub mod libfuncs;
pub mod metadata;
pub mod types;
pub mod utils;
pub mod values;

type BlockStorage<'c, 'a> =
    HashMap<StatementIdx, (Option<(BlockRef<'c, 'a>, Vec<VarId>)>, BlockRef<'c, 'a>)>;

pub fn compile<'c, TType, TLibfunc>(
    context: &'c Context,
    module: &Module<'c>,
    program: &Program,
    registry: &ProgramRegistry<TType, TLibfunc>,
    metadata: &mut MetadataStorage,
) -> Result<(), Box<dyn std::error::Error>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    for function in &program.funcs {
        tracing::info!("Compiling function `{}`.", function.id);
        compile_func::<TType, TLibfunc>(
            context,
            module,
            registry,
            function,
            &program.statements,
            metadata,
        )?;
    }

    tracing::info!("The program was compiled successfully.");
    Ok(())
}

fn compile_func<TType, TLibfunc>(
    context: &Context,
    module: &Module,
    registry: &ProgramRegistry<TType, TLibfunc>,
    function: &Function,
    statements: &[Statement],
    metadata: &mut MetadataStorage,
) -> Result<(), Box<dyn std::error::Error>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let region = Region::new();
    let blocks_arena = Arena::new();

    tracing::debug!("Generating function structure (region with blocks).");
    let (entry_block, blocks) = generate_function_structure(
        context, module, &region, registry, function, statements, metadata,
    )
    .unwrap();

    tracing::debug!("Generating the function implementation.");
    let initial_state = edit_state::put_results(
        HashMap::<_, Value>::new(),
        function
            .params
            .iter()
            .enumerate()
            .map(|(idx, param)| (&param.id, entry_block.argument(idx).unwrap().into())),
    )
    .unwrap();

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
    foreach_statement_in_function(
        statements,
        function.entry_point,
        (initial_state, BTreeMap::<usize, usize>::new()),
        |statement_idx, (mut state, mut tailrec_state)| {
            let (landing_block, block) = &blocks[&statement_idx];

            if let Some((landing_block, _)) = landing_block {
                tracing::trace!("Implementing the statement {statement_idx}'s landing block.");

                state = edit_state::put_results(
                    HashMap::default(),
                    state
                        .keys()
                        .sorted_by_key(|x| x.id)
                        .enumerate()
                        .map(|(idx, var_id)| (var_id, landing_block.argument(idx).unwrap().into())),
                )
                .unwrap();

                landing_block.append_operation(cf::br(
                    block,
                    &edit_state::take_args(
                        state.clone(),
                        match &statements[statement_idx.0] {
                            Statement::Invocation(x) => &x.args,
                            Statement::Return(x) => x,
                        }
                        .iter(),
                    )
                    .unwrap()
                    .1,
                    Location::unknown(context),
                ));
            }

            match &statements[statement_idx.0] {
                Statement::Invocation(invocation) => {
                    tracing::trace!(
                        "Implementing the invocation statement at {statement_idx}: {}.",
                        invocation.libfunc_id
                    );

                    let (state, _) = edit_state::take_args(state, invocation.args.iter()).unwrap();

                    let helper = LibfuncHelper {
                        module,
                        region: &region,
                        blocks_arena: &blocks_arena,
                        last_block: Cell::new(block),
                        branches: invocation
                            .branches
                            .iter()
                            .map(|branch| {
                                let target_idx = statement_idx.next(&branch.target);
                                let (landing_block, block) = &blocks[&target_idx];

                                match landing_block {
                                    Some((landing_block, state_vars)) => {
                                        let target_vars = state_vars
                                            .iter()
                                            .map(|var_id| {
                                                match branch
                                                    .results
                                                    .iter()
                                                    .find_position(|id| *id == var_id)
                                                {
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
                            .collect(),
                        results: invocation
                            .branches
                            .iter()
                            .map(|x| vec![Cell::new(None); x.results.len()])
                            .collect::<Vec<_>>(),
                    };

                    let concrete_libfunc = registry.get_libfunc(&invocation.libfunc_id).unwrap();
                    if let Some(target) = concrete_libfunc.is_function_call() {
                        if target == &function.id && state.is_empty() {
                            let op0 = entry_block.insert_operation(
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
                            let op1 = entry_block.insert_operation_after(
                                op0,
                                index::constant(
                                    context,
                                    IntegerAttribute::new(0, Type::index(context)),
                                    Location::unknown(context),
                                ),
                            );
                            entry_block.insert_operation_after(
                                op1,
                                memref::store(
                                    op1.result(0).unwrap().into(),
                                    op0.result(0).unwrap().into(),
                                    &[],
                                    Location::unknown(context),
                                ),
                            );

                            metadata
                                .insert(TailRecursionMeta::new(
                                    op0.result(0).unwrap().into(),
                                    &entry_block,
                                ))
                                .unwrap();
                        }
                    }

                    concrete_libfunc
                        .build(
                            context,
                            registry,
                            block,
                            Location::unknown(context),
                            &helper,
                            metadata,
                        )
                        .unwrap();
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

                            (
                                edit_state::put_results(
                                    state.clone(),
                                    branch_info
                                        .results
                                        .iter()
                                        .zip(result_values.iter().copied()),
                                )
                                .unwrap(),
                                tailrec_state.clone(),
                            )
                        })
                        .collect()
                }
                Statement::Return(var_ids) => {
                    tracing::trace!("Implementing the return statement at {statement_idx}");

                    let (_, values) = edit_state::take_args(state, var_ids.iter()).unwrap();

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
                                op0.result(0).unwrap().into(),
                                op1.result(0).unwrap().into(),
                                Location::unknown(context),
                            ));

                            let op3 = block.append_operation(index::constant(
                                context,
                                IntegerAttribute::new(1, Type::index(context)),
                                Location::unknown(context),
                            ));
                            let op4 = block.append_operation(index::sub(
                                op0.result(0).unwrap().into(),
                                op3.result(0).unwrap().into(),
                                Location::unknown(context),
                            ));
                            block.append_operation(memref::store(
                                op4.result(0).unwrap().into(),
                                depth_counter,
                                &[],
                                Location::unknown(context),
                            ));

                            block.append_operation(cf::cond_br(
                                context,
                                op2.result(0).unwrap().into(),
                                &cont_block,
                                &return_target,
                                &[],
                                &values,
                                Location::unknown(context),
                            ));

                            block = cont_block;
                        }
                    }

                    block.append_operation(func::r#return(&values, Location::unknown(context)));

                    Vec::new()
                }
            }
        },
    );

    // Workaround for the `entry block of region may not have predecessors` error:
    if !tailrec_storage.is_empty() {
        let new_entry_block = region.insert_block_before(
            entry_block,
            Block::new(
                &extract_types(
                    context,
                    module,
                    &function.signature.param_types,
                    registry,
                    metadata,
                )
                .map(|ty| (ty, Location::unknown(context)))
                .collect::<Vec<_>>(),
            ),
        );
        new_entry_block.append_operation(cf::br(
            &entry_block,
            &(0..entry_block.argument_count())
                .map(|i| new_entry_block.argument(i).unwrap().into())
                .collect::<Vec<_>>(),
            Location::unknown(context),
        ));
    }

    let function_name = generate_function_name(&function.id);
    tracing::debug!("Creating the actual function, named `{function_name}`.");
    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, &function_name),
        TypeAttribute::new(
            FunctionType::new(
                context,
                &extract_types(
                    context,
                    module,
                    &function.signature.param_types,
                    registry,
                    metadata,
                )
                .collect::<Vec<_>>(),
                &extract_types(
                    context,
                    module,
                    &function.signature.ret_types,
                    registry,
                    metadata,
                )
                .collect::<Vec<_>>(),
            )
            .into(),
        ),
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

fn generate_function_structure<'c, 'a, TType, TLibfunc>(
    context: &'c Context,
    module: &'a Module<'c>,
    region: &'a Region<'c>,
    registry: &ProgramRegistry<TType, TLibfunc>,
    function: &Function,
    statements: &[Statement],
    metadata_storage: &mut MetadataStorage,
) -> Result<(BlockRef<'c, 'a>, BlockStorage<'c, 'a>), Box<dyn std::error::Error>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let initial_state = edit_state::put_results::<Type>(
        HashMap::new(),
        function
            .params
            .iter()
            .zip(&function.signature.param_types)
            .map(|(param, ty)| {
                (
                    &param.id,
                    registry
                        .get_type(ty)
                        .unwrap()
                        .build(context, module, registry, metadata_storage)
                        .unwrap(),
                )
            }),
    )
    .unwrap();

    let mut blocks = BTreeMap::new();
    let mut predecessors = HashMap::from([(function.entry_point, (initial_state.clone(), 0))]);

    foreach_statement_in_function(
        statements,
        function.entry_point,
        initial_state,
        |statement_idx, state| {
            let block = blocks.try_insert(statement_idx.0, Block::new(&[])).unwrap();

            match &statements[statement_idx.0] {
                Statement::Invocation(invocation) => {
                    tracing::trace!(
                        "Creating block for invocation statement at index {statement_idx}: {}",
                        invocation.libfunc_id
                    );

                    let (state, types) =
                        edit_state::take_args(state.clone(), invocation.args.iter()).unwrap();

                    types.into_iter().for_each(|ty| {
                        block.add_argument(ty, Location::unknown(context));
                    });

                    let libfunc = registry.get_libfunc(&invocation.libfunc_id).unwrap();
                    invocation
                        .branches
                        .iter()
                        .zip(libfunc.branch_signatures())
                        .map(|(branch, branch_signature)| {
                            let state = edit_state::put_results(
                                state.clone(),
                                branch.results.iter().zip(branch_signature.vars.iter().map(
                                    |var_info| {
                                        registry
                                            .get_type(&var_info.ty)
                                            .unwrap()
                                            .build(context, module, registry, metadata_storage)
                                            .unwrap()
                                    },
                                )),
                            )
                            .unwrap();

                            let (prev_state, pred_count) =
                                match predecessors.entry(statement_idx.next(&branch.target)) {
                                    Entry::Occupied(entry) => entry.into_mut(),
                                    Entry::Vacant(entry) => entry.insert((state.clone(), 0)),
                                };
                            assert_eq!(prev_state, &state, "Branch target states do not match.");
                            *pred_count += 1;

                            state
                        })
                        .collect()
                }
                Statement::Return(var_ids) => {
                    tracing::trace!(
                        "Creating block for return statement at index {statement_idx}."
                    );

                    let (state, types) =
                        edit_state::take_args(state.clone(), var_ids.iter()).unwrap();
                    assert!(
                        state.is_empty(),
                        "State must be empty after a return statement."
                    );

                    types.into_iter().for_each(|ty| {
                        block.add_argument(ty, Location::unknown(context));
                    });

                    Vec::new()
                }
            }
        },
    );

    tracing::trace!("Generating function entry block.");
    let entry_block = region.append_block(Block::new(
        &extract_types(
            context,
            module,
            &function.signature.param_types,
            registry,
            metadata_storage,
        )
        .map(|ty| (ty, Location::unknown(context)))
        .collect::<Vec<_>>(),
    ));

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
                                .map(|ty| (ty, Location::unknown(context)))
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

fn extract_types<'c, 'a, TType, TLibfunc>(
    context: &'c Context,
    module: &'a Module<'c>,
    type_ids: &'a [ConcreteTypeId],
    registry: &'a ProgramRegistry<TType, TLibfunc>,
    metadata_storage: &'a mut MetadataStorage,
) -> impl 'a + Iterator<Item = Type<'c>>
where
    'c: 'a,
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    type_ids.iter().map(|id| {
        registry
            .get_type(id)
            .map(|ty| {
                ty.build(context, module, registry, metadata_storage)
                    .unwrap()
            })
            .unwrap()
    })
}

fn foreach_statement_in_function<S>(
    statements: &[Statement],
    entry_point: StatementIdx,
    initial_state: S,
    mut closure: impl FnMut(StatementIdx, S) -> Vec<S>,
) where
    S: Clone,
{
    let mut queue = vec![(entry_point, initial_state)];
    let mut visited = HashSet::new();

    while let Some((statement_idx, state)) = queue.pop() {
        if !visited.insert(statement_idx) {
            continue;
        }

        let branch_states = closure(statement_idx, state);

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
}
