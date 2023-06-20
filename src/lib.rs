#![feature(box_into_inner)]
#![feature(iter_intersperse)]
#![feature(iterator_try_collect)]
#![feature(map_try_insert)]
#![feature(trait_upcasting)]

pub use self::debug_info::DebugInfo;
use self::libfuncs::{BranchArg, LibfuncHelper};
use crate::metadata::tail_recursion::TailRecursionMeta;
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
    dialect::{cf, func},
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        r#type::FunctionType,
        Block, BlockRef, Identifier, Location, Module, Region, Type, Value,
    },
    Context,
};
use metadata::MetadataStorage;
use std::{
    borrow::Cow,
    cell::Cell,
    collections::{hash_map::Entry, BTreeMap, HashMap, HashSet},
    ops::Deref,
};
use types::TypeBuilder;

mod debug_info;
pub(crate) mod ffi;
pub mod libfuncs;
pub mod metadata;
pub mod types;

type BlockStorage<'c, 'a> =
    HashMap<StatementIdx, (Option<(BlockRef<'c, 'a>, Vec<VarId>)>, BlockRef<'c, 'a>)>;

pub fn compile<'c, TType, TLibfunc>(
    context: &'c Context,
    program: &Program,
) -> Result<Module<'c>, Box<dyn std::error::Error>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let module = Module::new(Location::unknown(context));

    let mut metadata_storage = MetadataStorage::new();

    let program_registry = ProgramRegistry::<TType, TLibfunc>::new(program)?;
    for function in &program.funcs {
        tracing::info!("Compiling function `{}`.", function.id);
        compile_func::<TType, TLibfunc>(
            context,
            &module,
            &program_registry,
            function,
            &program.statements,
            &mut metadata_storage,
        )?;
    }

    Ok(module)
}

fn compile_func<TType, TLibfunc>(
    context: &Context,
    module: &Module,
    registry: &ProgramRegistry<TType, TLibfunc>,
    function: &Function,
    statements: &[Statement],
    metadata_storage: &mut MetadataStorage,
) -> Result<(), Box<dyn std::error::Error>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let region = Region::new();

    tracing::debug!("Generating function structure (region with blocks).");
    let (entry_block, blocks) =
        generate_function_structure(context, &region, registry, function, statements).unwrap();

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

    foreach_statement_in_function(
        statements,
        function.entry_point,
        initial_state,
        |statement_idx, mut state| {
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
                        _module: module,
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
                        metadata_storage
                            .insert(TailRecursionMeta::new(target == &function.id))
                            .unwrap();
                    }

                    concrete_libfunc
                        .build(
                            context,
                            registry,
                            block,
                            Location::unknown(context),
                            &helper,
                        )
                        .unwrap();
                    assert!(block.terminator().is_some());

                    metadata_storage.remove::<TailRecursionMeta>();

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
                            edit_state::put_results(
                                state.clone(),
                                branch_info
                                    .results
                                    .iter()
                                    .zip(result_values.iter().copied()),
                            )
                            .unwrap()
                        })
                        .collect()
                }
                Statement::Return(var_ids) => {
                    tracing::trace!("Implementing the return statement at {statement_idx}");

                    let (_, values) = edit_state::take_args(state, var_ids.iter()).unwrap();
                    block.append_operation(func::r#return(&values, Location::unknown(context)));

                    Vec::new()
                }
            }
        },
    );

    let function_name = generate_function_name(function);
    tracing::debug!("Creating the actual function, named `{function_name}`.");
    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, &function_name),
        TypeAttribute::new(
            FunctionType::new(
                context,
                &extract_types(context, &function.signature.param_types, registry)
                    .collect::<Vec<_>>(),
                &extract_types(context, &function.signature.ret_types, registry)
                    .collect::<Vec<_>>(),
            )
            .into(),
        ),
        region,
        &[(
            Identifier::new(context, "sym_visibility"),
            StringAttribute::new(context, "public").into(),
        )],
        Location::unknown(context),
    ));

    tracing::debug!("Done generating function {}.", function.id);
    Ok(())
}

fn generate_function_structure<'c, 'a, TType, TLibfunc>(
    context: &'c Context,
    region: &'a Region<'c>,
    registry: &ProgramRegistry<TType, TLibfunc>,
    function: &Function,
    statements: &[Statement],
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
                        .build(context, registry)
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
                                            .build(context, registry)
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
        &extract_types(context, &function.signature.param_types, registry)
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

fn generate_function_name(function: &Function) -> Cow<str> {
    function
        .id
        .debug_name
        .as_deref()
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(format!("f{}", function.id.id)))
}

fn extract_types<'c, 'a, TType, TLibfunc>(
    context: &'c Context,
    type_ids: &'a [ConcreteTypeId],
    registry: &'a ProgramRegistry<TType, TLibfunc>,
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
            .map(|ty| ty.build(context, registry).unwrap())
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
