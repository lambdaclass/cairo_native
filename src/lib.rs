#![feature(iter_intersperse)]
#![feature(iterator_try_collect)]
#![feature(map_try_insert)]

use crate::{
    libfuncs::{BranchArg, LibfuncBuilderContext},
    types::TypeBuilderContext,
};
use cairo_lang_sierra::{
    edit_state,
    extensions::{ConcreteLibfunc, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
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
        Block, Identifier, Location, Module, Region, Type, Value,
    },
    Context,
};
use std::{
    borrow::Cow,
    cell::Cell,
    collections::{hash_map::Entry, BTreeMap, HashMap, HashSet},
    ops::Deref,
    rc::Rc,
};
use types::TypeBuilder;

// mod debug_info;
pub(crate) mod ffi;
pub mod libfuncs;
pub mod types;

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

    let program_registry = ProgramRegistry::<TType, TLibfunc>::new(program)?;
    for function in &program.funcs {
        compile_func::<TType, TLibfunc>(
            context,
            &module,
            &program_registry,
            function,
            &program.statements,
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
) -> Result<(), Box<dyn std::error::Error>>
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
                        .build(TypeBuilderContext::new(context, registry))
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
                                            .build(TypeBuilderContext::new(context, registry))
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

    let region = Region::new();
    let entry_block = region.append_block(Block::new(
        &extract_types(context, &function.signature.param_types, registry)
            .map(|ty| (ty, Location::unknown(context)))
            .collect::<Vec<_>>(),
    ));

    let blocks = blocks
        .into_iter()
        .map(|(i, block)| {
            let statement_idx = StatementIdx(i);

            let libfunc_block = region.append_block(block);
            let landing_block = (predecessors[&statement_idx].1 > 1).then(|| {
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

    let initial_state = edit_state::put_results(
        HashMap::<_, Value>::new(),
        function
            .params
            .iter()
            .enumerate()
            .map(|(idx, param)| (&param.id, entry_block.argument(idx).unwrap().into())),
    )
    .unwrap();

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
                    let (state, _) = edit_state::take_args(state, invocation.args.iter()).unwrap();

                    let result_variables = Rc::new(
                        invocation
                            .branches
                            .iter()
                            .map(|x| vec![Cell::new(None); x.results.len()])
                            .collect::<Vec<_>>(),
                    );
                    let build_ctx = LibfuncBuilderContext::new(
                        context,
                        registry,
                        module,
                        block,
                        Location::unknown(context),
                        invocation
                            .branches
                            .iter()
                            .map(|branch| {
                                let target_idx = statement_idx.next(&branch.target);
                                let (landing_block, block) = &blocks[&target_idx];

                                match landing_block {
                                    Some((landing_block, state_vars)) => {
                                        let target_vars = state_vars
                                            .iter()
                                            .map(|(var_id, _)| {
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
                        Rc::clone(&result_variables),
                    );

                    registry
                        .get_libfunc(&invocation.libfunc_id)
                        .unwrap()
                        .build(build_ctx)
                        .unwrap();
                    assert!(block.terminator().is_some());

                    invocation
                        .branches
                        .iter()
                        .zip(result_variables.iter())
                        .map(|(branch_info, result_values)| {
                            assert_eq!(
                                branch_info.results.len(),
                                result_values.len(),
                                "Mismatched number of returned values from branch."
                            );
                            let state = edit_state::put_results(
                                state.clone(),
                                branch_info
                                    .results
                                    .iter()
                                    .zip(result_values.iter().map(|x| x.get().unwrap())),
                            )
                            .unwrap();

                            predecessors[&statement_idx.next(&branch_info.target)]
                                .0
                                .keys()
                                .map(|var_id| (var_id.clone(), state[var_id]))
                                .collect()
                        })
                        .collect()
                }
                Statement::Return(var_ids) => {
                    let (_, values) = edit_state::take_args(state, var_ids.iter()).unwrap();
                    block.append_operation(func::r#return(&values, Location::unknown(context)));

                    Vec::new()
                }
            }
        },
    );

    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, &generate_function_name(function)),
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

    Ok(())
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
            .map(|ty| {
                ty.build(TypeBuilderContext::new(context, registry))
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
