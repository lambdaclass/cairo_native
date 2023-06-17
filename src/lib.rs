#![feature(hash_drain_filter)]
#![feature(iter_intersperse)]
#![feature(iterator_try_collect)]

use crate::types::TypeBuilderContext;
use cairo_lang_sierra::{
    edit_state,
    extensions::{ConcreteLibfunc, GenericLibfunc, GenericType},
    program::{Function, Program, Statement, StatementIdx},
    program_registry::ProgramRegistry,
};
use itertools::Itertools;
use libfuncs::{BranchArg, LibfuncBuilder, LibfuncBuilderContext};
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
    collections::{hash_map::Entry, BTreeMap, HashMap},
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
    let function_name = function
        .id
        .debug_name
        .as_deref()
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(format!("f{}", function.id.id)));

    let param_types = function
        .signature
        .param_types
        .iter()
        .map(|id| {
            registry
                .get_type(id)
                .map(|ty| {
                    ty.build(TypeBuilderContext::new(context, registry))
                        .unwrap()
                })
                .unwrap()
        })
        .collect::<Vec<_>>();
    let ret_types = function
        .signature
        .ret_types
        .iter()
        .map(|id| {
            registry
                .get_type(id)
                .map(|ty| {
                    ty.build(TypeBuilderContext::new(context, registry))
                        .unwrap()
                })
                .unwrap()
        })
        .collect::<Vec<_>>();

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

    let mut queue = vec![(function.entry_point, initial_state)];
    while let Some((statement_idx, state)) = queue.pop() {
        let block = Block::new(&[]);

        match &statements[statement_idx.0] {
            Statement::Invocation(invocation) => {
                let (state, _) = edit_state::take_args(state, invocation.args.iter()).unwrap();

                let branch_signatures = registry
                    .get_libfunc(&invocation.libfunc_id)
                    .unwrap()
                    .branch_signatures();
                for (branch, branch_signature) in invocation.branches.iter().zip(branch_signatures)
                {
                    let state = edit_state::put_results(
                        state.clone(),
                        branch.results.iter().zip(&branch_signature.vars).map(
                            |(var_id, var_info)| {
                                (
                                    var_id,
                                    registry
                                        .get_type(&var_info.ty)
                                        .unwrap()
                                        .build(TypeBuilderContext::new(context, registry))
                                        .unwrap(),
                                )
                            },
                        ),
                    )
                    .unwrap();

                    let target = statement_idx.next(&branch.target);
                    let entry = predecessors.entry(target);
                    let num_entries = match entry {
                        Entry::Occupied(x) => x.into_mut(),
                        Entry::Vacant(x) => {
                            queue.push((target, state.clone()));
                            x.insert((state.clone(), 0))
                        }
                    };

                    assert_eq!(num_entries.0, state);
                    num_entries.1 += 1;
                }

                for param_signature in registry
                    .get_libfunc(&invocation.libfunc_id)?
                    .param_signatures()
                {
                    block.add_argument(
                        registry
                            .get_type(&param_signature.ty)?
                            .build(TypeBuilderContext::new(context, registry))
                            .unwrap(),
                        Location::unknown(context),
                    );
                }
            }
            Statement::Return(_) => {
                for arg_ty in &ret_types {
                    block.add_argument(*arg_ty, Location::unknown(context));
                }
            }
        };

        blocks.insert(statement_idx.0, block);
    }

    let region = Region::new();
    let entry_block = region.append_block(Block::new(
        &param_types
            .iter()
            .copied()
            .map(|ty| (ty, Location::unknown(context)))
            .collect::<Vec<_>>(),
    ));

    let blocks = blocks
        .into_iter()
        .map(|(i, block)| {
            let statement_idx = StatementIdx(i);

            let libfunc_block = region.append_block(block);
            let landing_block = (predecessors[&statement_idx].1 > 1).then(|| {
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

    let mut queue = vec![(function.entry_point, initial_state)];
    while let Some((statement_idx, mut state)) = queue.pop() {
        let (landing_block, block) = &blocks[&statement_idx];

        if let Some(landing_block) = landing_block {
            state = edit_state::put_results::<Value>(
                HashMap::new(),
                state
                    .keys()
                    .sorted_by_key(|x| x.id)
                    .enumerate()
                    .map(|(idx, var_id)| (var_id, landing_block.argument(idx).unwrap().into())),
            )
            .unwrap();
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
                                Some(_) => todo!(),
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

                for (branch, result_vars) in invocation.branches.iter().zip(&*result_variables) {
                    let next_statement_idx = statement_idx.next(&branch.target);

                    let state = edit_state::put_results(
                        state.clone(),
                        branch.results.iter().zip(
                            result_vars
                                .iter()
                                .cloned()
                                .map(Cell::into_inner)
                                .map(Option::unwrap),
                        ),
                    )
                    .unwrap();

                    queue.push((next_statement_idx, state));
                }
            }
            Statement::Return(return_vars) => {
                let (state, return_vars) =
                    edit_state::take_args(state, return_vars.iter()).unwrap();
                assert!(state.is_empty());

                block.append_operation(func::r#return(&return_vars, Location::unknown(context)));
            }
        }
    }

    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, &function_name),
        TypeAttribute::new(FunctionType::new(context, &param_types, &ret_types).into()),
        region,
        &[
            // (
            //     Identifier::new(context, "sym_name"),
            //     StringAttribute::new(context, &function_name).into(),
            // ),
            (
                Identifier::new(context, "sym_visibility"),
                StringAttribute::new(context, "public").into(),
            ),
        ],
        Location::unknown(context),
    ));

    Ok(())
}
