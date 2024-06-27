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
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        ConcreteLibfunc,
    },
    ids::{ConcreteTypeId, FunctionId, VarId},
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
        Attribute, Block, BlockRef, Identifier, Location, Module, Region, Type, Value,
    },
    Context,
};
use std::{
    cell::{Cell, RefCell},
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

/// The compiler struct is used to hold the MLIR context, the output module, the program registry,
/// the metadata and the debug information. The only field that needs mutability is the metadata,
/// which is stored in a RefCell in order to allow for interior mutability.
///
/// The generics `TType` and `TLibfunc` contain the information required to generate the MLIR types
/// and statement operations. Most of the time you'll want to use the default ones, which are
/// [CoreType](cairo_lang_sierra::extensions::core::CoreType) and
/// [CoreLibfunc](cairo_lang_sierra::extensions::core::CoreLibfunc) respectively.
///
/// The compiler uses a reference to the metadata which is passed externally so that stuff can be
/// initialized if necessary.
///
/// The compiler needs the program and the program's registry, which doesn't need to have AP
/// tracking information.
pub struct Compiler<'c> {
    context: &'c Context,
    module: &'c Module<'c>,
    program: &'c Program,
    registry: &'c ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: RefCell<&'c mut MetadataStorage>,
    debug_info: Option<&'c DebugLocations<'c>>,
}

impl<'c> Compiler<'c> {
    pub fn new(
        context: &'c Context,
        module: &'c Module<'c>,
        program: &'c Program,
        registry: &'c ProgramRegistry<CoreType, CoreLibfunc>,
        metadata: &'c mut MetadataStorage,
        debug_info: Option<&'c DebugLocations<'c>>,
    ) -> Self {
        Self {
            context,
            module,
            program,
            registry,
            metadata: RefCell::new(metadata),
            debug_info,
        }
    }

    /// Run the compiler on a program. The compiled program is stored in the MLIR module.
    pub fn compile(&'c self) -> Result<(), Error> {
        for function in &self.program.funcs {
            tracing::info!("Compiling function `{}`.", function.id);
            self.compile_func(function, &self.program.statements)?;
        }

        tracing::info!("The program was compiled successfully.");
        Ok(())
    }

    /// Compile a single Sierra function.
    ///
    /// The function accepts a `Function` argument, which provides the function's entry point, signature
    /// and name. Check out [compile](Compiler::compile) for a description of the other arguments.
    ///
    /// The [module docs](self) contain more information about the compilation process.
    fn compile_func(
        &'c self,
        function: &'c Function,
        statements: &'c [Statement],
    ) -> Result<(), Error> {
        let region = Region::new();
        let blocks_arena = Bump::new();

        let mut arg_types = self
            .extract_types(&function.signature.param_types)
            .collect::<Result<Vec<_>, _>>()?;
        let mut return_types = self
            .extract_types(&function.signature.ret_types)
            .collect::<Result<Vec<_>, _>>()?;

        // Replace memory-allocated arguments with pointers.
        for (ty, type_info) in
            arg_types
                .iter_mut()
                .zip(function.signature.param_types.iter().filter_map(|type_id| {
                    let type_info = self.get_type(type_id).unwrap();
                    if self.type_is_zst_builtin(type_id) {
                        None
                    } else {
                        Some(type_info)
                    }
                }))
        {
            if type_info.is_memory_allocated(self.registry) {
                *ty = llvm::r#type::pointer(self.context, 0);
            }
        }

        // Extract memory-allocated return types from return_types and insert them in arg_types as a
        // pointer.
        let return_type_infos = function
            .signature
            .ret_types
            .iter()
            .filter_map(|type_id| {
                let type_info = self.get_type(type_id).unwrap();
                if self.type_is_zst_builtin(type_id) {
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
            .is_some_and(|(_, type_info)| type_info.is_memory_allocated(self.registry))
        {
            assert_eq!(return_types.len(), 1);

            return_types.remove(0);
            arg_types.insert(0, llvm::r#type::pointer(self.context, 0));

            Some(true)
        } else {
            None
        };

        tracing::debug!("Generating function structure (region with blocks).");
        let (entry_block, blocks) =
            self.generate_function_structure(&region, function, statements)?;

        tracing::debug!("Generating the function implementation.");
        // Workaround for the `entry block of region may not have predecessors` error:
        let pre_entry_block = region.insert_block_before(
            entry_block,
            Block::new(
                &arg_types
                    .iter()
                    .map(|ty| (*ty, Location::unknown(self.context)))
                    .collect::<Vec<_>>(),
            ),
        );

        let initial_state = edit_state::put_results(HashMap::<_, Value>::new(), {
            let mut values = Vec::new();

            let mut count = 0;
            for param in &function.params {
                values.push((
                    &param.id,
                    if self.type_is_zst_builtin(&param.ty) {
                        pre_entry_block
                            .append_operation(llvm::undef(
                                self.build_type(&param.ty)?,
                                Location::unknown(self.context),
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
            Location::unknown(self.context),
        ));

        let mut tailrec_storage = Vec::<(Value, BlockRef)>::new();
        foreach_statement_in_function::<_, Error>(
            statements,
            function.entry_point,
            (initial_state, BTreeMap::<usize, usize>::new()),
            |statement_idx, (mut state, mut tailrec_state)| {
                let has_gas_metadata = self.metadata.borrow().get::<GasMetadata>().is_some();
                if has_gas_metadata {
                    let gas_cost = self
                        .metadata
                        .borrow()
                        .get::<GasMetadata>()
                        .expect("has gas metadata")
                        .get_gas_cost_for_statement(statement_idx);
                    self.metadata.borrow_mut().remove::<GasCost>();
                    self.metadata.borrow_mut().insert(GasCost(gas_cost));
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
                            self.context,
                            &format!("landing_block(stmt_idx={})", statement_idx),
                            Location::unknown(self.context),
                        ),
                    ));
                }

                Ok(match &statements[statement_idx.0] {
                    Statement::Invocation(invocation) => {
                        tracing::trace!(
                            "Implementing the invocation statement at {statement_idx}: {}.",
                            invocation.libfunc_id
                        );
                        let libfunc_name =
                            format!("{}(stmt_idx={})", invocation.libfunc_id, statement_idx);

                        let (state, _) = edit_state::take_args(state, invocation.args.iter())?;

                        let helper = LibfuncHelper {
                            module: self.module,
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

                        let concrete_libfunc = self.registry.get_libfunc(&invocation.libfunc_id)?;
                        if let Some(target) = concrete_libfunc.is_function_call() {
                            self.generate_function_call(
                                function,
                                target,
                                &libfunc_name,
                                &pre_entry_block,
                                &entry_block,
                                &state,
                            )?;
                        }

                        concrete_libfunc.build(
                            self.context,
                            self.registry,
                            block,
                            Location::name(
                                self.context,
                                &libfunc_name,
                                self.debug_info
                                    .and_then(|debug_info| {
                                        debug_info.statements.get(&statement_idx).copied()
                                    })
                                    .unwrap_or_else(|| Location::unknown(self.context)),
                            ),
                            &helper,
                            *self.metadata.borrow_mut(),
                        )?;
                        assert!(block.terminator().is_some());

                        if let Some(tailrec_meta) =
                            self.metadata.borrow_mut().remove::<TailRecursionMeta>()
                        {
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
                            self.context,
                            &format!("return(stmt_idx={})", statement_idx),
                            Location::unknown(self.context),
                        );

                        let (_, mut values) = edit_state::take_args(state, var_ids.iter())?;

                        let mut block = *block;
                        if !tailrec_state.is_empty() {
                            let location = Location::name(
                                self.context,
                                &format!("return(stmt_idx={}, tail_recursion)", statement_idx),
                                Location::unknown(self.context),
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
                                        self.context,
                                        IntegerAttribute::new(Type::index(self.context), 0),
                                        location,
                                    ))
                                    .result(0)?
                                    .into();
                                let is_zero_depth = block
                                    .append_operation(index::cmp(
                                        self.context,
                                        CmpiPredicate::Eq,
                                        depth_counter_value,
                                        k0,
                                        location,
                                    ))
                                    .result(0)?
                                    .into();

                                let k1 = block
                                    .append_operation(index::constant(
                                        self.context,
                                        IntegerAttribute::new(Type::index(self.context), 1),
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
                                            let type_info = self.get_type(type_id).unwrap();
                                            if type_info.is_zst(self.registry)
                                                || type_info.is_memory_allocated(self.registry)
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
                                            let type_info = self.get_type(type_id).unwrap();
                                            if type_info.is_zst(self.registry) {
                                                None
                                            } else {
                                                Some(*value)
                                            }
                                        })
                                        .collect::<Vec<_>>(),
                                    None => todo!(),
                                };

                                block.append_operation(cf::cond_br(
                                    self.context,
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
                        for (idx, type_id) in function.signature.ret_types.iter().enumerate().rev()
                        {
                            if self.type_is_zst_builtin(type_id) {
                                values.remove(idx);
                            }
                        }

                        // Store the return value in the return pointer, if there's one.
                        if let Some(true) = has_return_ptr {
                            let (_ret_type_id, ret_type_info) = return_type_infos[0];
                            let ret_layout = ret_type_info.layout(self.registry)?;

                            let ptr = values.remove(0);
                            block.append_operation(llvm::store(
                                self.context,
                                ptr,
                                pre_entry_block.argument(0)?.into(),
                                location,
                                LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                                    IntegerType::new(self.context, 64).into(),
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
        let mut arg_values = Vec::with_capacity(function.signature.param_types.len());
        for (i, type_id_and_info) in function
            .signature
            .param_types
            .iter()
            .filter_map(|type_id| {
                self.get_type(type_id)
                    .map(|type_info| {
                        if self.type_is_zst_builtin(type_id) {
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
            if type_info.is_memory_allocated(self.registry) {
                value = pre_entry_block
                    .append_operation(llvm::load(
                        self.context,
                        value,
                        self.build_type(type_id)?,
                        Location::unknown(self.context),
                        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                            IntegerType::new(self.context, 64).into(),
                            type_info.layout(self.registry)?.align() as i64,
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
            Location::unknown(self.context),
        ));

        let function_name = generate_function_name(&function.id);
        tracing::debug!("Creating the actual function, named `{function_name}`.");

        self.module.body().append_operation(func::func(
            self.context,
            StringAttribute::new(self.context, &function_name),
            TypeAttribute::new(FunctionType::new(self.context, &arg_types, &return_types).into()),
            region,
            &[
                (
                    Identifier::new(self.context, "sym_visibility"),
                    StringAttribute::new(self.context, "public").into(),
                ),
                (
                    Identifier::new(self.context, "llvm.emit_c_interface"),
                    Attribute::unit(self.context),
                ),
            ],
            Location::unknown(self.context),
        ));

        tracing::debug!("Done generating function {}.", function.id);
        Ok(())
    }

    fn generate_function_structure<'a>(
        &'c self,
        region: &'a Region<'c>,
        function: &'c Function,
        statements: &'c [Statement],
    ) -> Result<(BlockRef<'c, 'a>, BlockStorage<'c, 'a>), Error> {
        let initial_state = edit_state::put_results::<Type>(
            HashMap::new(),
            function
                .params
                .iter()
                .zip(&function.signature.param_types)
                .map(|(param, ty)| Ok((&param.id, self.build_type(ty)?)))
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
                    if let std::collections::btree_map::Entry::Vacant(e) =
                        blocks.entry(statement_idx.0)
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
                            block.add_argument(ty, Location::unknown(self.context));
                        }

                        let libfunc = self.registry.get_libfunc(&invocation.libfunc_id)?;
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
                                                self.build_type(&var_info.ty)
                                            })
                                            .collect::<Result<Vec<_>, _>>()?,
                                    ),
                                )?;

                                let (prev_state, pred_count) =
                                    match predecessors.entry(statement_idx.next(&branch.target)) {
                                        Entry::Occupied(entry) => entry.into_mut(),
                                        Entry::Vacant(entry) => entry.insert((state.clone(), 0)),
                                    };
                                assert_eq!(
                                    prev_state, &state,
                                    "Branch target states do not match."
                                );
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
                            block.add_argument(ty, Location::unknown(self.context));
                        }

                        Vec::new()
                    }
                })
            },
        )?;

        tracing::trace!("Generating function entry block.");
        let entry_block = region.append_block(Block::new(&{
            self.extract_types(&function.signature.param_types)
                .map(|ty| Ok((ty?, Location::unknown(self.context))))
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
                                    .map(|ty| (ty, Location::unknown(self.context)))
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

    /// Generates the MLIR operations in the case where the libfunc is a function call.
    fn generate_function_call(
        &self,
        function: &Function,
        target: &FunctionId,
        libfunc_name: &str,
        pre_entry_block: &Block,
        entry_block: &Block,
        state: &HashMap<VarId, Value>,
    ) -> Result<(), Error> {
        if target == &function.id && state.is_empty() {
            // TODO: Defer insertions until after the recursion has been confirmed
            //   (when removing the meta, if a return target is set).
            // TODO: Explore replacing the `memref` counter with a normal variable.
            let location = Location::name(
                self.context,
                &format!("recursion_counter({})", libfunc_name),
                Location::unknown(self.context),
            );
            let op0 = pre_entry_block.insert_operation(
                0,
                memref::alloca(
                    self.context,
                    MemRefType::new(Type::index(self.context), &[], None, None),
                    &[],
                    &[],
                    None,
                    location,
                ),
            );
            let op1 = pre_entry_block.insert_operation_after(
                op0,
                index::constant(
                    self.context,
                    IntegerAttribute::new(Type::index(self.context), 0),
                    location,
                ),
            );
            pre_entry_block.insert_operation_after(
                op1,
                memref::store(op1.result(0)?.into(), op0.result(0)?.into(), &[], location),
            );

            self.metadata
                .borrow_mut()
                .insert(TailRecursionMeta::new(op0.result(0)?.into(), entry_block))
                .expect("should not have this metadata inserted yet");
        }

        Ok(())
    }

    /// Returns the [`CoreTypeConcrete`] for the given type id.
    fn get_type(&self, type_id: &ConcreteTypeId) -> Result<&CoreTypeConcrete, Error> {
        match self.registry.get_type(type_id) {
            Ok(x) => Ok(x),
            Err(e) => Err(e.into()),
        }
    }

    /// Builds the MLIR types from the given type id.
    fn build_type(&self, type_id: &ConcreteTypeId) -> Result<Type, Error> {
        self.get_type(type_id)?.build(
            self.context,
            self.module,
            self.registry,
            *self.metadata.borrow_mut(),
            type_id,
        )
    }

    /// Returns true if the type is a zero-sized type built-in.
    fn type_is_zst_builtin(&self, type_id: &ConcreteTypeId) -> bool {
        self.get_type(type_id)
            .map_or(false, |x| x.is_builtin() && x.is_zst(self.registry))
    }

    /// Extracts the types from the registry and builds them into MLIR types.
    fn extract_types(
        &'c self,
        type_ids: &'c [ConcreteTypeId],
    ) -> impl 'c + Iterator<Item = Result<Type<'c>, Error>> {
        type_ids.iter().filter_map(|id| {
            if self.type_is_zst_builtin(id) {
                None
            } else {
                Some(self.build_type(id))
            }
        })
    }
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
