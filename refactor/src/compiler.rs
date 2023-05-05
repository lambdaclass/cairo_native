//! The compiler implementation.

use self::{
    funcs::FuncStorage,
    libfuncs::{CompiledLibfunc, LibfuncStorage},
    types::{CompiledType, TypeStorage},
};
pub use self::{
    funcs::SierraFuncId,
    libfuncs::{LibfuncImpl, SierraLibfuncId},
    types::{SierraTypeId, TypeLayout},
};
use crate::{
    compiler::funcs::CompiledFunc,
    database::{libfuncs::LibfuncDatabase, types::TypeDatabase},
    ffi::{
        self, aux_Block_createAndPush, aux_ModuleOp_print, aux_Region_getFirstBlock,
        aux_TypeAttr_get, make_Block, make_MLIRContext, make_ModuleOp, make_OpBuilder,
        make_OperationState, make_Region, proxy_Block_getArgument, proxy_Block_push_back,
        proxy_ModuleOp_getBodyRegion, proxy_OpBuilder_getFunctionType,
        proxy_OpBuilder_getIntegerType, proxy_OpBuilder_getStringAttr,
        proxy_OpBuilder_getUnknownLoc, proxy_OperationState_addAttribute,
        proxy_OperationState_addOperand, proxy_OperationState_addRegion,
        proxy_OperationState_addSuccessor, proxy_OperationState_addType,
        proxy_Operation_getNumResults, proxy_Operation_getResult, Block, MLIRContext, ModuleOp,
        OpBuilder, Operation, OperationState, Region, Type,
    },
};
use cairo_lang_sierra::program::{
    Function, GenStatement, LibfuncDeclaration, Statement, TypeDeclaration,
};
use cxx::UniquePtr;
use std::{
    collections::{btree_map::Entry, BTreeMap},
    rc::Rc,
};

mod funcs;
mod libfuncs;
mod types;

/// A compiled program.
pub struct CompiledProgram {
    context: UniquePtr<MLIRContext>,
    module: UniquePtr<ModuleOp>,
}

/// The MLIR compiler.
///
/// Contains MLIR stuff:
///   - `Context`
///   - `ModuleOp`
///   - `OpBuilder`
///
/// Contains the declared items:
///   - Functions
///   - Libfuncs
///   - Types
///
/// Once everything has been processed, calling `.build()` should build everything togerther into a
/// built module.
///
pub struct Compiler {
    context: UniquePtr<MLIRContext>,
    builder: UniquePtr<OpBuilder>,

    compiled_funcs: FuncStorage,
    compiled_libfuncs: LibfuncStorage,
    compiled_types: TypeStorage,
}

impl Compiler {
    /// Process a libfunc into a compiled libfunc.
    pub fn process_libfunc(
        &mut self,
        libfunc_database: &LibfuncDatabase,
        libfunc_declaration: &LibfuncDeclaration,
    ) -> SierraLibfuncId {
        let sierra_libfunc_id = SierraLibfuncId::new(libfunc_declaration.id.id);

        let processor = &libfunc_database[libfunc_declaration.long_id.generic_id.0.as_str()];
        let libfunc_impl = processor(&TypeFactory(self), libfunc_declaration);

        self.compiled_libfuncs.insert(
            sierra_libfunc_id,
            CompiledLibfunc {
                _id: libfunc_declaration.id.clone(),
                _libfunc_setup: libfunc_impl.libfunc_setup,
                libfunc_invoke: libfunc_impl.libfunc_invoke,
            },
        );
        sierra_libfunc_id
    }

    /// Process a type into a compiled type.
    pub fn process_type(
        &mut self,
        type_database: &TypeDatabase,
        type_declaration: &TypeDeclaration,
    ) -> SierraTypeId {
        let sierra_type_id = SierraTypeId::new(type_declaration.id.id);
        assert!(!self.compiled_types.contains_key(&sierra_type_id));

        let processor = &type_database[type_declaration.long_id.generic_id.0.as_str()];
        let type_layout = processor(&TypeFactory(self), type_declaration);

        self.compiled_types.insert(sierra_type_id, type_layout.0);
        sierra_type_id
    }

    /// Declare a function to be generated later on.
    pub fn declare_func(&mut self, func_declaration: &Function) -> SierraFuncId {
        let func_id = SierraFuncId::new(func_declaration.id.id);
        assert!(!self.compiled_funcs.contains_key(&func_id));

        let compiled_func = CompiledFunc::new(self, func_declaration);

        self.compiled_funcs.insert(func_id, compiled_func);
        func_id
    }

    /// Implement the declared functions' bodies and link everything together.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn build(self, body: &[Statement]) -> CompiledProgram {
        // Algorithm:
        //   - For each statement:
        //     - Create a new block with the libfunc's args as args.
        //   - For each statement:
        //     - Invoke the libfunc with the block to generate the MLIR.
        //   - Find branch boundaries.
        //   - Combine blocks which do not cross boundaries.

        // Create the program module and global region.
        let module = make_ModuleOp(&proxy_OpBuilder_getUnknownLoc(&self.builder));
        let region = proxy_ModuleOp_getBodyRegion(&module);

        // TODO: Process libfunc setups.

        for compiled_func in self.compiled_funcs.values() {
            let mut local_region = make_Region();

            // Create and populate the state with the function arguments.
            let mut state = BTreeMap::new();

            let entry_block = unsafe {
                make_Block(
                    &self.builder,
                    local_region.pin_mut().get_unchecked_mut() as *mut _,
                )
            };
            for (arg_index, state_loc) in compiled_func.arg_mappings.iter().enumerate() {
                assert!(state
                    .insert(
                        state_loc.id,
                        Rc::new(unsafe {
                            proxy_Block_getArgument(entry_block, arg_index.try_into().unwrap())
                        })
                    )
                    .is_none());
            }

            // Walk the function's execution tree, creating a block and invoking the corresponding
            // libfunc for each statement.
            let mut blocks = BTreeMap::<usize, (*mut Block, Option<bool>)>::from([(
                compiled_func.entry_point.0,
                (entry_block, None),
            )]);

            let mut queue = vec![compiled_func.entry_point];
            while let Some(statement_idx) = queue.pop() {
                let block_ref = match blocks.entry(statement_idx.0) {
                    Entry::Vacant(entry) => Some(
                        entry
                            .insert((
                                unsafe {
                                    make_Block(
                                        &self.builder,
                                        local_region.pin_mut().get_unchecked_mut() as *mut _,
                                    )
                                },
                                None,
                            ))
                            .0,
                    ),
                    Entry::Occupied(mut entry) => match &mut entry.get_mut().1 {
                        Some(is_branch_target) => {
                            *is_branch_target = true;
                            None
                        }
                        None => {
                            entry.get_mut().1 = Some(false);
                            Some(entry.into_mut().0)
                        }
                    },
                };

                if let Some(block_ref) = block_ref {
                    match &body[statement_idx.0] {
                        GenStatement::Invocation(invocation) => {
                            queue.extend(
                                invocation
                                    .branches
                                    .iter()
                                    .map(|x| statement_idx.next(&x.target)),
                            );

                            let compiled_libfunc = &self.compiled_libfuncs
                                [&SierraLibfuncId::new(invocation.libfunc_id.id)];

                            let args = invocation
                                .args
                                .iter()
                                .map(|x| Rc::clone(&state[&x.id]))
                                .map(Value)
                                .collect::<Vec<_>>();

                            let targets = invocation
                                .branches
                                .iter()
                                .map(|branch| {
                                    blocks
                                        .entry(statement_idx.next(&branch.target).0)
                                        .or_insert_with(|| unsafe {
                                            (
                                                make_Block(
                                                    &self.builder,
                                                    local_region.pin_mut().get_unchecked_mut()
                                                        as *mut _,
                                                ),
                                                None,
                                            )
                                        })
                                        .0
                                })
                                .map(Successor)
                                .collect::<Vec<_>>();

                            let rets = (compiled_libfunc.libfunc_invoke)(
                                &OpFactory(&self, block_ref),
                                &args,
                                &targets,
                            );
                            assert_eq!(rets.len(), invocation.branches.len());
                            assert!(rets
                                .iter()
                                .zip(&invocation.branches)
                                .all(|(a, b)| a.len() == b.results.len()));

                            rets.into_iter()
                                .zip(&invocation.branches)
                                .flat_map(|(a, b)| a.into_iter().zip(&b.results))
                                .for_each(|(val, loc)| {
                                    assert!(state.insert(loc.id, val.0).is_none());
                                });
                        }
                        GenStatement::Return(ret_vars) => {
                            ret_vars
                                .iter()
                                .fold(
                                    OpFactory(&self, block_ref).builder("func.return"),
                                    |builder, ret_var| {
                                        builder.add_operand(&Value(Rc::clone(&state[&ret_var.id])))
                                    },
                                )
                                .build();
                        }
                    }
                }
            }

            // Function op.
            let fn_args = compiled_func
                .arguments
                .iter()
                .map(|x| Rc::clone(&self.compiled_types[x]))
                .map(TypeLayout)
                .collect::<Vec<_>>();
            let fn_rets = compiled_func
                .return_types
                .iter()
                .map(|x| Rc::clone(&self.compiled_types[x]))
                .map(TypeLayout)
                .collect::<Vec<_>>();

            OpFactory(&self, unsafe { aux_Region_getFirstBlock(region) })
                .builder("func.func")
                .add_region(local_region)
                .add_string_attribute("sym_name", "main")
                .add_string_attribute("sym_visibility", "public")
                .add_type_attribute(
                    "function_type",
                    &TypeFactory(&self).function_type(&fn_args, &fn_rets),
                )
                .build();
        }

        // TODO: Canonicalize.

        CompiledProgram {
            context: self.context,
            module,
        }
    }
}

impl Default for Compiler {
    fn default() -> Self {
        let context = make_MLIRContext();
        let builder = make_OpBuilder(&context);

        Self {
            context,
            builder,
            compiled_funcs: BTreeMap::default(),
            compiled_libfuncs: BTreeMap::default(),
            compiled_types: BTreeMap::default(),
        }
    }
}

/// A proxy to the context with extra methods, used for building types.
pub struct TypeFactory<'c>(&'c Compiler);

impl<'c> TypeFactory<'c> {
    /// Create an array type.
    #[must_use]
    pub fn array_type(&self, _inner: &TypeLayout) -> TypeLayout {
        todo!()
    }

    /// Create a function type.
    #[must_use]
    pub fn function_type(&self, args: &[TypeLayout], rets: &[TypeLayout]) -> TypeLayout {
        TypeLayout(Rc::new(CompiledType::Other {
            mlir_type: proxy_OpBuilder_getFunctionType(
                &self.0.builder,
                &args
                    .iter()
                    .map(|x| x.0.mlir_type() as *const Type as *mut Type)
                    .collect::<Vec<_>>(),
                &rets
                    .iter()
                    .map(|x| x.0.mlir_type() as *const Type as *mut Type)
                    .collect::<Vec<_>>(),
            ),
        }))
    }

    /// Create an integer type.
    #[must_use]
    pub fn integer_type(&self, num_bits: u32) -> TypeLayout {
        // TODO: Check if type has been declared (id).
        TypeLayout(Rc::new(CompiledType::Integer {
            id: None,
            mlir_type: proxy_OpBuilder_getIntegerType(&self.0.builder, num_bits),
            width: num_bits,
        }))
    }

    /// Create a struct type.
    #[must_use]
    pub fn struct_type<'a>(&self, fields: impl IntoIterator<Item = &'a TypeLayout>) -> TypeLayout {
        let _ = fields.into_iter();
        todo!()
    }
}

/// A proxy to the context with extra methods, used for building libfuncs.
pub struct OpFactory<'a>(&'a Compiler, *mut Block);

impl<'a> OpFactory<'a> {
    /// Create a new operation builder.
    #[must_use]
    pub fn builder(&'a self, name: &str) -> OperationBuilder<'a> {
        OperationBuilder {
            factory: self,
            state: make_OperationState(&proxy_OpBuilder_getUnknownLoc(&self.0.builder), name),
        }
    }
}

/// An operation builder.
pub struct OperationBuilder<'a> {
    factory: &'a OpFactory<'a>,
    state: UniquePtr<OperationState>,
}

impl<'a> OperationBuilder<'a> {
    /// Finalize and build the operation.
    #[allow(clippy::must_use_candidate)]
    pub fn build(self) -> Vec<Value> {
        unsafe {
            let op = aux_Block_createAndPush(self.factory.1, &self.state);

            let num_results = proxy_Operation_getNumResults(op);
            (0..num_results)
                .map(|x| proxy_Operation_getResult(op, x))
                .map(Rc::new)
                .map(Value)
                .collect()
        }
    }

    /// Finalize and build the operation.
    #[allow(clippy::must_use_candidate)]
    pub fn build_with_regions(self) -> Vec<Value> {
        unsafe {
            let op = aux_Block_createAndPush(self.factory.1, &self.state);

            let num_results = proxy_Operation_getNumResults(op);
            (0..num_results)
                .map(|x| proxy_Operation_getResult(op, x))
                .map(Rc::new)
                .map(Value)
                .collect()
        }
    }

    /// Add an operand.
    #[must_use]
    pub fn add_operand(self, value: &Value) -> Self {
        proxy_OperationState_addOperand(&self.state, &value.0);
        self
    }

    /// Add a region.
    #[must_use]
    pub fn add_region(self, region: UniquePtr<Region>) -> Self {
        proxy_OperationState_addRegion(&self.state, region);
        self
    }

    /// Add an expected return value.
    #[must_use]
    pub fn add_return_value(self, return_type: &TypeLayout) -> Self {
        proxy_OperationState_addType(&self.state, return_type.0.mlir_type());
        self
    }

    /// Add a successor block to the operation.
    #[must_use]
    pub fn add_successor(self, successor: &Successor) -> Self {
        unsafe {
            proxy_OperationState_addSuccessor(&self.state, successor.0);
        }
        self
    }

    /// Add an attribute to the operation.
    #[must_use]
    pub fn add_string_attribute(self, name: &str, value: &str) -> Self {
        proxy_OperationState_addAttribute(
            &self.state,
            name,
            &proxy_OpBuilder_getStringAttr(&self.factory.0.builder, value),
        );
        self
    }

    /// Add an attribute to the operation.
    #[must_use]
    pub fn add_type_attribute(self, name: &str, type_layout: &TypeLayout) -> Self {
        proxy_OperationState_addAttribute(
            &self.state,
            name,
            &aux_TypeAttr_get(type_layout.0.mlir_type()),
        );
        self
    }
}

/// An MLIR value.
#[derive(Clone)]
pub struct Value(Rc<UniquePtr<ffi::Value>>);

/// An operation's successor block.
pub struct Successor(*mut Block);
