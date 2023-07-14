//! # Runtime library bindings
//!
//! This metadata ensures that the bindings to the runtime functions exist in the current
//! compilation context.

use crate::error::libfuncs::Result;
use melior::{
    dialect::{func, llvm},
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType},
        Block, Identifier, Location, Module, OperationRef, Region, Value,
    },
    Context,
};
use std::{collections::HashSet, marker::PhantomData};

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
enum RuntimeBinding {
    LibfuncDebugPrint,
    LibfuncPedersen,
    DictNew,
    DictGet,
    DictInsert,
}

/// Runtime library bindings metadata.
#[derive(Debug)]
pub struct RuntimeBindingsMeta {
    active_map: HashSet<RuntimeBinding>,
    phantom: PhantomData<()>,
}

impl RuntimeBindingsMeta {
    /// Register if necessary, then invoke the `debug::print()` function.
    #[allow(clippy::too_many_arguments)]
    pub fn libfunc_debug_print<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        target_fd: Value<'c, '_>,
        values_ptr: Value<'c, '_>,
        values_len: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, 'a>>
    where
        'c: 'a,
    {
        if self.active_map.insert(RuntimeBinding::LibfuncDebugPrint) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "cairo_native__libfunc__debug__print"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            IntegerType::new(context, 32).into(),
                            llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0),
                            IntegerType::new(context, 32).into(),
                        ],
                        &[IntegerType::new(context, 32).into()],
                    )
                    .into(),
                ),
                Region::new(),
                &[(
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                )],
                Location::unknown(context),
            ));
        }

        Ok(block
            .append_operation(func::call(
                context,
                FlatSymbolRefAttribute::new(context, "cairo_native__libfunc__debug__print"),
                &[target_fd, values_ptr, values_len],
                &[IntegerType::new(context, 32).into()],
                location,
            ))
            .result(0)?
            .into())
    }

    /// Register if necessary, then invoke the `pedersen()` function.
    #[allow(clippy::too_many_arguments)]
    pub fn libfunc_pedersen<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        dst_ptr: Value<'c, '_>,
        lhs_ptr: Value<'c, '_>,
        rhs_ptr: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<OperationRef<'c, 'a>>
    where
        'c: 'a,
    {
        if self.active_map.insert(RuntimeBinding::LibfuncPedersen) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "cairo_native__libfunc_pedersen"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            llvm::r#type::pointer(IntegerType::new(context, 256).into(), 0),
                            llvm::r#type::pointer(IntegerType::new(context, 256).into(), 0),
                            llvm::r#type::pointer(IntegerType::new(context, 256).into(), 0),
                        ],
                        &[],
                    )
                    .into(),
                ),
                Region::new(),
                &[(
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                )],
                Location::unknown(context),
            ));
        }

        Ok(block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "cairo_native__libfunc_pedersen"),
            &[dst_ptr, lhs_ptr, rhs_ptr],
            &[],
            location,
        )))
    }

    /// Register if necessary, then invoke the `dict_alloc_new()` function.
    ///
    /// Returns a opaque pointer as the result.
    #[allow(clippy::too_many_arguments)]
    pub fn dict_alloc_new<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
    ) -> Result<OperationRef<'c, 'a>>
    where
        'c: 'a,
    {
        if self.active_map.insert(RuntimeBinding::DictNew) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "cairo_native__alloc_dict"),
                TypeAttribute::new(
                    FunctionType::new(context, &[], &[llvm::r#type::opaque_pointer(context)])
                        .into(),
                ),
                Region::new(),
                &[(
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                )],
                Location::unknown(context),
            ));
        }

        Ok(block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "cairo_native__alloc_dict"),
            &[],
            &[llvm::r#type::opaque_pointer(context)],
            location,
        )))
    }

    /// Register if necessary, then invoke the `dict_get()` function.
    ///
    /// Gets the value for a given key, the returned pointer is null if not found.
    ///
    /// Returns a opaque pointer as the result.
    #[allow(clippy::too_many_arguments)]
    pub fn dict_get<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        dict_ptr: Value<'c, 'a>, // ptr to the dict
        key_ptr: Value<'c, 'a>,  // key must be a ptr to felt252
        location: Location<'c>,
    ) -> Result<OperationRef<'c, 'a>>
    where
        'c: 'a,
    {
        if self.active_map.insert(RuntimeBinding::DictGet) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "cairo_native__dict_get"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            llvm::r#type::opaque_pointer(context),
                            llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0),
                        ],
                        &[llvm::r#type::opaque_pointer(context)],
                    )
                    .into(),
                ),
                Region::new(),
                &[(
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                )],
                Location::unknown(context),
            ));
        }

        Ok(block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "cairo_native__dict_get"),
            &[dict_ptr, key_ptr],
            &[llvm::r#type::opaque_pointer(context)],
            location,
        )))
    }

    /// Register if necessary, then invoke the `dict_insert()` function.
    ///
    /// Inserts the provided key value. Returning the old one or nullptr if there was none.
    ///
    /// Returns a opaque pointer as the result.
    #[allow(clippy::too_many_arguments)]
    pub fn dict_insert<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        dict_ptr: Value<'c, 'a>,  // ptr to the dict
        key_ptr: Value<'c, 'a>,   // key must be a ptr to felt252
        value_ptr: Value<'c, 'a>, // value must be a opaque non null ptr
        location: Location<'c>,
    ) -> Result<OperationRef<'c, 'a>>
    where
        'c: 'a,
    {
        if self.active_map.insert(RuntimeBinding::DictGet) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "cairo_native__dict_insert"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            llvm::r#type::opaque_pointer(context),
                            llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0),
                            llvm::r#type::opaque_pointer(context),
                        ],
                        &[llvm::r#type::opaque_pointer(context)],
                    )
                    .into(),
                ),
                Region::new(),
                &[(
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                )],
                Location::unknown(context),
            ));
        }

        Ok(block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "cairo_native__dict_insert"),
            &[dict_ptr, key_ptr, value_ptr],
            &[llvm::r#type::opaque_pointer(context)],
            location,
        )))
    }
}

impl Default for RuntimeBindingsMeta {
    fn default() -> Self {
        Self {
            active_map: HashSet::new(),
            phantom: PhantomData,
        }
    }
}
