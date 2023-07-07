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
        if self.active_map.insert(RuntimeBinding::LibfuncDebugPrint) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "cairo_native__libfunc_pedersen"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0),
                            llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0),
                            llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0),
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
}

impl Default for RuntimeBindingsMeta {
    fn default() -> Self {
        Self {
            active_map: HashSet::new(),
            phantom: PhantomData,
        }
    }
}
