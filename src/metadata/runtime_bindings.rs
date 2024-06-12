////! # Runtime library bindings
//! # Runtime library bindings
////!
//!
////! This metadata ensures that the bindings to the runtime functions exist in the current
//! This metadata ensures that the bindings to the runtime functions exist in the current
////! compilation context.
//! compilation context.
//

//use crate::error::Result;
use crate::error::Result;
//use melior::{
use melior::{
//    dialect::{func, llvm},
    dialect::{func, llvm},
//    ir::{
    ir::{
//        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
//        r#type::{FunctionType, IntegerType},
        r#type::{FunctionType, IntegerType},
//        Block, Identifier, Location, Module, OperationRef, Region, Value,
        Block, Identifier, Location, Module, OperationRef, Region, Value,
//    },
    },
//    Context,
    Context,
//};
};
//use std::{collections::HashSet, marker::PhantomData};
use std::{collections::HashSet, marker::PhantomData};
//

//#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
//enum RuntimeBinding {
enum RuntimeBinding {
//    DebugPrint,
    DebugPrint,
//    Pedersen,
    Pedersen,
//    HadesPermutation,
    HadesPermutation,
//    EcPointFromXNz,
    EcPointFromXNz,
//    EcPointTryNewNz,
    EcPointTryNewNz,
//    EcStateAdd,
    EcStateAdd,
//    EcStateAddMul,
    EcStateAddMul,
//    EcStateTryFinalizeNz,
    EcStateTryFinalizeNz,
//    DictNew,
    DictNew,
//    DictGet,
    DictGet,
//    DictGasRefund,
    DictGasRefund,
//    DictInsert,
    DictInsert,
//    DictFree,
    DictFree,
//}
}
//

///// Runtime library bindings metadata.
/// Runtime library bindings metadata.
//#[derive(Debug)]
#[derive(Debug)]
//pub struct RuntimeBindingsMeta {
pub struct RuntimeBindingsMeta {
//    active_map: HashSet<RuntimeBinding>,
    active_map: HashSet<RuntimeBinding>,
//    phantom: PhantomData<()>,
    phantom: PhantomData<()>,
//}
}
//

//impl RuntimeBindingsMeta {
impl RuntimeBindingsMeta {
//    /// Register if necessary, then invoke the `debug::print()` function.
    /// Register if necessary, then invoke the `debug::print()` function.
//    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
//    pub fn libfunc_debug_print<'c, 'a>(
    pub fn libfunc_debug_print<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        target_fd: Value<'c, '_>,
        target_fd: Value<'c, '_>,
//        values_ptr: Value<'c, '_>,
        values_ptr: Value<'c, '_>,
//        values_len: Value<'c, '_>,
        values_len: Value<'c, '_>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<Value<'c, 'a>>
    ) -> Result<Value<'c, 'a>>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(RuntimeBinding::DebugPrint) {
        if self.active_map.insert(RuntimeBinding::DebugPrint) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "cairo_native__libfunc__debug__print"),
                StringAttribute::new(context, "cairo_native__libfunc__debug__print"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(
                    FunctionType::new(
//                        context,
                        context,
//                        &[
                        &[
//                            IntegerType::new(context, 32).into(),
                            IntegerType::new(context, 32).into(),
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                            IntegerType::new(context, 32).into(),
                            IntegerType::new(context, 32).into(),
//                        ],
                        ],
//                        &[IntegerType::new(context, 32).into()],
                        &[IntegerType::new(context, 32).into()],
//                    )
                    )
//                    .into(),
                    .into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        Ok(block
        Ok(block
//            .append_operation(func::call(
            .append_operation(func::call(
//                context,
                context,
//                FlatSymbolRefAttribute::new(context, "cairo_native__libfunc__debug__print"),
                FlatSymbolRefAttribute::new(context, "cairo_native__libfunc__debug__print"),
//                &[target_fd, values_ptr, values_len],
                &[target_fd, values_ptr, values_len],
//                &[IntegerType::new(context, 32).into()],
                &[IntegerType::new(context, 32).into()],
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into())
            .into())
//    }
    }
//

//    /// Register if necessary, then invoke the `pedersen()` function.
    /// Register if necessary, then invoke the `pedersen()` function.
//    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
//    pub fn libfunc_pedersen<'c, 'a>(
    pub fn libfunc_pedersen<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        dst_ptr: Value<'c, '_>,
        dst_ptr: Value<'c, '_>,
//        lhs_ptr: Value<'c, '_>,
        lhs_ptr: Value<'c, '_>,
//        rhs_ptr: Value<'c, '_>,
        rhs_ptr: Value<'c, '_>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<OperationRef<'c, 'a>>
    ) -> Result<OperationRef<'c, 'a>>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(RuntimeBinding::Pedersen) {
        if self.active_map.insert(RuntimeBinding::Pedersen) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "cairo_native__libfunc__pedersen"),
                StringAttribute::new(context, "cairo_native__libfunc__pedersen"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(
                    FunctionType::new(
//                        context,
                        context,
//                        &[
                        &[
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                        ],
                        ],
//                        &[],
                        &[],
//                    )
                    )
//                    .into(),
                    .into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        Ok(block.append_operation(func::call(
        Ok(block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "cairo_native__libfunc__pedersen"),
            FlatSymbolRefAttribute::new(context, "cairo_native__libfunc__pedersen"),
//            &[dst_ptr, lhs_ptr, rhs_ptr],
            &[dst_ptr, lhs_ptr, rhs_ptr],
//            &[],
            &[],
//            location,
            location,
//        )))
        )))
//    }
    }
//

//    /// Register if necessary, then invoke the `poseidon()` function.
    /// Register if necessary, then invoke the `poseidon()` function.
//    /// The passed pointers serve both as in/out pointers. I.E results are stored in the given pointers.
    /// The passed pointers serve both as in/out pointers. I.E results are stored in the given pointers.
//    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
//    pub fn libfunc_hades_permutation<'c, 'a>(
    pub fn libfunc_hades_permutation<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        op0_ptr: Value<'c, '_>,
        op0_ptr: Value<'c, '_>,
//        op1_ptr: Value<'c, '_>,
        op1_ptr: Value<'c, '_>,
//        op2_ptr: Value<'c, '_>,
        op2_ptr: Value<'c, '_>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<OperationRef<'c, 'a>>
    ) -> Result<OperationRef<'c, 'a>>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(RuntimeBinding::HadesPermutation) {
        if self.active_map.insert(RuntimeBinding::HadesPermutation) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "cairo_native__libfunc__hades_permutation"),
                StringAttribute::new(context, "cairo_native__libfunc__hades_permutation"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(
                    FunctionType::new(
//                        context,
                        context,
//                        &[
                        &[
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                        ],
                        ],
//                        &[],
                        &[],
//                    )
                    )
//                    .into(),
                    .into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        Ok(block.append_operation(func::call(
        Ok(block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "cairo_native__libfunc__hades_permutation"),
            FlatSymbolRefAttribute::new(context, "cairo_native__libfunc__hades_permutation"),
//            &[op0_ptr, op1_ptr, op2_ptr],
            &[op0_ptr, op1_ptr, op2_ptr],
//            &[],
            &[],
//            location,
            location,
//        )))
        )))
//    }
    }
//

//    /// Register if necessary, then invoke the `ec_point_from_x_nz()` function.
    /// Register if necessary, then invoke the `ec_point_from_x_nz()` function.
//    pub fn libfunc_ec_point_from_x_nz<'c, 'a>(
    pub fn libfunc_ec_point_from_x_nz<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        point_ptr: Value<'c, '_>,
        point_ptr: Value<'c, '_>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<OperationRef<'c, 'a>>
    ) -> Result<OperationRef<'c, 'a>>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(RuntimeBinding::EcPointFromXNz) {
        if self.active_map.insert(RuntimeBinding::EcPointFromXNz) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "cairo_native__libfunc__ec__ec_point_from_x_nz"),
                StringAttribute::new(context, "cairo_native__libfunc__ec__ec_point_from_x_nz"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(
                    FunctionType::new(
//                        context,
                        context,
//                        &[llvm::r#type::pointer(context, 0)],
                        &[llvm::r#type::pointer(context, 0)],
//                        &[IntegerType::new(context, 1).into()],
                        &[IntegerType::new(context, 1).into()],
//                    )
                    )
//                    .into(),
                    .into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        Ok(block.append_operation(func::call(
        Ok(block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "cairo_native__libfunc__ec__ec_point_from_x_nz"),
            FlatSymbolRefAttribute::new(context, "cairo_native__libfunc__ec__ec_point_from_x_nz"),
//            &[point_ptr],
            &[point_ptr],
//            &[IntegerType::new(context, 1).into()],
            &[IntegerType::new(context, 1).into()],
//            location,
            location,
//        )))
        )))
//    }
    }
//

//    /// Register if necessary, then invoke the `ec_point_try_new_nz()` function.
    /// Register if necessary, then invoke the `ec_point_try_new_nz()` function.
//    pub fn libfunc_ec_point_try_new_nz<'c, 'a>(
    pub fn libfunc_ec_point_try_new_nz<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        point_ptr: Value<'c, '_>,
        point_ptr: Value<'c, '_>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<OperationRef<'c, 'a>>
    ) -> Result<OperationRef<'c, 'a>>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(RuntimeBinding::EcPointTryNewNz) {
        if self.active_map.insert(RuntimeBinding::EcPointTryNewNz) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "cairo_native__libfunc__ec__ec_point_try_new_nz"),
                StringAttribute::new(context, "cairo_native__libfunc__ec__ec_point_try_new_nz"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(
                    FunctionType::new(
//                        context,
                        context,
//                        &[llvm::r#type::pointer(context, 0)],
                        &[llvm::r#type::pointer(context, 0)],
//                        &[IntegerType::new(context, 1).into()],
                        &[IntegerType::new(context, 1).into()],
//                    )
                    )
//                    .into(),
                    .into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        Ok(block.append_operation(func::call(
        Ok(block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "cairo_native__libfunc__ec__ec_point_try_new_nz"),
            FlatSymbolRefAttribute::new(context, "cairo_native__libfunc__ec__ec_point_try_new_nz"),
//            &[point_ptr],
            &[point_ptr],
//            &[IntegerType::new(context, 1).into()],
            &[IntegerType::new(context, 1).into()],
//            location,
            location,
//        )))
        )))
//    }
    }
//

//    /// Register if necessary, then invoke the `ec_state_add()` function.
    /// Register if necessary, then invoke the `ec_state_add()` function.
//    pub fn libfunc_ec_state_add<'c, 'a>(
    pub fn libfunc_ec_state_add<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        state_ptr: Value<'c, '_>,
        state_ptr: Value<'c, '_>,
//        point_ptr: Value<'c, '_>,
        point_ptr: Value<'c, '_>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<OperationRef<'c, 'a>>
    ) -> Result<OperationRef<'c, 'a>>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(RuntimeBinding::EcStateAdd) {
        if self.active_map.insert(RuntimeBinding::EcStateAdd) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "cairo_native__libfunc__ec__ec_state_add"),
                StringAttribute::new(context, "cairo_native__libfunc__ec__ec_state_add"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(
                    FunctionType::new(
//                        context,
                        context,
//                        &[
                        &[
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                        ],
                        ],
//                        &[],
                        &[],
//                    )
                    )
//                    .into(),
                    .into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        Ok(block.append_operation(func::call(
        Ok(block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "cairo_native__libfunc__ec__ec_state_add"),
            FlatSymbolRefAttribute::new(context, "cairo_native__libfunc__ec__ec_state_add"),
//            &[state_ptr, point_ptr],
            &[state_ptr, point_ptr],
//            &[],
            &[],
//            location,
            location,
//        )))
        )))
//    }
    }
//

//    /// Register if necessary, then invoke the `ec_state_add_mul()` function.
    /// Register if necessary, then invoke the `ec_state_add_mul()` function.
//    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
//    pub fn libfunc_ec_state_add_mul<'c, 'a>(
    pub fn libfunc_ec_state_add_mul<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        state_ptr: Value<'c, '_>,
        state_ptr: Value<'c, '_>,
//        scalar_ptr: Value<'c, '_>,
        scalar_ptr: Value<'c, '_>,
//        point_ptr: Value<'c, '_>,
        point_ptr: Value<'c, '_>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<OperationRef<'c, 'a>>
    ) -> Result<OperationRef<'c, 'a>>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(RuntimeBinding::EcStateAddMul) {
        if self.active_map.insert(RuntimeBinding::EcStateAddMul) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "cairo_native__libfunc__ec__ec_state_add_mul"),
                StringAttribute::new(context, "cairo_native__libfunc__ec__ec_state_add_mul"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(
                    FunctionType::new(
//                        context,
                        context,
//                        &[
                        &[
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                        ],
                        ],
//                        &[],
                        &[],
//                    )
                    )
//                    .into(),
                    .into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        Ok(block.append_operation(func::call(
        Ok(block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "cairo_native__libfunc__ec__ec_state_add_mul"),
            FlatSymbolRefAttribute::new(context, "cairo_native__libfunc__ec__ec_state_add_mul"),
//            &[state_ptr, scalar_ptr, point_ptr],
            &[state_ptr, scalar_ptr, point_ptr],
//            &[],
            &[],
//            location,
            location,
//        )))
        )))
//    }
    }
//

//    pub fn libfunc_ec_state_try_finalize_nz<'c, 'a>(
    pub fn libfunc_ec_state_try_finalize_nz<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        point_ptr: Value<'c, '_>,
        point_ptr: Value<'c, '_>,
//        state_ptr: Value<'c, '_>,
        state_ptr: Value<'c, '_>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<OperationRef<'c, 'a>>
    ) -> Result<OperationRef<'c, 'a>>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(RuntimeBinding::EcStateTryFinalizeNz) {
        if self.active_map.insert(RuntimeBinding::EcStateTryFinalizeNz) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(
                StringAttribute::new(
//                    context,
                    context,
//                    "cairo_native__libfunc__ec__ec_state_try_finalize_nz",
                    "cairo_native__libfunc__ec__ec_state_try_finalize_nz",
//                ),
                ),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(
                    FunctionType::new(
//                        context,
                        context,
//                        &[
                        &[
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                        ],
                        ],
//                        &[IntegerType::new(context, 1).into()],
                        &[IntegerType::new(context, 1).into()],
//                    )
                    )
//                    .into(),
                    .into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                location,
                location,
//            ));
            ));
//        }
        }
//

//        Ok(block.append_operation(func::call(
        Ok(block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(
            FlatSymbolRefAttribute::new(
//                context,
                context,
//                "cairo_native__libfunc__ec__ec_state_try_finalize_nz",
                "cairo_native__libfunc__ec__ec_state_try_finalize_nz",
//            ),
            ),
//            &[point_ptr, state_ptr],
            &[point_ptr, state_ptr],
//            &[IntegerType::new(context, 1).into()],
            &[IntegerType::new(context, 1).into()],
//            location,
            location,
//        )))
        )))
//    }
    }
//

//    /// Register if necessary, then invoke the `dict_alloc_new()` function.
    /// Register if necessary, then invoke the `dict_alloc_new()` function.
//    ///
    ///
//    /// Returns a opaque pointer as the result.
    /// Returns a opaque pointer as the result.
//    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
//    pub fn dict_alloc_new<'c, 'a>(
    pub fn dict_alloc_new<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<OperationRef<'c, 'a>>
    ) -> Result<OperationRef<'c, 'a>>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(RuntimeBinding::DictNew) {
        if self.active_map.insert(RuntimeBinding::DictNew) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "cairo_native__alloc_dict"),
                StringAttribute::new(context, "cairo_native__alloc_dict"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(context, &[], &[llvm::r#type::pointer(context, 0)]).into(),
                    FunctionType::new(context, &[], &[llvm::r#type::pointer(context, 0)]).into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        Ok(block.append_operation(func::call(
        Ok(block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "cairo_native__alloc_dict"),
            FlatSymbolRefAttribute::new(context, "cairo_native__alloc_dict"),
//            &[],
            &[],
//            &[llvm::r#type::pointer(context, 0)],
            &[llvm::r#type::pointer(context, 0)],
//            location,
            location,
//        )))
        )))
//    }
    }
//

//    /// Register if necessary, then invoke the `dict_alloc_new()` function.
    /// Register if necessary, then invoke the `dict_alloc_new()` function.
//    ///
    ///
//    /// Returns a opaque pointer as the result.
    /// Returns a opaque pointer as the result.
//    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
//    pub fn dict_alloc_free<'c, 'a>(
    pub fn dict_alloc_free<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        ptr: Value<'c, 'a>,
        ptr: Value<'c, 'a>,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<OperationRef<'c, 'a>>
    ) -> Result<OperationRef<'c, 'a>>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(RuntimeBinding::DictFree) {
        if self.active_map.insert(RuntimeBinding::DictFree) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "cairo_native__dict_free"),
                StringAttribute::new(context, "cairo_native__dict_free"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(context, &[llvm::r#type::pointer(context, 0)], &[]).into(),
                    FunctionType::new(context, &[llvm::r#type::pointer(context, 0)], &[]).into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        Ok(block.append_operation(func::call(
        Ok(block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "cairo_native__dict_free"),
            FlatSymbolRefAttribute::new(context, "cairo_native__dict_free"),
//            &[ptr],
            &[ptr],
//            &[],
            &[],
//            location,
            location,
//        )))
        )))
//    }
    }
//

//    /// Register if necessary, then invoke the `dict_get()` function.
    /// Register if necessary, then invoke the `dict_get()` function.
//    ///
    ///
//    /// Gets the value for a given key, the returned pointer is null if not found.
    /// Gets the value for a given key, the returned pointer is null if not found.
//    ///
    ///
//    /// Returns a opaque pointer as the result.
    /// Returns a opaque pointer as the result.
//    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
//    pub fn dict_get<'c, 'a>(
    pub fn dict_get<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        dict_ptr: Value<'c, 'a>, // ptr to the dict
        dict_ptr: Value<'c, 'a>, // ptr to the dict
//        key_ptr: Value<'c, 'a>,  // key must be a ptr to Felt
        key_ptr: Value<'c, 'a>,  // key must be a ptr to Felt
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<OperationRef<'c, 'a>>
    ) -> Result<OperationRef<'c, 'a>>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(RuntimeBinding::DictGet) {
        if self.active_map.insert(RuntimeBinding::DictGet) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "cairo_native__dict_get"),
                StringAttribute::new(context, "cairo_native__dict_get"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(
                    FunctionType::new(
//                        context,
                        context,
//                        &[
                        &[
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                        ],
                        ],
//                        &[llvm::r#type::pointer(context, 0)],
                        &[llvm::r#type::pointer(context, 0)],
//                    )
                    )
//                    .into(),
                    .into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        Ok(block.append_operation(func::call(
        Ok(block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "cairo_native__dict_get"),
            FlatSymbolRefAttribute::new(context, "cairo_native__dict_get"),
//            &[dict_ptr, key_ptr],
            &[dict_ptr, key_ptr],
//            &[llvm::r#type::pointer(context, 0)],
            &[llvm::r#type::pointer(context, 0)],
//            location,
            location,
//        )))
        )))
//    }
    }
//

//    /// Register if necessary, then invoke the `dict_insert()` function.
    /// Register if necessary, then invoke the `dict_insert()` function.
//    ///
    ///
//    /// Inserts the provided key value. Returning the old one or nullptr if there was none.
    /// Inserts the provided key value. Returning the old one or nullptr if there was none.
//    ///
    ///
//    /// Returns a opaque pointer as the result.
    /// Returns a opaque pointer as the result.
//    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
//    pub fn dict_insert<'c, 'a>(
    pub fn dict_insert<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        dict_ptr: Value<'c, 'a>,  // ptr to the dict
        dict_ptr: Value<'c, 'a>,  // ptr to the dict
//        key_ptr: Value<'c, 'a>,   // key must be a ptr to Felt
        key_ptr: Value<'c, 'a>,   // key must be a ptr to Felt
//        value_ptr: Value<'c, 'a>, // value must be a opaque non null ptr
        value_ptr: Value<'c, 'a>, // value must be a opaque non null ptr
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<OperationRef<'c, 'a>>
    ) -> Result<OperationRef<'c, 'a>>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(RuntimeBinding::DictInsert) {
        if self.active_map.insert(RuntimeBinding::DictInsert) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "cairo_native__dict_insert"),
                StringAttribute::new(context, "cairo_native__dict_insert"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(
                    FunctionType::new(
//                        context,
                        context,
//                        &[
                        &[
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                        ],
                        ],
//                        &[llvm::r#type::pointer(context, 0)],
                        &[llvm::r#type::pointer(context, 0)],
//                    )
                    )
//                    .into(),
                    .into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        Ok(block.append_operation(func::call(
        Ok(block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "cairo_native__dict_insert"),
            FlatSymbolRefAttribute::new(context, "cairo_native__dict_insert"),
//            &[dict_ptr, key_ptr, value_ptr],
            &[dict_ptr, key_ptr, value_ptr],
//            &[llvm::r#type::pointer(context, 0)],
            &[llvm::r#type::pointer(context, 0)],
//            location,
            location,
//        )))
        )))
//    }
    }
//

//    /// Register if necessary, then invoke the `dict_gas_refund()` function.
    /// Register if necessary, then invoke the `dict_gas_refund()` function.
//    ///
    ///
//    /// Compute the total gas refund for the dictionary.
    /// Compute the total gas refund for the dictionary.
//    ///
    ///
//    /// Returns a u64 of the result.
    /// Returns a u64 of the result.
//    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
//    pub fn dict_gas_refund<'c, 'a>(
    pub fn dict_gas_refund<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        dict_ptr: Value<'c, 'a>, // ptr to the dict
        dict_ptr: Value<'c, 'a>, // ptr to the dict
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<OperationRef<'c, 'a>>
    ) -> Result<OperationRef<'c, 'a>>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(RuntimeBinding::DictGasRefund) {
        if self.active_map.insert(RuntimeBinding::DictGasRefund) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "cairo_native__dict_gas_refund"),
                StringAttribute::new(context, "cairo_native__dict_gas_refund"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(
                    FunctionType::new(
//                        context,
                        context,
//                        &[llvm::r#type::pointer(context, 0)],
                        &[llvm::r#type::pointer(context, 0)],
//                        &[IntegerType::new(context, 64).into()],
                        &[IntegerType::new(context, 64).into()],
//                    )
                    )
//                    .into(),
                    .into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        Ok(block.append_operation(func::call(
        Ok(block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "cairo_native__dict_gas_refund"),
            FlatSymbolRefAttribute::new(context, "cairo_native__dict_gas_refund"),
//            &[dict_ptr],
            &[dict_ptr],
//            &[IntegerType::new(context, 64).into()],
            &[IntegerType::new(context, 64).into()],
//            location,
            location,
//        )))
        )))
//    }
    }
//}
}
//

//impl Default for RuntimeBindingsMeta {
impl Default for RuntimeBindingsMeta {
//    fn default() -> Self {
    fn default() -> Self {
//        Self {
        Self {
//            active_map: HashSet::new(),
            active_map: HashSet::new(),
//            phantom: PhantomData,
            phantom: PhantomData,
//        }
        }
//    }
    }
//}
}
