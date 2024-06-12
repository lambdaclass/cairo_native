////! # Pedersen hashing libfuncs
//! # Pedersen hashing libfuncs
////!
//!
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    block_ext::BlockExt,
    block_ext::BlockExt,
//    error::Result,
    error::Result,
//    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
//    utils::{get_integer_layout, ProgramRegistryExt},
    utils::{get_integer_layout, ProgramRegistryExt},
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        lib_func::SignatureOnlyConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
//        pedersen::PedersenConcreteLibfunc,
        pedersen::PedersenConcreteLibfunc,
//        ConcreteLibfunc,
        ConcreteLibfunc,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::{
    dialect::{
//        arith,
        arith,
//        llvm::{self, LoadStoreOptions},
        llvm::{self, LoadStoreOptions},
//        ods,
        ods,
//    },
    },
//    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location},
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location},
//    Context,
    Context,
//};
};
//

///// Select and call the correct libfunc builder function from the selector.
/// Select and call the correct libfunc builder function from the selector.
//pub fn build<'ctx, 'this>(
pub fn build<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    selector: &PedersenConcreteLibfunc,
    selector: &PedersenConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        PedersenConcreteLibfunc::PedersenHash(info) => {
        PedersenConcreteLibfunc::PedersenHash(info) => {
//            build_pedersen(context, registry, entry, location, helper, metadata, info)
            build_pedersen(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

//pub fn build_pedersen<'ctx>(
pub fn build_pedersen<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'ctx Block<'ctx>,
    entry: &'ctx Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, '_>,
    helper: &LibfuncHelper<'ctx, '_>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    metadata
    metadata
//        .get_mut::<RuntimeBindingsMeta>()
        .get_mut::<RuntimeBindingsMeta>()
//        .expect("Runtime library not available.");
        .expect("Runtime library not available.");
//

//    let pedersen_builtin =
    let pedersen_builtin =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let felt252_ty = registry.build_type(
    let felt252_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.param_signatures()[1].ty,
        &info.param_signatures()[1].ty,
//    )?;
    )?;
//

//    let i256_ty = IntegerType::new(context, 256).into();
    let i256_ty = IntegerType::new(context, 256).into();
//    let layout_i256 = get_integer_layout(256);
    let layout_i256 = get_integer_layout(256);
//

//    let lhs = entry.argument(1)?.into();
    let lhs = entry.argument(1)?.into();
//    let rhs = entry.argument(2)?.into();
    let rhs = entry.argument(2)?.into();
//

//    // We must extend to i256 because bswap must be an even number of bytes.
    // We must extend to i256 because bswap must be an even number of bytes.
//

//    let lhs_ptr =
    let lhs_ptr =
//        helper
        helper
//            .init_block()
            .init_block()
//            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;
            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;
//    let rhs_ptr =
    let rhs_ptr =
//        helper
        helper
//            .init_block()
            .init_block()
//            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;
            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;
//    let dst_ptr =
    let dst_ptr =
//        helper
        helper
//            .init_block()
            .init_block()
//            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;
            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;
//

//    let lhs_i256 = entry.append_op_result(arith::extui(lhs, i256_ty, location))?;
    let lhs_i256 = entry.append_op_result(arith::extui(lhs, i256_ty, location))?;
//    let rhs_i256 = entry.append_op_result(arith::extui(rhs, i256_ty, location))?;
    let rhs_i256 = entry.append_op_result(arith::extui(rhs, i256_ty, location))?;
//

//    let lhs_be =
    let lhs_be =
//        entry.append_op_result(ods::llvm::intr_bswap(context, lhs_i256, location).into())?;
        entry.append_op_result(ods::llvm::intr_bswap(context, lhs_i256, location).into())?;
//

//    let rhs_be =
    let rhs_be =
//        entry.append_op_result(ods::llvm::intr_bswap(context, rhs_i256, location).into())?;
        entry.append_op_result(ods::llvm::intr_bswap(context, rhs_i256, location).into())?;
//

//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        lhs_be,
        lhs_be,
//        lhs_ptr,
        lhs_ptr,
//        location,
        location,
//        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
//            IntegerType::new(context, 64).into(),
            IntegerType::new(context, 64).into(),
//            layout_i256.align().try_into()?,
            layout_i256.align().try_into()?,
//        ))),
        ))),
//    ));
    ));
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        rhs_be,
        rhs_be,
//        rhs_ptr,
        rhs_ptr,
//        location,
        location,
//        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
//            IntegerType::new(context, 64).into(),
            IntegerType::new(context, 64).into(),
//            layout_i256.align().try_into()?,
            layout_i256.align().try_into()?,
//        ))),
        ))),
//    ));
    ));
//

//    let runtime_bindings = metadata
    let runtime_bindings = metadata
//        .get_mut::<RuntimeBindingsMeta>()
        .get_mut::<RuntimeBindingsMeta>()
//        .expect("Runtime library not available.");
        .expect("Runtime library not available.");
//

//    runtime_bindings
    runtime_bindings
//        .libfunc_pedersen(context, helper, entry, dst_ptr, lhs_ptr, rhs_ptr, location)?;
        .libfunc_pedersen(context, helper, entry, dst_ptr, lhs_ptr, rhs_ptr, location)?;
//

//    let result_be = entry.append_op_result(llvm::load(
    let result_be = entry.append_op_result(llvm::load(
//        context,
        context,
//        dst_ptr,
        dst_ptr,
//        i256_ty,
        i256_ty,
//        location,
        location,
//        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
//            IntegerType::new(context, 64).into(),
            IntegerType::new(context, 64).into(),
//            layout_i256.align().try_into()?,
            layout_i256.align().try_into()?,
//        ))),
        ))),
//    ))?;
    ))?;
//

//    let op = entry.append_op_result(ods::llvm::intr_bswap(context, result_be, location).into())?;
    let op = entry.append_op_result(ods::llvm::intr_bswap(context, result_be, location).into())?;
//

//    let result = entry.append_op_result(arith::trunci(op, felt252_ty, location))?;
    let result = entry.append_op_result(arith::trunci(op, felt252_ty, location))?;
//

//    entry.append_operation(helper.br(0, &[pedersen_builtin, result], location));
    entry.append_operation(helper.br(0, &[pedersen_builtin, result], location));
//

//    Ok(())
    Ok(())
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use crate::utils::test::{load_cairo, run_program_assert_output};
    use crate::utils::test::{load_cairo, run_program_assert_output};
//

//    use starknet_types_core::felt::Felt;
    use starknet_types_core::felt::Felt;
//

//    #[test]
    #[test]
//    fn run_pedersen() {
    fn run_pedersen() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use core::pedersen::pedersen;
            use core::pedersen::pedersen;
//

//            fn run_test(a: felt252, b: felt252) -> felt252 {
            fn run_test(a: felt252, b: felt252) -> felt252 {
//                pedersen(a, b)
                pedersen(a, b)
//            }
            }
//        );
        );
//

//        run_program_assert_output(
        run_program_assert_output(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[Felt::from(2).into(), Felt::from(4).into()],
            &[Felt::from(2).into(), Felt::from(4).into()],
//            Felt::from_dec_str(
            Felt::from_dec_str(
//                "2178161520066714737684323463974044933282313051386084149915030950231093462467",
                "2178161520066714737684323463974044933282313051386084149915030950231093462467",
//            )
            )
//            .unwrap()
            .unwrap()
//            .into(),
            .into(),
//        );
        );
//    }
    }
//}
}
