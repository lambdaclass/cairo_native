////! # Elliptic curve libfuncs
//! # Elliptic curve libfuncs
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    block_ext::BlockExt,
    block_ext::BlockExt,
//    error::{Error, Result},
    error::{Error, Result},
//    metadata::{
    metadata::{
//        prime_modulo::PrimeModuloMeta, runtime_bindings::RuntimeBindingsMeta, MetadataStorage,
        prime_modulo::PrimeModuloMeta, runtime_bindings::RuntimeBindingsMeta, MetadataStorage,
//    },
    },
//    types::felt252::register_prime_modulo_meta,
    types::felt252::register_prime_modulo_meta,
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
//        ec::EcConcreteLibfunc,
        ec::EcConcreteLibfunc,
//        lib_func::SignatureOnlyConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
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
//        arith::{self, CmpiPredicate},
        arith::{self, CmpiPredicate},
//        llvm,
        llvm,
//    },
    },
//    ir::{operation::OperationBuilder, r#type::IntegerType, Block, Location},
    ir::{operation::OperationBuilder, r#type::IntegerType, Block, Location},
//    Context,
    Context,
//};
};
//use num_bigint::{BigInt, ToBigInt};
use num_bigint::{BigInt, ToBigInt};
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
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
//    selector: &EcConcreteLibfunc,
    selector: &EcConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        EcConcreteLibfunc::IsZero(info) => {
        EcConcreteLibfunc::IsZero(info) => {
//            build_is_zero(context, registry, entry, location, helper, metadata, info)
            build_is_zero(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        EcConcreteLibfunc::Neg(info) => {
        EcConcreteLibfunc::Neg(info) => {
//            build_neg(context, registry, entry, location, helper, metadata, info)
            build_neg(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        EcConcreteLibfunc::PointFromX(info) => {
        EcConcreteLibfunc::PointFromX(info) => {
//            build_point_from_x(context, registry, entry, location, helper, metadata, info)
            build_point_from_x(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        EcConcreteLibfunc::StateAdd(info) => {
        EcConcreteLibfunc::StateAdd(info) => {
//            build_state_add(context, registry, entry, location, helper, metadata, info)
            build_state_add(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        EcConcreteLibfunc::StateAddMul(info) => {
        EcConcreteLibfunc::StateAddMul(info) => {
//            build_state_add_mul(context, registry, entry, location, helper, metadata, info)
            build_state_add_mul(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        EcConcreteLibfunc::StateFinalize(info) => {
        EcConcreteLibfunc::StateFinalize(info) => {
//            build_state_finalize(context, registry, entry, location, helper, metadata, info)
            build_state_finalize(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        EcConcreteLibfunc::StateInit(info) => {
        EcConcreteLibfunc::StateInit(info) => {
//            build_state_init(context, registry, entry, location, helper, metadata, info)
            build_state_init(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        EcConcreteLibfunc::TryNew(info) => {
        EcConcreteLibfunc::TryNew(info) => {
//            build_try_new(context, registry, entry, location, helper, metadata, info)
            build_try_new(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        EcConcreteLibfunc::UnwrapPoint(info) => {
        EcConcreteLibfunc::UnwrapPoint(info) => {
//            build_unwrap_point(context, registry, entry, location, helper, metadata, info)
            build_unwrap_point(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        EcConcreteLibfunc::Zero(info) => {
        EcConcreteLibfunc::Zero(info) => {
//            build_zero(context, registry, entry, location, helper, metadata, info)
            build_zero(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

///// Generate MLIR operations for the `ec_point_is_zero` libfunc.
/// Generate MLIR operations for the `ec_point_is_zero` libfunc.
//pub fn build_is_zero<'ctx, 'this>(
pub fn build_is_zero<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let x = entry.extract_value(
    let x = entry.extract_value(
//        context,
        context,
//        location,
        location,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        IntegerType::new(context, 252).into(),
        IntegerType::new(context, 252).into(),
//        0,
        0,
//    )?;
    )?;
//    let y = entry.extract_value(
    let y = entry.extract_value(
//        context,
        context,
//        location,
        location,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        IntegerType::new(context, 252).into(),
        IntegerType::new(context, 252).into(),
//        1,
        1,
//    )?;
    )?;
//

//    let k0 = entry.const_int(context, location, 0, 252)?;
    let k0 = entry.const_int(context, location, 0, 252)?;
//

//    let x_is_zero =
    let x_is_zero =
//        entry.append_op_result(arith::cmpi(context, CmpiPredicate::Eq, x, k0, location))?;
        entry.append_op_result(arith::cmpi(context, CmpiPredicate::Eq, x, k0, location))?;
//    let y_is_zero =
    let y_is_zero =
//        entry.append_op_result(arith::cmpi(context, CmpiPredicate::Eq, y, k0, location))?;
        entry.append_op_result(arith::cmpi(context, CmpiPredicate::Eq, y, k0, location))?;
//

//    let point_is_zero = entry.append_op_result(arith::andi(x_is_zero, y_is_zero, location))?;
    let point_is_zero = entry.append_op_result(arith::andi(x_is_zero, y_is_zero, location))?;
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        point_is_zero,
        point_is_zero,
//        [0, 1],
        [0, 1],
//        [&[], &[entry.argument(0)?.into()]],
        [&[], &[entry.argument(0)?.into()]],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `ec_neg` libfunc.
/// Generate MLIR operations for the `ec_neg` libfunc.
//pub fn build_neg<'ctx, 'this>(
pub fn build_neg<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let y = entry.extract_value(
    let y = entry.extract_value(
//        context,
        context,
//        location,
        location,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        IntegerType::new(context, 252).into(),
        IntegerType::new(context, 252).into(),
//        1,
        1,
//    )?;
    )?;
//

//    let prime = match metadata.get::<PrimeModuloMeta<Felt>>() {
    let prime = match metadata.get::<PrimeModuloMeta<Felt>>() {
//        Some(x) => x.prime(),
        Some(x) => x.prime(),
//        None => {
        None => {
//            // Since the `EcPoint` type is external, there is no guarantee that
            // Since the `EcPoint` type is external, there is no guarantee that
//            // `PrimeModuloMeta<Felt252>` will be available.
            // `PrimeModuloMeta<Felt252>` will be available.
//            register_prime_modulo_meta(metadata).prime()
            register_prime_modulo_meta(metadata).prime()
//        }
        }
//    };
    };
//

//    let k_prime = entry.const_int(context, location, prime.to_bigint().unwrap(), 252)?;
    let k_prime = entry.const_int(context, location, prime.to_bigint().unwrap(), 252)?;
//

//    let k0 = entry.const_int(context, location, 0, 252)?;
    let k0 = entry.const_int(context, location, 0, 252)?;
//

//    let y_is_zero =
    let y_is_zero =
//        entry.append_op_result(arith::cmpi(context, CmpiPredicate::Eq, y, k0, location))?;
        entry.append_op_result(arith::cmpi(context, CmpiPredicate::Eq, y, k0, location))?;
//

//    let y_neg = entry.append_op_result(arith::subi(k_prime, y, location))?;
    let y_neg = entry.append_op_result(arith::subi(k_prime, y, location))?;
//

//    let y_neg = entry.append_op_result(
    let y_neg = entry.append_op_result(
//        OperationBuilder::new("arith.select", location)
        OperationBuilder::new("arith.select", location)
//            .add_operands(&[y_is_zero, k0, y_neg])
            .add_operands(&[y_is_zero, k0, y_neg])
//            .add_results(&[IntegerType::new(context, 252).into()])
            .add_results(&[IntegerType::new(context, 252).into()])
//            .build()?,
            .build()?,
//    )?;
    )?;
//

//    let result = entry.insert_value(context, location, entry.argument(0)?.into(), y_neg, 1)?;
    let result = entry.insert_value(context, location, entry.argument(0)?.into(), y_neg, 1)?;
//

//    entry.append_operation(helper.br(0, &[result], location));
    entry.append_operation(helper.br(0, &[result], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `ec_point_from_x_nz` libfunc.
/// Generate MLIR operations for the `ec_point_from_x_nz` libfunc.
//pub fn build_point_from_x<'ctx, 'this>(
pub fn build_point_from_x<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let range_check =
    let range_check =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let ec_point_ty = llvm::r#type::r#struct(
    let ec_point_ty = llvm::r#type::r#struct(
//        context,
        context,
//        &[
        &[
//            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
//            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
//        ],
        ],
//        false,
        false,
//    );
    );
//

//    let point_ptr = helper.init_block().alloca1(
    let point_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        ec_point_ty,
        ec_point_ty,
//        Some(get_integer_layout(252).align()),
        Some(get_integer_layout(252).align()),
//    )?;
    )?;
//

//    let point = entry.append_op_result(llvm::undef(ec_point_ty, location))?;
    let point = entry.append_op_result(llvm::undef(ec_point_ty, location))?;
//    let point = entry.insert_value(context, location, point, entry.argument(1)?.into(), 0)?;
    let point = entry.insert_value(context, location, point, entry.argument(1)?.into(), 0)?;
//

//    entry.store(context, location, point_ptr, point, None);
    entry.store(context, location, point_ptr, point, None);
//    let result = metadata
    let result = metadata
//        .get_mut::<RuntimeBindingsMeta>()
        .get_mut::<RuntimeBindingsMeta>()
//        .ok_or(Error::MissingMetadata)?
        .ok_or(Error::MissingMetadata)?
//        .libfunc_ec_point_from_x_nz(context, helper, entry, point_ptr, location)?
        .libfunc_ec_point_from_x_nz(context, helper, entry, point_ptr, location)?
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let point = entry.load(context, location, point_ptr, ec_point_ty, None)?;
    let point = entry.load(context, location, point_ptr, ec_point_ty, None)?;
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        result,
        result,
//        [0, 1],
        [0, 1],
//        [&[range_check, point], &[range_check]],
        [&[range_check, point], &[range_check]],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `ec_state_add` libfunc.
/// Generate MLIR operations for the `ec_state_add` libfunc.
//pub fn build_state_add<'ctx, 'this>(
pub fn build_state_add<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let ec_state_ty = llvm::r#type::r#struct(
    let ec_state_ty = llvm::r#type::r#struct(
//        context,
        context,
//        &[
        &[
//            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
//            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
//            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
//            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
//        ],
        ],
//        false,
        false,
//    );
    );
//

//    let state_ptr = helper.init_block().alloca1(
    let state_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        ec_state_ty,
        ec_state_ty,
//        Some(get_integer_layout(252).align()),
        Some(get_integer_layout(252).align()),
//    )?;
    )?;
//    let point_ptr = helper.init_block().alloca1(
    let point_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        ec_state_ty,
        ec_state_ty,
//        Some(get_integer_layout(252).align()),
        Some(get_integer_layout(252).align()),
//    )?;
    )?;
//

//    entry.store(
    entry.store(
//        context,
        context,
//        location,
        location,
//        state_ptr,
        state_ptr,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        None,
        None,
//    );
    );
//

//    entry.store(
    entry.store(
//        context,
        context,
//        location,
        location,
//        point_ptr,
        point_ptr,
//        entry.argument(1)?.into(),
        entry.argument(1)?.into(),
//        None,
        None,
//    );
    );
//

//    metadata
    metadata
//        .get_mut::<RuntimeBindingsMeta>()
        .get_mut::<RuntimeBindingsMeta>()
//        .ok_or(Error::MissingMetadata)?
        .ok_or(Error::MissingMetadata)?
//        .libfunc_ec_state_add(context, helper, entry, state_ptr, point_ptr, location)?;
        .libfunc_ec_state_add(context, helper, entry, state_ptr, point_ptr, location)?;
//

//    let state = entry.load(context, location, state_ptr, ec_state_ty, None)?;
    let state = entry.load(context, location, state_ptr, ec_state_ty, None)?;
//

//    entry.append_operation(helper.br(0, &[state], location));
    entry.append_operation(helper.br(0, &[state], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `ec_state_add_mul` libfunc.
/// Generate MLIR operations for the `ec_state_add_mul` libfunc.
//pub fn build_state_add_mul<'ctx, 'this>(
pub fn build_state_add_mul<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let ec_op =
    let ec_op =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let felt252_ty = IntegerType::new(context, 252).into();
    let felt252_ty = IntegerType::new(context, 252).into();
//    let ec_state_ty = llvm::r#type::r#struct(
    let ec_state_ty = llvm::r#type::r#struct(
//        context,
        context,
//        &[felt252_ty, felt252_ty, felt252_ty, felt252_ty],
        &[felt252_ty, felt252_ty, felt252_ty, felt252_ty],
//        false,
        false,
//    );
    );
//    let ec_point_ty = llvm::r#type::r#struct(context, &[felt252_ty, felt252_ty], false);
    let ec_point_ty = llvm::r#type::r#struct(context, &[felt252_ty, felt252_ty], false);
//

//    let state_ptr = helper.init_block().alloca1(
    let state_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        ec_state_ty,
        ec_state_ty,
//        Some(get_integer_layout(252).align()),
        Some(get_integer_layout(252).align()),
//    )?;
    )?;
//    let scalar_ptr = helper.init_block().alloca1(
    let scalar_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        felt252_ty,
        felt252_ty,
//        Some(get_integer_layout(252).align()),
        Some(get_integer_layout(252).align()),
//    )?;
    )?;
//    let point_ptr = helper.init_block().alloca1(
    let point_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        ec_point_ty,
        ec_point_ty,
//        Some(get_integer_layout(252).align()),
        Some(get_integer_layout(252).align()),
//    )?;
    )?;
//

//    entry.store(
    entry.store(
//        context,
        context,
//        location,
        location,
//        state_ptr,
        state_ptr,
//        entry.argument(1)?.into(),
        entry.argument(1)?.into(),
//        None,
        None,
//    );
    );
//    entry.store(
    entry.store(
//        context,
        context,
//        location,
        location,
//        scalar_ptr,
        scalar_ptr,
//        entry.argument(2)?.into(),
        entry.argument(2)?.into(),
//        None,
        None,
//    );
    );
//    entry.store(
    entry.store(
//        context,
        context,
//        location,
        location,
//        point_ptr,
        point_ptr,
//        entry.argument(3)?.into(),
        entry.argument(3)?.into(),
//        None,
        None,
//    );
    );
//

//    metadata
    metadata
//        .get_mut::<RuntimeBindingsMeta>()
        .get_mut::<RuntimeBindingsMeta>()
//        .ok_or(Error::MissingMetadata)?
        .ok_or(Error::MissingMetadata)?
//        .libfunc_ec_state_add_mul(
        .libfunc_ec_state_add_mul(
//            context, helper, entry, state_ptr, scalar_ptr, point_ptr, location,
            context, helper, entry, state_ptr, scalar_ptr, point_ptr, location,
//        )?;
        )?;
//

//    let state = entry.load(context, location, state_ptr, ec_state_ty, None)?;
    let state = entry.load(context, location, state_ptr, ec_state_ty, None)?;
//

//    entry.append_operation(helper.br(0, &[ec_op, state], location));
    entry.append_operation(helper.br(0, &[ec_op, state], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `ec_state_try_finalize_nz` libfunc.
/// Generate MLIR operations for the `ec_state_try_finalize_nz` libfunc.
//pub fn build_state_finalize<'ctx, 'this>(
pub fn build_state_finalize<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let felt252_ty = IntegerType::new(context, 252).into();
    let felt252_ty = IntegerType::new(context, 252).into();
//    let ec_state_ty = llvm::r#type::r#struct(
    let ec_state_ty = llvm::r#type::r#struct(
//        context,
        context,
//        &[felt252_ty, felt252_ty, felt252_ty, felt252_ty],
        &[felt252_ty, felt252_ty, felt252_ty, felt252_ty],
//        false,
        false,
//    );
    );
//    let ec_point_ty = llvm::r#type::r#struct(context, &[felt252_ty, felt252_ty], false);
    let ec_point_ty = llvm::r#type::r#struct(context, &[felt252_ty, felt252_ty], false);
//

//    let point_ptr = helper.init_block().alloca1(
    let point_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        ec_point_ty,
        ec_point_ty,
//        Some(get_integer_layout(252).align()),
        Some(get_integer_layout(252).align()),
//    )?;
    )?;
//    let state_ptr = helper.init_block().alloca1(
    let state_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        ec_state_ty,
        ec_state_ty,
//        Some(get_integer_layout(252).align()),
        Some(get_integer_layout(252).align()),
//    )?;
    )?;
//

//    entry.store(
    entry.store(
//        context,
        context,
//        location,
        location,
//        state_ptr,
        state_ptr,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        None,
        None,
//    );
    );
//

//    let is_zero = metadata
    let is_zero = metadata
//        .get_mut::<RuntimeBindingsMeta>()
        .get_mut::<RuntimeBindingsMeta>()
//        .ok_or(Error::MissingMetadata)?
        .ok_or(Error::MissingMetadata)?
//        .libfunc_ec_state_try_finalize_nz(context, helper, entry, point_ptr, state_ptr, location)?
        .libfunc_ec_state_try_finalize_nz(context, helper, entry, point_ptr, state_ptr, location)?
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let point = entry.load(context, location, point_ptr, ec_point_ty, None)?;
    let point = entry.load(context, location, point_ptr, ec_point_ty, None)?;
//

//    entry.append_operation(helper.cond_br(context, is_zero, [0, 1], [&[point], &[]], location));
    entry.append_operation(helper.cond_br(context, is_zero, [0, 1], [&[point], &[]], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `ec_state_init` libfunc.
/// Generate MLIR operations for the `ec_state_init` libfunc.
//pub fn build_state_init<'ctx, 'this>(
pub fn build_state_init<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let ec_state_ty = llvm::r#type::r#struct(
    let ec_state_ty = llvm::r#type::r#struct(
//        context,
        context,
//        &[
        &[
//            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
//            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
//            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
//            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
//        ],
        ],
//        false,
        false,
//    );
    );
//

//    let point = entry
    let point = entry
//        .append_operation(llvm::undef(ec_state_ty, location))
        .append_operation(llvm::undef(ec_state_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let value = BigInt::parse_bytes(
    let value = BigInt::parse_bytes(
//        b"3151312365169595090315724863753927489909436624354740709748557281394568342450",
        b"3151312365169595090315724863753927489909436624354740709748557281394568342450",
//        10,
        10,
//    )
    )
//    .unwrap();
    .unwrap();
//    let x = entry.const_int(context, location, value, 252)?;
    let x = entry.const_int(context, location, value, 252)?;
//

//    let value = BigInt::parse_bytes(
    let value = BigInt::parse_bytes(
//        b"2835232394579952276045648147338966184268723952674536708929458753792035266179",
        b"2835232394579952276045648147338966184268723952674536708929458753792035266179",
//        10,
        10,
//    )
    )
//    .unwrap();
    .unwrap();
//    let y = entry.const_int(context, location, value, 252)?;
    let y = entry.const_int(context, location, value, 252)?;
//

//    let point = entry.insert_value(context, location, point, x, 0)?;
    let point = entry.insert_value(context, location, point, x, 0)?;
//

//    let point = entry.insert_value(context, location, point, y, 1)?;
    let point = entry.insert_value(context, location, point, y, 1)?;
//

//    let point = entry.insert_value(context, location, point, x, 2)?;
    let point = entry.insert_value(context, location, point, x, 2)?;
//

//    let point = entry.insert_value(context, location, point, y, 3)?;
    let point = entry.insert_value(context, location, point, y, 3)?;
//

//    entry.append_operation(helper.br(0, &[point], location));
    entry.append_operation(helper.br(0, &[point], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `ec_point_try_new_nz` libfunc.
/// Generate MLIR operations for the `ec_point_try_new_nz` libfunc.
//pub fn build_try_new<'ctx, 'this>(
pub fn build_try_new<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let ec_point_ty = llvm::r#type::r#struct(
    let ec_point_ty = llvm::r#type::r#struct(
//        context,
        context,
//        &[
        &[
//            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
//            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
//        ],
        ],
//        false,
        false,
//    );
    );
//

//    let point_ptr = helper.init_block().alloca1(
    let point_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        ec_point_ty,
        ec_point_ty,
//        Some(get_integer_layout(252).align()),
        Some(get_integer_layout(252).align()),
//    )?;
    )?;
//

//    let point = entry.append_op_result(llvm::undef(ec_point_ty, location))?;
    let point = entry.append_op_result(llvm::undef(ec_point_ty, location))?;
//    let point = entry.insert_value(context, location, point, entry.argument(0)?.into(), 0)?;
    let point = entry.insert_value(context, location, point, entry.argument(0)?.into(), 0)?;
//    let point = entry.insert_value(context, location, point, entry.argument(1)?.into(), 1)?;
    let point = entry.insert_value(context, location, point, entry.argument(1)?.into(), 1)?;
//

//    entry.store(context, location, point_ptr, point, None);
    entry.store(context, location, point_ptr, point, None);
//

//    let result = metadata
    let result = metadata
//        .get_mut::<RuntimeBindingsMeta>()
        .get_mut::<RuntimeBindingsMeta>()
//        .ok_or(Error::MissingMetadata)?
        .ok_or(Error::MissingMetadata)?
//        .libfunc_ec_point_try_new_nz(context, helper, entry, point_ptr, location)?
        .libfunc_ec_point_try_new_nz(context, helper, entry, point_ptr, location)?
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.cond_br(context, result, [0, 1], [&[point], &[]], location));
    entry.append_operation(helper.cond_br(context, result, [0, 1], [&[point], &[]], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `ec_point_unwrap` libfunc.
/// Generate MLIR operations for the `ec_point_unwrap` libfunc.
//pub fn build_unwrap_point<'ctx, 'this>(
pub fn build_unwrap_point<'ctx, 'this>(
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
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let x = entry.extract_value(
    let x = entry.extract_value(
//        context,
        context,
//        location,
        location,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        registry.build_type(
        registry.build_type(
//            context,
            context,
//            helper,
            helper,
//            registry,
            registry,
//            metadata,
            metadata,
//            &info.branch_signatures()[0].vars[0].ty,
            &info.branch_signatures()[0].vars[0].ty,
//        )?,
        )?,
//        0,
        0,
//    )?;
    )?;
//

//    let y = entry.extract_value(
    let y = entry.extract_value(
//        context,
        context,
//        location,
        location,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        registry.build_type(
        registry.build_type(
//            context,
            context,
//            helper,
            helper,
//            registry,
            registry,
//            metadata,
            metadata,
//            &info.branch_signatures()[0].vars[1].ty,
            &info.branch_signatures()[0].vars[1].ty,
//        )?,
        )?,
//        1,
        1,
//    )?;
    )?;
//

//    entry.append_operation(helper.br(0, &[x, y], location));
    entry.append_operation(helper.br(0, &[x, y], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `ec_point_zero` libfunc.
/// Generate MLIR operations for the `ec_point_zero` libfunc.
//pub fn build_zero<'ctx, 'this>(
pub fn build_zero<'ctx, 'this>(
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
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let ec_point_ty = registry.build_type(
    let ec_point_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.branch_signatures()[0].vars[0].ty,
        &info.branch_signatures()[0].vars[0].ty,
//    )?;
    )?;
//

//    let point = entry
    let point = entry
//        .append_operation(llvm::undef(ec_point_ty, location))
        .append_operation(llvm::undef(ec_point_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let k0 = entry.const_int(context, location, 0, 252)?;
    let k0 = entry.const_int(context, location, 0, 252)?;
//

//    let point = entry.insert_value(context, location, point, k0, 0)?;
    let point = entry.insert_value(context, location, point, k0, 0)?;
//

//    let point = entry.insert_value(context, location, point, k0, 1)?;
    let point = entry.insert_value(context, location, point, k0, 1)?;
//

//    entry.append_operation(helper.br(0, &[point], location));
    entry.append_operation(helper.br(0, &[point], location));
//    Ok(())
    Ok(())
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use crate::{
    use crate::{
//        utils::test::{jit_enum, jit_struct, load_cairo, run_program, run_program_assert_output},
        utils::test::{jit_enum, jit_struct, load_cairo, run_program, run_program_assert_output},
//        values::JitValue,
        values::JitValue,
//    };
    };
//    use cairo_lang_sierra::program::Program;
    use cairo_lang_sierra::program::Program;
//    use lazy_static::lazy_static;
    use lazy_static::lazy_static;
//    use starknet_types_core::felt::Felt;
    use starknet_types_core::felt::Felt;
//    use std::ops::Neg;
    use std::ops::Neg;
//

//    lazy_static! {
    lazy_static! {
//        static ref EC_POINT_IS_ZERO: (String, Program) = load_cairo! {
        static ref EC_POINT_IS_ZERO: (String, Program) = load_cairo! {
//            use core::{ec::{ec_point_is_zero, EcPoint}, zeroable::IsZeroResult};
            use core::{ec::{ec_point_is_zero, EcPoint}, zeroable::IsZeroResult};
//

//            fn run_test(point: EcPoint) -> IsZeroResult<EcPoint> {
            fn run_test(point: EcPoint) -> IsZeroResult<EcPoint> {
//                ec_point_is_zero(point)
                ec_point_is_zero(point)
//            }
            }
//        };
        };
//        static ref EC_NEG: (String, Program) = load_cairo! {
        static ref EC_NEG: (String, Program) = load_cairo! {
//            use core::ec::{ec_neg, EcPoint};
            use core::ec::{ec_neg, EcPoint};
//

//            fn run_test(point: EcPoint) -> EcPoint {
            fn run_test(point: EcPoint) -> EcPoint {
//                ec_neg(point)
                ec_neg(point)
//            }
            }
//        };
        };
//        static ref EC_POINT_FROM_X_NZ: (String, Program) = load_cairo! {
        static ref EC_POINT_FROM_X_NZ: (String, Program) = load_cairo! {
//            use core::ec::{ec_point_from_x_nz, EcPoint};
            use core::ec::{ec_point_from_x_nz, EcPoint};
//            use core::zeroable::NonZero;
            use core::zeroable::NonZero;
//

//            fn run_test(x: felt252) -> Option<NonZero<EcPoint>> {
            fn run_test(x: felt252) -> Option<NonZero<EcPoint>> {
//                ec_point_from_x_nz(x)
                ec_point_from_x_nz(x)
//            }
            }
//        };
        };
//        static ref EC_STATE_ADD: (String, Program) = load_cairo! {
        static ref EC_STATE_ADD: (String, Program) = load_cairo! {
//            use core::ec::{ec_state_add, EcPoint, EcState};
            use core::ec::{ec_state_add, EcPoint, EcState};
//            use core::zeroable::NonZero;
            use core::zeroable::NonZero;
//

//            fn run_test(mut state: EcState, point: NonZero<EcPoint>) -> EcState {
            fn run_test(mut state: EcState, point: NonZero<EcPoint>) -> EcState {
//                ec_state_add(ref state, point);
                ec_state_add(ref state, point);
//                state
                state
//            }
            }
//        };
        };
//        static ref EC_STATE_ADD_MUL: (String, Program) = load_cairo! {
        static ref EC_STATE_ADD_MUL: (String, Program) = load_cairo! {
//            use core::ec::{ec_state_add_mul, EcPoint, EcState};
            use core::ec::{ec_state_add_mul, EcPoint, EcState};
//            use core::zeroable::NonZero;
            use core::zeroable::NonZero;
//

//            fn run_test(mut state: EcState, scalar: felt252, point: NonZero<EcPoint>) -> EcState {
            fn run_test(mut state: EcState, scalar: felt252, point: NonZero<EcPoint>) -> EcState {
//                ec_state_add_mul(ref state, scalar, point);
                ec_state_add_mul(ref state, scalar, point);
//                state
                state
//            }
            }
//        };
        };
//        static ref EC_STATE_FINALIZE: (String, Program) = load_cairo! {
        static ref EC_STATE_FINALIZE: (String, Program) = load_cairo! {
//            use core::ec::{ec_state_try_finalize_nz, EcPoint, EcState};
            use core::ec::{ec_state_try_finalize_nz, EcPoint, EcState};
//            use core::zeroable::NonZero;
            use core::zeroable::NonZero;
//

//            fn run_test(state: EcState) -> Option<NonZero<EcPoint>> {
            fn run_test(state: EcState) -> Option<NonZero<EcPoint>> {
//                ec_state_try_finalize_nz(state)
                ec_state_try_finalize_nz(state)
//            }
            }
//        };
        };
//        static ref EC_STATE_INIT: (String, Program) = load_cairo! {
        static ref EC_STATE_INIT: (String, Program) = load_cairo! {
//            use core::ec::{ec_state_init, EcState};
            use core::ec::{ec_state_init, EcState};
//

//            fn run_test() -> EcState {
            fn run_test() -> EcState {
//                ec_state_init()
                ec_state_init()
//            }
            }
//        };
        };
//        static ref EC_POINT_TRY_NEW_NZ: (String, Program) = load_cairo! {
        static ref EC_POINT_TRY_NEW_NZ: (String, Program) = load_cairo! {
//            use core::ec::{ec_point_try_new_nz, EcPoint};
            use core::ec::{ec_point_try_new_nz, EcPoint};
//            use core::zeroable::NonZero;
            use core::zeroable::NonZero;
//

//            fn run_test(x: felt252, y: felt252) -> Option<NonZero<EcPoint>> {
            fn run_test(x: felt252, y: felt252) -> Option<NonZero<EcPoint>> {
//                ec_point_try_new_nz(x, y)
                ec_point_try_new_nz(x, y)
//            }
            }
//        };
        };
//        static ref EC_POINT_UNWRAP: (String, Program) = load_cairo! {
        static ref EC_POINT_UNWRAP: (String, Program) = load_cairo! {
//            use core::{ec::{ec_point_unwrap, EcPoint}, zeroable::NonZero};
            use core::{ec::{ec_point_unwrap, EcPoint}, zeroable::NonZero};
//

//            fn run_test(point: NonZero<EcPoint>) -> (felt252, felt252) {
            fn run_test(point: NonZero<EcPoint>) -> (felt252, felt252) {
//                ec_point_unwrap(point)
                ec_point_unwrap(point)
//            }
            }
//        };
        };
//        static ref EC_POINT_ZERO: (String, Program) = load_cairo! {
        static ref EC_POINT_ZERO: (String, Program) = load_cairo! {
//            use core::ec::{ec_point_zero, EcPoint};
            use core::ec::{ec_point_zero, EcPoint};
//

//            fn run_test() -> EcPoint {
            fn run_test() -> EcPoint {
//                ec_point_zero()
                ec_point_zero()
//            }
            }
//        };
        };
//    }
    }
//

//    #[test]
    #[test]
//    fn ec_point_is_zero() {
    fn ec_point_is_zero() {
//        let r = |x, y| {
        let r = |x, y| {
//            run_program(&EC_POINT_IS_ZERO, "run_test", &[JitValue::EcPoint(x, y)]).return_value
            run_program(&EC_POINT_IS_ZERO, "run_test", &[JitValue::EcPoint(x, y)]).return_value
//        };
        };
//

//        assert_eq!(r(0.into(), 0.into()), jit_enum!(0, jit_struct!()));
        assert_eq!(r(0.into(), 0.into()), jit_enum!(0, jit_struct!()));
//        assert_eq!(
        assert_eq!(
//            r(0.into(), 1.into()),
            r(0.into(), 1.into()),
//            jit_enum!(1, JitValue::EcPoint(0.into(), 1.into()))
            jit_enum!(1, JitValue::EcPoint(0.into(), 1.into()))
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(1.into(), 0.into()),
            r(1.into(), 0.into()),
//            jit_enum!(1, JitValue::EcPoint(1.into(), 0.into()))
            jit_enum!(1, JitValue::EcPoint(1.into(), 0.into()))
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(1.into(), 1.into()),
            r(1.into(), 1.into()),
//            jit_enum!(1, JitValue::EcPoint(1.into(), 1.into()))
            jit_enum!(1, JitValue::EcPoint(1.into(), 1.into()))
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn ec_neg() {
    fn ec_neg() {
//        let r = |x, y| run_program(&EC_NEG, "run_test", &[JitValue::EcPoint(x, y)]).return_value;
        let r = |x, y| run_program(&EC_NEG, "run_test", &[JitValue::EcPoint(x, y)]).return_value;
//

//        assert_eq!(r(0.into(), 0.into()), JitValue::EcPoint(0.into(), 0.into()));
        assert_eq!(r(0.into(), 0.into()), JitValue::EcPoint(0.into(), 0.into()));
//        assert_eq!(
        assert_eq!(
//            r(0.into(), 1.into()),
            r(0.into(), 1.into()),
//            JitValue::EcPoint(0.into(), Felt::from(-1))
            JitValue::EcPoint(0.into(), Felt::from(-1))
//        );
        );
//        assert_eq!(r(1.into(), 0.into()), JitValue::EcPoint(1.into(), 0.into()));
        assert_eq!(r(1.into(), 0.into()), JitValue::EcPoint(1.into(), 0.into()));
//        assert_eq!(
        assert_eq!(
//            r(1.into(), 1.into()),
            r(1.into(), 1.into()),
//            JitValue::EcPoint(1.into(), Felt::from(-1))
            JitValue::EcPoint(1.into(), Felt::from(-1))
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn ec_point_from_x() {
    fn ec_point_from_x() {
//        let r =
        let r =
//            |x| run_program(&EC_POINT_FROM_X_NZ, "run_test", &[JitValue::Felt252(x)]).return_value;
            |x| run_program(&EC_POINT_FROM_X_NZ, "run_test", &[JitValue::Felt252(x)]).return_value;
//

//        assert_eq!(r(0.into()), jit_enum!(1, jit_struct!()));
        assert_eq!(r(0.into()), jit_enum!(1, jit_struct!()));
//        assert_eq!(r(1234.into()), jit_enum!(0, JitValue::EcPoint(
        assert_eq!(r(1234.into()), jit_enum!(0, JitValue::EcPoint(
//            Felt::from(1234),
            Felt::from(1234),
//            Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
            Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
//        )));
        )));
//    }
    }
//

//    #[test]
    #[test]
//    fn ec_state_add() {
    fn ec_state_add() {
//        run_program_assert_output(&EC_STATE_ADD, "run_test", &[
        run_program_assert_output(&EC_STATE_ADD, "run_test", &[
//            JitValue::EcState(
            JitValue::EcState(
//                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
//                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap(),
                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap(),
//                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
//                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
//            ),
            ),
//            JitValue::EcPoint(
            JitValue::EcPoint(
//                Felt::from_dec_str("1234").unwrap(),
                Felt::from_dec_str("1234").unwrap(),
//                Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
                Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
//            )
            )
//        ],
        ],
//        JitValue::EcState(
        JitValue::EcState(
//            Felt::from_dec_str("763975897824944497806946001227010133599886598340174017198031710397718335159").unwrap(),
            Felt::from_dec_str("763975897824944497806946001227010133599886598340174017198031710397718335159").unwrap(),
//            Felt::from_dec_str("2805180267536471620369715068237762638204710971142209985448115065526708105983").unwrap(),
            Felt::from_dec_str("2805180267536471620369715068237762638204710971142209985448115065526708105983").unwrap(),
//            Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
            Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
//            Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
            Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
//        ));
        ));
//    }
    }
//

//    #[test]
    #[test]
//    fn ec_state_add_mul() {
    fn ec_state_add_mul() {
//        run_program_assert_output(&EC_STATE_ADD_MUL, "run_test", &[
        run_program_assert_output(&EC_STATE_ADD_MUL, "run_test", &[
//            JitValue::EcState(
            JitValue::EcState(
//                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
//                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap(),
                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap(),
//                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
//                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
//            ),
            ),
//            Felt::ONE.into(), // scalar
            Felt::ONE.into(), // scalar
//            JitValue::EcPoint(
            JitValue::EcPoint(
//                Felt::from_dec_str("1234").unwrap(),
                Felt::from_dec_str("1234").unwrap(),
//                Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
                Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
//            )
            )
//        ],
        ],
//            JitValue::EcState(
            JitValue::EcState(
//                Felt::from_dec_str("763975897824944497806946001227010133599886598340174017198031710397718335159").unwrap(),
                Felt::from_dec_str("763975897824944497806946001227010133599886598340174017198031710397718335159").unwrap(),
//                Felt::from_dec_str("2805180267536471620369715068237762638204710971142209985448115065526708105983").unwrap(),
                Felt::from_dec_str("2805180267536471620369715068237762638204710971142209985448115065526708105983").unwrap(),
//                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
//                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
//            )
            )
//        );
        );
//

//        run_program_assert_output(&EC_STATE_ADD_MUL, "run_test", &[
        run_program_assert_output(&EC_STATE_ADD_MUL, "run_test", &[
//            JitValue::EcState(
            JitValue::EcState(
//                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
//                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap(),
                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap(),
//                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
//                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
//            ),
            ),
//            Felt::from(2).into(), // scalar
            Felt::from(2).into(), // scalar
//            JitValue::EcPoint(
            JitValue::EcPoint(
//                Felt::from_dec_str("1234").unwrap(),
                Felt::from_dec_str("1234").unwrap(),
//                Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
                Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
//            )
            )
//        ],
        ],
//            JitValue::EcState(
            JitValue::EcState(
//                Felt::from_dec_str("3016674370847061744386893405108272070153695046160622325692702034987910716850").unwrap(),
                Felt::from_dec_str("3016674370847061744386893405108272070153695046160622325692702034987910716850").unwrap(),
//                Felt::from_dec_str("898133181809473419542838028331350248951548889944002871647069130998202992502").unwrap(),
                Felt::from_dec_str("898133181809473419542838028331350248951548889944002871647069130998202992502").unwrap(),
//                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
//                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
//            )
            )
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn ec_state_finalize() {
    fn ec_state_finalize() {
//        run_program_assert_output(
        run_program_assert_output(
//            &EC_STATE_FINALIZE,
            &EC_STATE_FINALIZE,
//            "run_test",
            "run_test",
//            &[JitValue::EcState(
            &[JitValue::EcState(
//                Felt::from_dec_str(
                Felt::from_dec_str(
//                    "3151312365169595090315724863753927489909436624354740709748557281394568342450",
                    "3151312365169595090315724863753927489909436624354740709748557281394568342450",
//                )
                )
//                .unwrap(),
                .unwrap(),
//                Felt::from_dec_str(
                Felt::from_dec_str(
//                    "2835232394579952276045648147338966184268723952674536708929458753792035266179",
                    "2835232394579952276045648147338966184268723952674536708929458753792035266179",
//                )
                )
//                .unwrap(),
                .unwrap(),
//                Felt::from_dec_str(
                Felt::from_dec_str(
//                    "3151312365169595090315724863753927489909436624354740709748557281394568342450",
                    "3151312365169595090315724863753927489909436624354740709748557281394568342450",
//                )
                )
//                .unwrap(),
                .unwrap(),
//                Felt::from_dec_str(
                Felt::from_dec_str(
//                    "2835232394579952276045648147338966184268723952674536708929458753792035266179",
                    "2835232394579952276045648147338966184268723952674536708929458753792035266179",
//                )
                )
//                .unwrap(),
                .unwrap(),
//            )],
            )],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//        run_program_assert_output(&EC_STATE_FINALIZE, "run_test", &[
        run_program_assert_output(&EC_STATE_FINALIZE, "run_test", &[
//            JitValue::EcState(
            JitValue::EcState(
//                Felt::from_dec_str("763975897824944497806946001227010133599886598340174017198031710397718335159").unwrap(),
                Felt::from_dec_str("763975897824944497806946001227010133599886598340174017198031710397718335159").unwrap(),
//                Felt::from_dec_str("2805180267536471620369715068237762638204710971142209985448115065526708105983").unwrap(),
                Felt::from_dec_str("2805180267536471620369715068237762638204710971142209985448115065526708105983").unwrap(),
//                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
//                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
//            ),
            ),
//        ],
        ],
//            jit_enum!(0, JitValue::EcPoint(
            jit_enum!(0, JitValue::EcPoint(
//                    Felt::from(1234),
                    Felt::from(1234),
//                    Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
                    Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
//                )
                )
//            )
            )
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn ec_state_init() {
    fn ec_state_init() {
//        run_program_assert_output(
        run_program_assert_output(
//            &EC_STATE_INIT,
            &EC_STATE_INIT,
//            "run_test",
            "run_test",
//            &[],
            &[],
//            JitValue::EcState(
            JitValue::EcState(
//                Felt::from_dec_str(
                Felt::from_dec_str(
//                    "3151312365169595090315724863753927489909436624354740709748557281394568342450",
                    "3151312365169595090315724863753927489909436624354740709748557281394568342450",
//                )
                )
//                .unwrap(),
                .unwrap(),
//                Felt::from_dec_str(
                Felt::from_dec_str(
//                    "2835232394579952276045648147338966184268723952674536708929458753792035266179",
                    "2835232394579952276045648147338966184268723952674536708929458753792035266179",
//                )
                )
//                .unwrap(),
                .unwrap(),
//                Felt::from_dec_str(
                Felt::from_dec_str(
//                    "3151312365169595090315724863753927489909436624354740709748557281394568342450",
                    "3151312365169595090315724863753927489909436624354740709748557281394568342450",
//                )
                )
//                .unwrap(),
                .unwrap(),
//                Felt::from_dec_str(
                Felt::from_dec_str(
//                    "2835232394579952276045648147338966184268723952674536708929458753792035266179",
                    "2835232394579952276045648147338966184268723952674536708929458753792035266179",
//                )
                )
//                .unwrap(),
                .unwrap(),
//            ),
            ),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn ec_point_try_new_nz() {
    fn ec_point_try_new_nz() {
//        run_program_assert_output(
        run_program_assert_output(
//            &EC_POINT_TRY_NEW_NZ,
            &EC_POINT_TRY_NEW_NZ,
//            "run_test",
            "run_test",
//            &[
            &[
//                Felt::from_dec_str("0").unwrap().into(),
                Felt::from_dec_str("0").unwrap().into(),
//                Felt::from_dec_str("0").unwrap().into(),
                Felt::from_dec_str("0").unwrap().into(),
//            ],
            ],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            &EC_POINT_TRY_NEW_NZ,
            &EC_POINT_TRY_NEW_NZ,
//            "run_test",
            "run_test",
//            &[
            &[
//                Felt::from_dec_str("1234").unwrap().into(),
                Felt::from_dec_str("1234").unwrap().into(),
//                Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap().into()
                Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap().into()
//            ],
            ],
//                jit_enum!(0, JitValue::EcPoint(
                jit_enum!(0, JitValue::EcPoint(
//                    Felt::from_dec_str("1234").unwrap(),
                    Felt::from_dec_str("1234").unwrap(),
//                    Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
                    Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
//                ))
                ))
//            ,
            ,
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            &EC_POINT_TRY_NEW_NZ,
            &EC_POINT_TRY_NEW_NZ,
//            "run_test",
            "run_test",
//            &[  Felt::from_dec_str("1234").unwrap().into(),
            &[  Felt::from_dec_str("1234").unwrap().into(),
//                Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap().neg().into()
                Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap().neg().into()
//                ],
                ],
//                jit_enum!(0, JitValue::EcPoint(
                jit_enum!(0, JitValue::EcPoint(
//                    Felt::from_dec_str("1234").unwrap(),
                    Felt::from_dec_str("1234").unwrap(),
//                    Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap().neg()
                    Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap().neg()
//                ))
                ))
//                ,
                ,
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn ec_point_unwrap() {
    fn ec_point_unwrap() {
//        fn parse(x: &str) -> Felt {
        fn parse(x: &str) -> Felt {
//            if let Some(x) = x.strip_prefix('-') {
            if let Some(x) = x.strip_prefix('-') {
//                Felt::from_dec_str(x).unwrap().neg()
                Felt::from_dec_str(x).unwrap().neg()
//            } else {
            } else {
//                Felt::from_dec_str(x).unwrap()
                Felt::from_dec_str(x).unwrap()
//            }
            }
//        }
        }
//

//        #[track_caller]
        #[track_caller]
//        fn run(a: &str, b: &str, ea: &str, eb: &str) {
        fn run(a: &str, b: &str, ea: &str, eb: &str) {
//            run_program_assert_output(
            run_program_assert_output(
//                &EC_POINT_UNWRAP,
                &EC_POINT_UNWRAP,
//                "run_test",
                "run_test",
//                &[JitValue::EcPoint(parse(a), parse(b))],
                &[JitValue::EcPoint(parse(a), parse(b))],
//                jit_struct!(parse(ea).into(), parse(eb).into()),
                jit_struct!(parse(ea).into(), parse(eb).into()),
//            );
            );
//        }
        }
//

//        run("0", "0", "0", "0");
        run("0", "0", "0", "0");
//        run("0", "1", "0", "1");
        run("0", "1", "0", "1");
//        run("0", "-1", "0", "-1");
        run("0", "-1", "0", "-1");
//        run("1", "0", "1", "0");
        run("1", "0", "1", "0");
//        run("1", "1", "1", "1");
        run("1", "1", "1", "1");
//        run("1", "-1", "1", "-1");
        run("1", "-1", "1", "-1");
//        run("-1", "0", "-1", "0");
        run("-1", "0", "-1", "0");
//        run("-1", "1", "-1", "1");
        run("-1", "1", "-1", "1");
//        run("-1", "-1", "-1", "-1");
        run("-1", "-1", "-1", "-1");
//    }
    }
//

//    #[test]
    #[test]
//    fn ec_point_zero() {
    fn ec_point_zero() {
//        run_program_assert_output(
        run_program_assert_output(
//            &EC_POINT_ZERO,
            &EC_POINT_ZERO,
//            "run_test",
            "run_test",
//            &[],
            &[],
//            JitValue::EcPoint(
            JitValue::EcPoint(
//                Felt::from_dec_str("0").unwrap(),
                Felt::from_dec_str("0").unwrap(),
//                Felt::from_dec_str("0").unwrap().neg(),
                Felt::from_dec_str("0").unwrap().neg(),
//            ),
            ),
//        );
        );
//    }
    }
//}
}
