////! # Poseidon hashing libfuncs
//! # Poseidon hashing libfuncs
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
//        poseidon::PoseidonConcreteLibfunc,
        poseidon::PoseidonConcreteLibfunc,
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
//    selector: &PoseidonConcreteLibfunc,
    selector: &PoseidonConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        PoseidonConcreteLibfunc::HadesPermutation(info) => {
        PoseidonConcreteLibfunc::HadesPermutation(info) => {
//            build_hades_permutation(context, registry, entry, location, helper, metadata, info)
            build_hades_permutation(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

//pub fn build_hades_permutation<'ctx>(
pub fn build_hades_permutation<'ctx>(
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

//    let poseidon_builtin =
    let poseidon_builtin =
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

//    let op0 = entry.argument(1)?.into();
    let op0 = entry.argument(1)?.into();
//    let op1 = entry.argument(2)?.into();
    let op1 = entry.argument(2)?.into();
//    let op2 = entry.argument(3)?.into();
    let op2 = entry.argument(3)?.into();
//

//    // We must extend to i256 because bswap must be an even number of bytes.
    // We must extend to i256 because bswap must be an even number of bytes.
//

//    let op0_ptr =
    let op0_ptr =
//        helper
        helper
//            .init_block()
            .init_block()
//            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;
            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;
//    let op1_ptr =
    let op1_ptr =
//        helper
        helper
//            .init_block()
            .init_block()
//            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;
            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;
//    let op2_ptr =
    let op2_ptr =
//        helper
        helper
//            .init_block()
            .init_block()
//            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;
            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;
//

//    let op0_i256 = entry
    let op0_i256 = entry
//        .append_operation(ods::arith::extui(context, i256_ty, op0, location).into())
        .append_operation(ods::arith::extui(context, i256_ty, op0, location).into())
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let op1_i256 = entry
    let op1_i256 = entry
//        .append_operation(ods::arith::extui(context, i256_ty, op1, location).into())
        .append_operation(ods::arith::extui(context, i256_ty, op1, location).into())
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let op2_i256 = entry
    let op2_i256 = entry
//        .append_operation(ods::arith::extui(context, i256_ty, op2, location).into())
        .append_operation(ods::arith::extui(context, i256_ty, op2, location).into())
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let op0_be = entry
    let op0_be = entry
//        .append_operation(llvm::intr_bswap(op0_i256, i256_ty, location))
        .append_operation(llvm::intr_bswap(op0_i256, i256_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let op1_be = entry
    let op1_be = entry
//        .append_operation(llvm::intr_bswap(op1_i256, i256_ty, location))
        .append_operation(llvm::intr_bswap(op1_i256, i256_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let op2_be = entry
    let op2_be = entry
//        .append_operation(llvm::intr_bswap(op2_i256, i256_ty, location))
        .append_operation(llvm::intr_bswap(op2_i256, i256_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        op0_be,
        op0_be,
//        op0_ptr,
        op0_ptr,
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
//        op1_be,
        op1_be,
//        op1_ptr,
        op1_ptr,
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
//        op2_be,
        op2_be,
//        op2_ptr,
        op2_ptr,
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
//        .libfunc_hades_permutation(context, helper, entry, op0_ptr, op1_ptr, op2_ptr, location)?;
        .libfunc_hades_permutation(context, helper, entry, op0_ptr, op1_ptr, op2_ptr, location)?;
//

//    let op0_be = entry
    let op0_be = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            op0_ptr,
            op0_ptr,
//            i256_ty,
            i256_ty,
//            location,
            location,
//            LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            LoadStoreOptions::default().align(Some(IntegerAttribute::new(
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                layout_i256.align().try_into()?,
                layout_i256.align().try_into()?,
//            ))),
            ))),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let op1_be = entry
    let op1_be = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            op1_ptr,
            op1_ptr,
//            i256_ty,
            i256_ty,
//            location,
            location,
//            LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            LoadStoreOptions::default().align(Some(IntegerAttribute::new(
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                layout_i256.align().try_into()?,
                layout_i256.align().try_into()?,
//            ))),
            ))),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let op2_be = entry
    let op2_be = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            op2_ptr,
            op2_ptr,
//            i256_ty,
            i256_ty,
//            location,
            location,
//            LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            LoadStoreOptions::default().align(Some(IntegerAttribute::new(
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                layout_i256.align().try_into()?,
                layout_i256.align().try_into()?,
//            ))),
            ))),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let op0_i256 = entry
    let op0_i256 = entry
//        .append_operation(llvm::intr_bswap(op0_be, i256_ty, location))
        .append_operation(llvm::intr_bswap(op0_be, i256_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let op1_i256 = entry
    let op1_i256 = entry
//        .append_operation(llvm::intr_bswap(op1_be, i256_ty, location))
        .append_operation(llvm::intr_bswap(op1_be, i256_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let op2_i256 = entry
    let op2_i256 = entry
//        .append_operation(llvm::intr_bswap(op2_be, i256_ty, location))
        .append_operation(llvm::intr_bswap(op2_be, i256_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let op0 = entry
    let op0 = entry
//        .append_operation(arith::trunci(op0_i256, felt252_ty, location))
        .append_operation(arith::trunci(op0_i256, felt252_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let op1 = entry
    let op1 = entry
//        .append_operation(arith::trunci(op1_i256, felt252_ty, location))
        .append_operation(arith::trunci(op1_i256, felt252_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let op2 = entry
    let op2 = entry
//        .append_operation(arith::trunci(op2_i256, felt252_ty, location))
        .append_operation(arith::trunci(op2_i256, felt252_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.br(0, &[poseidon_builtin, op0, op1, op2], location));
    entry.append_operation(helper.br(0, &[poseidon_builtin, op0, op1, op2], location));
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
//    use crate::utils::test::{jit_struct, load_cairo, run_program_assert_output};
    use crate::utils::test::{jit_struct, load_cairo, run_program_assert_output};
//

//    use starknet_types_core::felt::Felt;
    use starknet_types_core::felt::Felt;
//

//    #[test]
    #[test]
//    fn run_hades_permutation() {
    fn run_hades_permutation() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use core::poseidon::hades_permutation;
            use core::poseidon::hades_permutation;
//

//            fn run_test(a: felt252, b: felt252, c: felt252) -> (felt252, felt252, felt252) {
            fn run_test(a: felt252, b: felt252, c: felt252) -> (felt252, felt252, felt252) {
//                hades_permutation(a, b, c)
                hades_permutation(a, b, c)
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
//            &[
            &[
//                Felt::from(2).into(),
                Felt::from(2).into(),
//                Felt::from(4).into(),
                Felt::from(4).into(),
//                Felt::from(4).into(),
                Felt::from(4).into(),
//            ],
            ],
//            jit_struct!(
            jit_struct!(
//                Felt::from_dec_str(
                Felt::from_dec_str(
//                    "1627044480024625333712068603977073585655327747658231320998869768849911913066"
                    "1627044480024625333712068603977073585655327747658231320998869768849911913066"
//                )
                )
//                .unwrap()
                .unwrap()
//                .into(),
                .into(),
//                Felt::from_dec_str(
                Felt::from_dec_str(
//                    "2368195581807763724810563135784547417602556730014791322540110420941926079965"
                    "2368195581807763724810563135784547417602556730014791322540110420941926079965"
//                )
                )
//                .unwrap()
                .unwrap()
//                .into(),
                .into(),
//                Felt::from_dec_str(
                Felt::from_dec_str(
//                    "2381325839211954898363395375151559373051496038592329842107874845056395867189"
                    "2381325839211954898363395375151559373051496038592329842107874845056395867189"
//                )
                )
//                .unwrap()
                .unwrap()
//                .into(),
                .into(),
//            ),
            ),
//        );
        );
//    }
    }
//}
}
