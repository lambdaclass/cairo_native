//use crate::{
use crate::{
//    block_ext::BlockExt,
    block_ext::BlockExt,
//    error::Result,
    error::Result,
//    libfuncs::LibfuncHelper,
    libfuncs::LibfuncHelper,
//    metadata::MetadataStorage,
    metadata::MetadataStorage,
//    starknet::handler::StarknetSyscallHandlerCallbacks,
    starknet::handler::StarknetSyscallHandlerCallbacks,
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
//        starknet::secp256::{Secp256ConcreteLibfunc, Secp256OpConcreteLibfunc},
        starknet::secp256::{Secp256ConcreteLibfunc, Secp256OpConcreteLibfunc},
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
//    },
    },
//    ir::{
    ir::{
//        attribute::{
        attribute::{
//            DenseI32ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute, TypeAttribute,
            DenseI32ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute, TypeAttribute,
//        },
        },
//        operation::OperationBuilder,
        operation::OperationBuilder,
//        r#type::IntegerType,
        r#type::IntegerType,
//        Block, Identifier, Location,
        Block, Identifier, Location,
//    },
    },
//    Context,
    Context,
//};
};
//use std::alloc::Layout;
use std::alloc::Layout;
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
//    selector: &Secp256ConcreteLibfunc,
    selector: &Secp256ConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        Secp256ConcreteLibfunc::K1(selector) => match selector {
        Secp256ConcreteLibfunc::K1(selector) => match selector {
//            Secp256OpConcreteLibfunc::New(info) => {
            Secp256OpConcreteLibfunc::New(info) => {
//                build_k1_new(context, registry, entry, location, helper, metadata, info)
                build_k1_new(context, registry, entry, location, helper, metadata, info)
//            }
            }
//            Secp256OpConcreteLibfunc::Add(info) => {
            Secp256OpConcreteLibfunc::Add(info) => {
//                build_k1_add(context, registry, entry, location, helper, metadata, info)
                build_k1_add(context, registry, entry, location, helper, metadata, info)
//            }
            }
//            Secp256OpConcreteLibfunc::Mul(info) => {
            Secp256OpConcreteLibfunc::Mul(info) => {
//                build_k1_mul(context, registry, entry, location, helper, metadata, info)
                build_k1_mul(context, registry, entry, location, helper, metadata, info)
//            }
            }
//            Secp256OpConcreteLibfunc::GetPointFromX(info) => build_k1_get_point_from_x(
            Secp256OpConcreteLibfunc::GetPointFromX(info) => build_k1_get_point_from_x(
//                context, registry, entry, location, helper, metadata, info,
                context, registry, entry, location, helper, metadata, info,
//            ),
            ),
//            Secp256OpConcreteLibfunc::GetXy(info) => {
            Secp256OpConcreteLibfunc::GetXy(info) => {
//                build_k1_get_xy(context, registry, entry, location, helper, metadata, info)
                build_k1_get_xy(context, registry, entry, location, helper, metadata, info)
//            }
            }
//        },
        },
//        Secp256ConcreteLibfunc::R1(selector) => match selector {
        Secp256ConcreteLibfunc::R1(selector) => match selector {
//            Secp256OpConcreteLibfunc::New(info) => {
            Secp256OpConcreteLibfunc::New(info) => {
//                build_r1_new(context, registry, entry, location, helper, metadata, info)
                build_r1_new(context, registry, entry, location, helper, metadata, info)
//            }
            }
//            Secp256OpConcreteLibfunc::Add(info) => {
            Secp256OpConcreteLibfunc::Add(info) => {
//                build_r1_add(context, registry, entry, location, helper, metadata, info)
                build_r1_add(context, registry, entry, location, helper, metadata, info)
//            }
            }
//            Secp256OpConcreteLibfunc::Mul(info) => {
            Secp256OpConcreteLibfunc::Mul(info) => {
//                build_r1_mul(context, registry, entry, location, helper, metadata, info)
                build_r1_mul(context, registry, entry, location, helper, metadata, info)
//            }
            }
//            Secp256OpConcreteLibfunc::GetPointFromX(info) => build_r1_get_point_from_x(
            Secp256OpConcreteLibfunc::GetPointFromX(info) => build_r1_get_point_from_x(
//                context, registry, entry, location, helper, metadata, info,
                context, registry, entry, location, helper, metadata, info,
//            ),
            ),
//            Secp256OpConcreteLibfunc::GetXy(info) => {
            Secp256OpConcreteLibfunc::GetXy(info) => {
//                build_r1_get_xy(context, registry, entry, location, helper, metadata, info)
                build_r1_get_xy(context, registry, entry, location, helper, metadata, info)
//            }
            }
//        },
        },
//    }
    }
//}
}
//

//pub fn build_k1_new<'ctx, 'this>(
pub fn build_k1_new<'ctx, 'this>(
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
//    // Extract self pointer.
    // Extract self pointer.
//    let ptr = entry
    let ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space for the return value.
    // Allocate space for the return value.
//    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
//        crate::types::r#enum::get_type_for_variants(
        crate::types::r#enum::get_type_for_variants(
//            context,
            context,
//            helper,
            helper,
//            registry,
            registry,
//            metadata,
            metadata,
//            &[
            &[
//                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[0].vars[2].ty.clone(),
//                info.branch_signatures()[1].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
//            ],
            ],
//        )?;
        )?;
//

//    let k1 = helper
    let k1 = helper
//        .init_block()
        .init_block()
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_ptr = helper
    let result_ptr = helper
//        .init_block()
        .init_block()
//        .append_operation(
        .append_operation(
//            OperationBuilder::new("llvm.alloca", location)
            OperationBuilder::new("llvm.alloca", location)
//                .add_attributes(&[
                .add_attributes(&[
//                    (
                    (
//                        Identifier::new(context, "alignment"),
                        Identifier::new(context, "alignment"),
//                        IntegerAttribute::new(
                        IntegerAttribute::new(
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            result_layout.align().try_into()?,
                            result_layout.align().try_into()?,
//                        )
                        )
//                        .into(),
                        .into(),
//                    ),
                    ),
//                    (
                    (
//                        Identifier::new(context, "elem_type"),
                        Identifier::new(context, "elem_type"),
//                        TypeAttribute::new(llvm::r#type::r#struct(
                        TypeAttribute::new(llvm::r#type::r#struct(
//                            context,
                            context,
//                            &[
                            &[
//                                result_tag_ty,
                                result_tag_ty,
//                                llvm::r#type::array(
                                llvm::r#type::array(
//                                    IntegerType::new(context, 8).into(),
                                    IntegerType::new(context, 8).into(),
//                                    (result_layout.size() - 1).try_into()?,
                                    (result_layout.size() - 1).try_into()?,
//                                ),
                                ),
//                            ],
                            ],
//                            false,
                            false,
//                        ))
                        ))
//                        .into(),
                        .into(),
//                    ),
                    ),
//                ])
                ])
//                .add_operands(&[k1])
                .add_operands(&[k1])
//                .add_results(&[llvm::r#type::pointer(context, 0)])
                .add_results(&[llvm::r#type::pointer(context, 0)])
//                .build()?,
                .build()?,
//        )
        )
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space and write the current gas.
    // Allocate space and write the current gas.
//    let gas_builtin_ptr = helper.init_block().alloca1(
    let gas_builtin_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        IntegerType::new(context, 128).into(),
        IntegerType::new(context, 128).into(),
//        Some(get_integer_layout(128).align()),
        Some(get_integer_layout(128).align()),
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        gas_builtin_ptr,
        gas_builtin_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    let i128_ty = IntegerType::new(context, 128).into();
    let i128_ty = IntegerType::new(context, 128).into();
//

//    // Allocate `x` argument and write the value.
    // Allocate `x` argument and write the value.
//    let x_arg_ptr = helper.init_block().alloca1(
    let x_arg_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
        llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
//        None,
        None,
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(2)?.into(),
        entry.argument(2)?.into(),
//        x_arg_ptr,
        x_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Allocate `y` argument and write the value.
    // Allocate `y` argument and write the value.
//    let y_arg_ptr = helper.init_block().alloca1(
    let y_arg_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
        llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
//        None,
        None,
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(3)?.into(),
        entry.argument(3)?.into(),
//        y_arg_ptr,
        y_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::get_element_ptr(
        .append_operation(llvm::get_element_ptr(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            DenseI32ArrayAttribute::new(
            DenseI32ArrayAttribute::new(
//                context,
                context,
//                &[StarknetSyscallHandlerCallbacks::<()>::SECP256K1_NEW.try_into()?],
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256K1_NEW.try_into()?],
//            ),
            ),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            fn_ptr,
            fn_ptr,
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(
    entry.append_operation(
//        OperationBuilder::new("llvm.call", location)
        OperationBuilder::new("llvm.call", location)
//            .add_operands(&[
            .add_operands(&[
//                fn_ptr,
                fn_ptr,
//                result_ptr,
                result_ptr,
//                ptr,
                ptr,
//                gas_builtin_ptr,
                gas_builtin_ptr,
//                x_arg_ptr,
                x_arg_ptr,
//                y_arg_ptr,
                y_arg_ptr,
//            ])
            ])
//            .build()?,
            .build()?,
//    );
    );
//

//    let result = entry
    let result = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            result_ptr,
            result_ptr,
//            llvm::r#type::r#struct(
            llvm::r#type::r#struct(
//                context,
                context,
//                &[
                &[
//                    result_tag_ty,
                    result_tag_ty,
//                    llvm::r#type::array(
                    llvm::r#type::array(
//                        IntegerType::new(context, 8).into(),
                        IntegerType::new(context, 8).into(),
//                        (result_layout.size() - 1).try_into()?,
                        (result_layout.size() - 1).try_into()?,
//                    ),
                    ),
//                ],
                ],
//                false,
                false,
//            ),
            ),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_tag = entry
    let result_tag = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            result,
            result,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            IntegerType::new(context, 1).into(),
            IntegerType::new(context, 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let payload_ok = {
    let payload_ok = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry.load(
        entry.load(
//            context,
            context,
//            location,
            location,
//            ptr,
            ptr,
//            variant_tys[0].0,
            variant_tys[0].0,
//            Some(variant_tys[0].1.align()),
            Some(variant_tys[0].1.align()),
//        )?
        )?
//    };
    };
//    let payload_err = {
    let payload_err = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry
        entry
//            .append_operation(llvm::load(
            .append_operation(llvm::load(
//                context,
                context,
//                ptr,
                ptr,
//                variant_tys[1].0,
                variant_tys[1].0,
//                location,
                location,
//                LoadStoreOptions::default(),
                LoadStoreOptions::default(),
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into()
            .into()
//    };
    };
//

//    let remaining_gas = entry
    let remaining_gas = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            gas_builtin_ptr,
            gas_builtin_ptr,
//            IntegerType::new(context, 128).into(),
            IntegerType::new(context, 128).into(),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        result_tag,
        result_tag,
//        [1, 0],
        [1, 0],
//        [
        [
//            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
//            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
//        ],
        ],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

//pub fn build_k1_add<'ctx, 'this>(
pub fn build_k1_add<'ctx, 'this>(
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
//    // Extract self pointer.
    // Extract self pointer.
//    let ptr = entry
    let ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space for the return value.
    // Allocate space for the return value.
//    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
//        crate::types::r#enum::get_type_for_variants(
        crate::types::r#enum::get_type_for_variants(
//            context,
            context,
//            helper,
            helper,
//            registry,
            registry,
//            metadata,
            metadata,
//            &[
            &[
//                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[0].vars[2].ty.clone(),
//                info.branch_signatures()[1].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
//            ],
            ],
//        )?;
        )?;
//

//    let k1 = helper
    let k1 = helper
//        .init_block()
        .init_block()
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_ptr = helper
    let result_ptr = helper
//        .init_block()
        .init_block()
//        .append_operation(
        .append_operation(
//            OperationBuilder::new("llvm.alloca", location)
            OperationBuilder::new("llvm.alloca", location)
//                .add_attributes(&[
                .add_attributes(&[
//                    (
                    (
//                        Identifier::new(context, "alignment"),
                        Identifier::new(context, "alignment"),
//                        IntegerAttribute::new(
                        IntegerAttribute::new(
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            result_layout.align().try_into()?,
                            result_layout.align().try_into()?,
//                        )
                        )
//                        .into(),
                        .into(),
//                    ),
                    ),
//                    (
                    (
//                        Identifier::new(context, "elem_type"),
                        Identifier::new(context, "elem_type"),
//                        TypeAttribute::new(llvm::r#type::r#struct(
                        TypeAttribute::new(llvm::r#type::r#struct(
//                            context,
                            context,
//                            &[
                            &[
//                                result_tag_ty,
                                result_tag_ty,
//                                llvm::r#type::array(
                                llvm::r#type::array(
//                                    IntegerType::new(context, 8).into(),
                                    IntegerType::new(context, 8).into(),
//                                    (result_layout.size() - 1).try_into()?,
                                    (result_layout.size() - 1).try_into()?,
//                                ),
                                ),
//                            ],
                            ],
//                            false,
                            false,
//                        ))
                        ))
//                        .into(),
                        .into(),
//                    ),
                    ),
//                ])
                ])
//                .add_operands(&[k1])
                .add_operands(&[k1])
//                .add_results(&[llvm::r#type::pointer(context, 0)])
                .add_results(&[llvm::r#type::pointer(context, 0)])
//                .build()?,
                .build()?,
//        )
        )
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space and write the current gas.
    // Allocate space and write the current gas.
//    let gas_builtin_ptr = helper.init_block().alloca1(
    let gas_builtin_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        IntegerType::new(context, 128).into(),
        IntegerType::new(context, 128).into(),
//        Some(get_integer_layout(128).align()),
        Some(get_integer_layout(128).align()),
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        gas_builtin_ptr,
        gas_builtin_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Allocate `p0` argument and write the value.
    // Allocate `p0` argument and write the value.
//    let p0_arg_ptr = helper.init_block().alloca1(
    let p0_arg_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        llvm::r#type::r#struct(
        llvm::r#type::r#struct(
//            context,
            context,
//            &[
            &[
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//            ],
            ],
//            false,
            false,
//        ),
        ),
//        None,
        None,
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(2)?.into(),
        entry.argument(2)?.into(),
//        p0_arg_ptr,
        p0_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Allocate `p1` argument and write the value.
    // Allocate `p1` argument and write the value.
//    let p1_arg_ptr = helper.init_block().alloca1(
    let p1_arg_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        llvm::r#type::r#struct(
        llvm::r#type::r#struct(
//            context,
            context,
//            &[
            &[
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//            ],
            ],
//            false,
            false,
//        ),
        ),
//        None,
        None,
//    )?;
    )?;
//

//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(3)?.into(),
        entry.argument(3)?.into(),
//        p1_arg_ptr,
        p1_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::get_element_ptr(
        .append_operation(llvm::get_element_ptr(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            DenseI32ArrayAttribute::new(
            DenseI32ArrayAttribute::new(
//                context,
                context,
//                &[StarknetSyscallHandlerCallbacks::<()>::SECP256K1_ADD.try_into()?],
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256K1_ADD.try_into()?],
//            ),
            ),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            fn_ptr,
            fn_ptr,
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(
    entry.append_operation(
//        OperationBuilder::new("llvm.call", location)
        OperationBuilder::new("llvm.call", location)
//            .add_operands(&[
            .add_operands(&[
//                fn_ptr,
                fn_ptr,
//                result_ptr,
                result_ptr,
//                ptr,
                ptr,
//                gas_builtin_ptr,
                gas_builtin_ptr,
//                p0_arg_ptr,
                p0_arg_ptr,
//                p1_arg_ptr,
                p1_arg_ptr,
//            ])
            ])
//            .build()?,
            .build()?,
//    );
    );
//

//    let result = entry
    let result = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            result_ptr,
            result_ptr,
//            llvm::r#type::r#struct(
            llvm::r#type::r#struct(
//                context,
                context,
//                &[
                &[
//                    result_tag_ty,
                    result_tag_ty,
//                    llvm::r#type::array(
                    llvm::r#type::array(
//                        IntegerType::new(context, 8).into(),
                        IntegerType::new(context, 8).into(),
//                        (result_layout.size() - 1).try_into()?,
                        (result_layout.size() - 1).try_into()?,
//                    ),
                    ),
//                ],
                ],
//                false,
                false,
//            ),
            ),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_tag = entry
    let result_tag = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            result,
            result,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            IntegerType::new(context, 1).into(),
            IntegerType::new(context, 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let payload_ok = {
    let payload_ok = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry
        entry
//            .append_operation(llvm::load(
            .append_operation(llvm::load(
//                context,
                context,
//                ptr,
                ptr,
//                variant_tys[0].0,
                variant_tys[0].0,
//                location,
                location,
//                LoadStoreOptions::default(),
                LoadStoreOptions::default(),
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into()
            .into()
//    };
    };
//    let payload_err = {
    let payload_err = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry
        entry
//            .append_operation(llvm::load(
            .append_operation(llvm::load(
//                context,
                context,
//                ptr,
                ptr,
//                variant_tys[1].0,
                variant_tys[1].0,
//                location,
                location,
//                LoadStoreOptions::default(),
                LoadStoreOptions::default(),
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into()
            .into()
//    };
    };
//

//    let remaining_gas = entry
    let remaining_gas = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            gas_builtin_ptr,
            gas_builtin_ptr,
//            IntegerType::new(context, 128).into(),
            IntegerType::new(context, 128).into(),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        result_tag,
        result_tag,
//        [1, 0],
        [1, 0],
//        [
        [
//            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
//            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
//        ],
        ],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

//pub fn build_k1_mul<'ctx, 'this>(
pub fn build_k1_mul<'ctx, 'this>(
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
//    // Extract self pointer.
    // Extract self pointer.
//    let ptr = entry
    let ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space for the return value.
    // Allocate space for the return value.
//    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
//        crate::types::r#enum::get_type_for_variants(
        crate::types::r#enum::get_type_for_variants(
//            context,
            context,
//            helper,
            helper,
//            registry,
            registry,
//            metadata,
            metadata,
//            &[
            &[
//                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[0].vars[2].ty.clone(),
//                info.branch_signatures()[1].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
//            ],
            ],
//        )?;
        )?;
//

//    let k1 = helper
    let k1 = helper
//        .init_block()
        .init_block()
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_ptr = helper
    let result_ptr = helper
//        .init_block()
        .init_block()
//        .append_operation(
        .append_operation(
//            OperationBuilder::new("llvm.alloca", location)
            OperationBuilder::new("llvm.alloca", location)
//                .add_attributes(&[
                .add_attributes(&[
//                    (
                    (
//                        Identifier::new(context, "alignment"),
                        Identifier::new(context, "alignment"),
//                        IntegerAttribute::new(
                        IntegerAttribute::new(
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            result_layout.align().try_into()?,
                            result_layout.align().try_into()?,
//                        )
                        )
//                        .into(),
                        .into(),
//                    ),
                    ),
//                    (
                    (
//                        Identifier::new(context, "elem_type"),
                        Identifier::new(context, "elem_type"),
//                        TypeAttribute::new(llvm::r#type::r#struct(
                        TypeAttribute::new(llvm::r#type::r#struct(
//                            context,
                            context,
//                            &[
                            &[
//                                result_tag_ty,
                                result_tag_ty,
//                                llvm::r#type::array(
                                llvm::r#type::array(
//                                    IntegerType::new(context, 8).into(),
                                    IntegerType::new(context, 8).into(),
//                                    (result_layout.size() - 1).try_into()?,
                                    (result_layout.size() - 1).try_into()?,
//                                ),
                                ),
//                            ],
                            ],
//                            false,
                            false,
//                        ))
                        ))
//                        .into(),
                        .into(),
//                    ),
                    ),
//                ])
                ])
//                .add_operands(&[k1])
                .add_operands(&[k1])
//                .add_results(&[llvm::r#type::pointer(context, 0)])
                .add_results(&[llvm::r#type::pointer(context, 0)])
//                .build()?,
                .build()?,
//        )
        )
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space and write the current gas.
    // Allocate space and write the current gas.
//    let gas_builtin_ptr = helper.init_block().alloca1(
    let gas_builtin_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        IntegerType::new(context, 128).into(),
        IntegerType::new(context, 128).into(),
//        Some(get_integer_layout(128).align()),
        Some(get_integer_layout(128).align()),
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        gas_builtin_ptr,
        gas_builtin_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Allocate `p` argument and write the value.
    // Allocate `p` argument and write the value.
//    let p_arg_ptr = helper.init_block().alloca1(
    let p_arg_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        llvm::r#type::r#struct(
        llvm::r#type::r#struct(
//            context,
            context,
//            &[
            &[
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//            ],
            ],
//            false,
            false,
//        ),
        ),
//        None,
        None,
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(2)?.into(),
        entry.argument(2)?.into(),
//        p_arg_ptr,
        p_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Allocate `scalar` argument and write the value.
    // Allocate `scalar` argument and write the value.
//    let scalar_arg_ptr = helper.init_block().alloca1(
    let scalar_arg_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        llvm::r#type::r#struct(
        llvm::r#type::r#struct(
//            context,
            context,
//            &[
            &[
//                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
//                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
//            ],
            ],
//            false,
            false,
//        ),
        ),
//        None,
        None,
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(3)?.into(),
        entry.argument(3)?.into(),
//        scalar_arg_ptr,
        scalar_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::get_element_ptr(
        .append_operation(llvm::get_element_ptr(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            DenseI32ArrayAttribute::new(
            DenseI32ArrayAttribute::new(
//                context,
                context,
//                &[StarknetSyscallHandlerCallbacks::<()>::SECP256K1_MUL.try_into()?],
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256K1_MUL.try_into()?],
//            ),
            ),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            fn_ptr,
            fn_ptr,
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(
    entry.append_operation(
//        OperationBuilder::new("llvm.call", location)
        OperationBuilder::new("llvm.call", location)
//            .add_operands(&[
            .add_operands(&[
//                fn_ptr,
                fn_ptr,
//                result_ptr,
                result_ptr,
//                ptr,
                ptr,
//                gas_builtin_ptr,
                gas_builtin_ptr,
//                p_arg_ptr,
                p_arg_ptr,
//                scalar_arg_ptr,
                scalar_arg_ptr,
//            ])
            ])
//            .build()?,
            .build()?,
//    );
    );
//

//    let result = entry
    let result = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            result_ptr,
            result_ptr,
//            llvm::r#type::r#struct(
            llvm::r#type::r#struct(
//                context,
                context,
//                &[
                &[
//                    result_tag_ty,
                    result_tag_ty,
//                    llvm::r#type::array(
                    llvm::r#type::array(
//                        IntegerType::new(context, 8).into(),
                        IntegerType::new(context, 8).into(),
//                        (result_layout.size() - 1).try_into()?,
                        (result_layout.size() - 1).try_into()?,
//                    ),
                    ),
//                ],
                ],
//                false,
                false,
//            ),
            ),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_tag = entry
    let result_tag = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            result,
            result,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            IntegerType::new(context, 1).into(),
            IntegerType::new(context, 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let payload_ok = {
    let payload_ok = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry
        entry
//            .append_operation(llvm::load(
            .append_operation(llvm::load(
//                context,
                context,
//                ptr,
                ptr,
//                variant_tys[0].0,
                variant_tys[0].0,
//                location,
                location,
//                LoadStoreOptions::default(),
                LoadStoreOptions::default(),
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into()
            .into()
//    };
    };
//    let payload_err = {
    let payload_err = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry
        entry
//            .append_operation(llvm::load(
            .append_operation(llvm::load(
//                context,
                context,
//                ptr,
                ptr,
//                variant_tys[1].0,
                variant_tys[1].0,
//                location,
                location,
//                LoadStoreOptions::default(),
                LoadStoreOptions::default(),
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into()
            .into()
//    };
    };
//

//    let remaining_gas = entry
    let remaining_gas = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            gas_builtin_ptr,
            gas_builtin_ptr,
//            IntegerType::new(context, 128).into(),
            IntegerType::new(context, 128).into(),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        result_tag,
        result_tag,
//        [1, 0],
        [1, 0],
//        [
        [
//            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
//            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
//        ],
        ],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

//pub fn build_k1_get_point_from_x<'ctx, 'this>(
pub fn build_k1_get_point_from_x<'ctx, 'this>(
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
//    // Extract self pointer.
    // Extract self pointer.
//    let ptr = entry
    let ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space for the return value.
    // Allocate space for the return value.
//    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
//        crate::types::r#enum::get_type_for_variants(
        crate::types::r#enum::get_type_for_variants(
//            context,
            context,
//            helper,
            helper,
//            registry,
            registry,
//            metadata,
            metadata,
//            &[
            &[
//                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[0].vars[2].ty.clone(),
//                info.branch_signatures()[1].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
//            ],
            ],
//        )?;
        )?;
//

//    let k1 = helper
    let k1 = helper
//        .init_block()
        .init_block()
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_ptr = helper
    let result_ptr = helper
//        .init_block()
        .init_block()
//        .append_operation(
        .append_operation(
//            OperationBuilder::new("llvm.alloca", location)
            OperationBuilder::new("llvm.alloca", location)
//                .add_attributes(&[
                .add_attributes(&[
//                    (
                    (
//                        Identifier::new(context, "alignment"),
                        Identifier::new(context, "alignment"),
//                        IntegerAttribute::new(
                        IntegerAttribute::new(
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            result_layout.align().try_into()?,
                            result_layout.align().try_into()?,
//                        )
                        )
//                        .into(),
                        .into(),
//                    ),
                    ),
//                    (
                    (
//                        Identifier::new(context, "elem_type"),
                        Identifier::new(context, "elem_type"),
//                        TypeAttribute::new(llvm::r#type::r#struct(
                        TypeAttribute::new(llvm::r#type::r#struct(
//                            context,
                            context,
//                            &[
                            &[
//                                result_tag_ty,
                                result_tag_ty,
//                                llvm::r#type::array(
                                llvm::r#type::array(
//                                    IntegerType::new(context, 8).into(),
                                    IntegerType::new(context, 8).into(),
//                                    (result_layout.size() - 1).try_into()?,
                                    (result_layout.size() - 1).try_into()?,
//                                ),
                                ),
//                            ],
                            ],
//                            false,
                            false,
//                        ))
                        ))
//                        .into(),
                        .into(),
//                    ),
                    ),
//                ])
                ])
//                .add_operands(&[k1])
                .add_operands(&[k1])
//                .add_results(&[llvm::r#type::pointer(context, 0)])
                .add_results(&[llvm::r#type::pointer(context, 0)])
//                .build()?,
                .build()?,
//        )
        )
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space and write the current gas.
    // Allocate space and write the current gas.
//    let gas_builtin_ptr = helper.init_block().alloca1(
    let gas_builtin_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        IntegerType::new(context, 128).into(),
        IntegerType::new(context, 128).into(),
//        Some(get_integer_layout(128).align()),
        Some(get_integer_layout(128).align()),
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        gas_builtin_ptr,
        gas_builtin_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Allocate `x` argument and write the value.
    // Allocate `x` argument and write the value.
//    let x_arg_ptr = helper.init_block().alloca1(
    let x_arg_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        llvm::r#type::r#struct(
        llvm::r#type::r#struct(
//            context,
            context,
//            &[
            &[
//                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
//                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
//            ],
            ],
//            false,
            false,
//        ),
        ),
//        None,
        None,
//    )?;
    )?;
//

//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(2)?.into(),
        entry.argument(2)?.into(),
//        x_arg_ptr,
        x_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Allocate `y_parity` argument and write the value.
    // Allocate `y_parity` argument and write the value.
//    let y_parity_arg_ptr = helper.init_block().alloca_int(context, location, 1)?;
    let y_parity_arg_ptr = helper.init_block().alloca_int(context, location, 1)?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(3)?.into(),
        entry.argument(3)?.into(),
//        y_parity_arg_ptr,
        y_parity_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::get_element_ptr(
        .append_operation(llvm::get_element_ptr(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            DenseI32ArrayAttribute::new(
            DenseI32ArrayAttribute::new(
//                context,
                context,
//                &[StarknetSyscallHandlerCallbacks::<()>::SECP256K1_GET_POINT_FROM_X.try_into()?],
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256K1_GET_POINT_FROM_X.try_into()?],
//            ),
            ),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            fn_ptr,
            fn_ptr,
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(
    entry.append_operation(
//        OperationBuilder::new("llvm.call", location)
        OperationBuilder::new("llvm.call", location)
//            .add_operands(&[
            .add_operands(&[
//                fn_ptr,
                fn_ptr,
//                result_ptr,
                result_ptr,
//                ptr,
                ptr,
//                gas_builtin_ptr,
                gas_builtin_ptr,
//                x_arg_ptr,
                x_arg_ptr,
//                y_parity_arg_ptr,
                y_parity_arg_ptr,
//            ])
            ])
//            .build()?,
            .build()?,
//    );
    );
//

//    let result = entry
    let result = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            result_ptr,
            result_ptr,
//            llvm::r#type::r#struct(
            llvm::r#type::r#struct(
//                context,
                context,
//                &[
                &[
//                    result_tag_ty,
                    result_tag_ty,
//                    llvm::r#type::array(
                    llvm::r#type::array(
//                        IntegerType::new(context, 8).into(),
                        IntegerType::new(context, 8).into(),
//                        (result_layout.size() - 1).try_into()?,
                        (result_layout.size() - 1).try_into()?,
//                    ),
                    ),
//                ],
                ],
//                false,
                false,
//            ),
            ),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_tag = entry
    let result_tag = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            result,
            result,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            IntegerType::new(context, 1).into(),
            IntegerType::new(context, 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Load the two variants of the result returned by the syscall handler.
    // Load the two variants of the result returned by the syscall handler.
//    let payload_ok = {
    let payload_ok = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry.load(
        entry.load(
//            context,
            context,
//            location,
            location,
//            ptr,
            ptr,
//            variant_tys[0].0,
            variant_tys[0].0,
//            Some(variant_tys[0].1.align()),
            Some(variant_tys[0].1.align()),
//        )?
        )?
//    };
    };
//    let payload_err = {
    let payload_err = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry
        entry
//            .append_operation(llvm::load(
            .append_operation(llvm::load(
//                context,
                context,
//                ptr,
                ptr,
//                variant_tys[1].0,
                variant_tys[1].0,
//                location,
                location,
//                LoadStoreOptions::default(),
                LoadStoreOptions::default(),
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into()
            .into()
//    };
    };
//

//    let remaining_gas = entry
    let remaining_gas = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            gas_builtin_ptr,
            gas_builtin_ptr,
//            IntegerType::new(context, 128).into(),
            IntegerType::new(context, 128).into(),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        result_tag,
        result_tag,
//        [1, 0],
        [1, 0],
//        [
        [
//            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
//            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
//        ],
        ],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

//pub fn build_k1_get_xy<'ctx, 'this>(
pub fn build_k1_get_xy<'ctx, 'this>(
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
//    // Extract self pointer.
    // Extract self pointer.
//    let ptr = entry
    let ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space for the return value.
    // Allocate space for the return value.
//    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) = {
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) = {
//        // Note: This libfunc has multiple return values when successful, therefore the method used
        // Note: This libfunc has multiple return values when successful, therefore the method used
//        //   for the other libfuncs cannot be reused here.
        //   for the other libfuncs cannot be reused here.
//

//        let u128_layout = get_integer_layout(128);
        let u128_layout = get_integer_layout(128);
//        let u256_layout = u128_layout.extend(u128_layout)?.0;
        let u256_layout = u128_layout.extend(u128_layout)?.0;
//        let u256_ty = llvm::r#type::r#struct(
        let u256_ty = llvm::r#type::r#struct(
//            context,
            context,
//            &[
            &[
//                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
//                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
//            ],
            ],
//            false,
            false,
//        );
        );
//

//        let (ok_ty, ok_layout) = (
        let (ok_ty, ok_layout) = (
//            llvm::r#type::r#struct(context, &[u256_ty, u256_ty], false),
            llvm::r#type::r#struct(context, &[u256_ty, u256_ty], false),
//            u256_layout.extend(u256_layout)?.0,
            u256_layout.extend(u256_layout)?.0,
//        );
        );
//        let (err_ty, err_layout) = registry.build_type_with_layout(
        let (err_ty, err_layout) = registry.build_type_with_layout(
//            context,
            context,
//            helper,
            helper,
//            registry,
            registry,
//            metadata,
            metadata,
//            &info.branch_signatures()[1].vars[2].ty,
            &info.branch_signatures()[1].vars[2].ty,
//        )?;
        )?;
//

//        let (tag_ty, tag_layout) = (IntegerType::new(context, 1).into(), get_integer_layout(1));
        let (tag_ty, tag_layout) = (IntegerType::new(context, 1).into(), get_integer_layout(1));
//

//        (
        (
//            tag_layout
            tag_layout
//                .extend(Layout::from_size_align(
                .extend(Layout::from_size_align(
//                    ok_layout.size().max(err_layout.size()),
                    ok_layout.size().max(err_layout.size()),
//                    ok_layout.align().max(err_layout.align()),
                    ok_layout.align().max(err_layout.align()),
//                )?)?
                )?)?
//                .0,
                .0,
//            (tag_ty, tag_layout),
            (tag_ty, tag_layout),
//            [(ok_ty, ok_layout), (err_ty, err_layout)],
            [(ok_ty, ok_layout), (err_ty, err_layout)],
//        )
        )
//    };
    };
//

//    let k1 = helper
    let k1 = helper
//        .init_block()
        .init_block()
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_ptr = helper
    let result_ptr = helper
//        .init_block()
        .init_block()
//        .append_operation(
        .append_operation(
//            OperationBuilder::new("llvm.alloca", location)
            OperationBuilder::new("llvm.alloca", location)
//                .add_attributes(&[
                .add_attributes(&[
//                    (
                    (
//                        Identifier::new(context, "alignment"),
                        Identifier::new(context, "alignment"),
//                        IntegerAttribute::new(
                        IntegerAttribute::new(
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            result_layout.align().try_into()?,
                            result_layout.align().try_into()?,
//                        )
                        )
//                        .into(),
                        .into(),
//                    ),
                    ),
//                    (
                    (
//                        Identifier::new(context, "elem_type"),
                        Identifier::new(context, "elem_type"),
//                        TypeAttribute::new(llvm::r#type::r#struct(
                        TypeAttribute::new(llvm::r#type::r#struct(
//                            context,
                            context,
//                            &[
                            &[
//                                result_tag_ty,
                                result_tag_ty,
//                                llvm::r#type::array(
                                llvm::r#type::array(
//                                    IntegerType::new(context, 8).into(),
                                    IntegerType::new(context, 8).into(),
//                                    (result_layout.size() - 1).try_into()?,
                                    (result_layout.size() - 1).try_into()?,
//                                ),
                                ),
//                            ],
                            ],
//                            false,
                            false,
//                        ))
                        ))
//                        .into(),
                        .into(),
//                    ),
                    ),
//                ])
                ])
//                .add_operands(&[k1])
                .add_operands(&[k1])
//                .add_results(&[llvm::r#type::pointer(context, 0)])
                .add_results(&[llvm::r#type::pointer(context, 0)])
//                .build()?,
                .build()?,
//        )
        )
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space and write the current gas.
    // Allocate space and write the current gas.
//    let gas_builtin_ptr = helper.init_block().alloca1(
    let gas_builtin_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        IntegerType::new(context, 128).into(),
        IntegerType::new(context, 128).into(),
//        Some(get_integer_layout(128).align()),
        Some(get_integer_layout(128).align()),
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        gas_builtin_ptr,
        gas_builtin_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Allocate `p` argument and write the value.
    // Allocate `p` argument and write the value.
//    let p_arg_ptr = helper.init_block().alloca1(
    let p_arg_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        llvm::r#type::r#struct(
        llvm::r#type::r#struct(
//            context,
            context,
//            &[
            &[
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//            ],
            ],
//            false,
            false,
//        ),
        ),
//        None,
        None,
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(2)?.into(),
        entry.argument(2)?.into(),
//        p_arg_ptr,
        p_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::get_element_ptr(
        .append_operation(llvm::get_element_ptr(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            DenseI32ArrayAttribute::new(
            DenseI32ArrayAttribute::new(
//                context,
                context,
//                &[StarknetSyscallHandlerCallbacks::<()>::SECP256K1_GET_XY.try_into()?],
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256K1_GET_XY.try_into()?],
//            ),
            ),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            fn_ptr,
            fn_ptr,
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(
    entry.append_operation(
//        OperationBuilder::new("llvm.call", location)
        OperationBuilder::new("llvm.call", location)
//            .add_operands(&[fn_ptr, result_ptr, ptr, gas_builtin_ptr, p_arg_ptr])
            .add_operands(&[fn_ptr, result_ptr, ptr, gas_builtin_ptr, p_arg_ptr])
//            .build()?,
            .build()?,
//    );
    );
//

//    let result = entry
    let result = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            result_ptr,
            result_ptr,
//            llvm::r#type::r#struct(
            llvm::r#type::r#struct(
//                context,
                context,
//                &[
                &[
//                    result_tag_ty,
                    result_tag_ty,
//                    llvm::r#type::array(
                    llvm::r#type::array(
//                        IntegerType::new(context, 8).into(),
                        IntegerType::new(context, 8).into(),
//                        (result_layout.size() - 1).try_into()?,
                        (result_layout.size() - 1).try_into()?,
//                    ),
                    ),
//                ],
                ],
//                false,
                false,
//            ),
            ),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_tag = entry
    let result_tag = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            result,
            result,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            IntegerType::new(context, 1).into(),
            IntegerType::new(context, 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let payload_ok = {
    let payload_ok = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let value = entry
        let value = entry
//            .append_operation(llvm::load(
            .append_operation(llvm::load(
//                context,
                context,
//                ptr,
                ptr,
//                variant_tys[0].0,
                variant_tys[0].0,
//                location,
                location,
//                LoadStoreOptions::default(),
                LoadStoreOptions::default(),
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        let x_value = entry
        let x_value = entry
//            .append_operation(llvm::extract_value(
            .append_operation(llvm::extract_value(
//                context,
                context,
//                value,
                value,
//                DenseI64ArrayAttribute::new(context, &[0]),
                DenseI64ArrayAttribute::new(context, &[0]),
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let y_value = entry
        let y_value = entry
//            .append_operation(llvm::extract_value(
            .append_operation(llvm::extract_value(
//                context,
                context,
//                value,
                value,
//                DenseI64ArrayAttribute::new(context, &[1]),
                DenseI64ArrayAttribute::new(context, &[1]),
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        (x_value, y_value)
        (x_value, y_value)
//    };
    };
//    let payload_err = {
    let payload_err = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry
        entry
//            .append_operation(llvm::load(
            .append_operation(llvm::load(
//                context,
                context,
//                ptr,
                ptr,
//                variant_tys[1].0,
                variant_tys[1].0,
//                location,
                location,
//                LoadStoreOptions::default(),
                LoadStoreOptions::default(),
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into()
            .into()
//    };
    };
//

//    let remaining_gas = entry
    let remaining_gas = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            gas_builtin_ptr,
            gas_builtin_ptr,
//            IntegerType::new(context, 128).into(),
            IntegerType::new(context, 128).into(),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        result_tag,
        result_tag,
//        [1, 0],
        [1, 0],
//        [
        [
//            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
//            &[
            &[
//                remaining_gas,
                remaining_gas,
//                entry.argument(1)?.into(),
                entry.argument(1)?.into(),
//                payload_ok.0,
                payload_ok.0,
//                payload_ok.1,
                payload_ok.1,
//            ],
            ],
//        ],
        ],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

//pub fn build_r1_new<'ctx, 'this>(
pub fn build_r1_new<'ctx, 'this>(
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
//    // Extract self pointer.
    // Extract self pointer.
//    let ptr = entry
    let ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space for the return value.
    // Allocate space for the return value.
//    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
//        crate::types::r#enum::get_type_for_variants(
        crate::types::r#enum::get_type_for_variants(
//            context,
            context,
//            helper,
            helper,
//            registry,
            registry,
//            metadata,
            metadata,
//            &[
            &[
//                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[0].vars[2].ty.clone(),
//                info.branch_signatures()[1].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
//            ],
            ],
//        )?;
        )?;
//

//    let k1 = helper
    let k1 = helper
//        .init_block()
        .init_block()
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_ptr = helper
    let result_ptr = helper
//        .init_block()
        .init_block()
//        .append_operation(
        .append_operation(
//            OperationBuilder::new("llvm.alloca", location)
            OperationBuilder::new("llvm.alloca", location)
//                .add_attributes(&[
                .add_attributes(&[
//                    (
                    (
//                        Identifier::new(context, "alignment"),
                        Identifier::new(context, "alignment"),
//                        IntegerAttribute::new(
                        IntegerAttribute::new(
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            result_layout.align().try_into()?,
                            result_layout.align().try_into()?,
//                        )
                        )
//                        .into(),
                        .into(),
//                    ),
                    ),
//                    (
                    (
//                        Identifier::new(context, "elem_type"),
                        Identifier::new(context, "elem_type"),
//                        TypeAttribute::new(llvm::r#type::r#struct(
                        TypeAttribute::new(llvm::r#type::r#struct(
//                            context,
                            context,
//                            &[
                            &[
//                                result_tag_ty,
                                result_tag_ty,
//                                llvm::r#type::array(
                                llvm::r#type::array(
//                                    IntegerType::new(context, 8).into(),
                                    IntegerType::new(context, 8).into(),
//                                    (result_layout.size() - 1).try_into()?,
                                    (result_layout.size() - 1).try_into()?,
//                                ),
                                ),
//                            ],
                            ],
//                            false,
                            false,
//                        ))
                        ))
//                        .into(),
                        .into(),
//                    ),
                    ),
//                ])
                ])
//                .add_operands(&[k1])
                .add_operands(&[k1])
//                .add_results(&[llvm::r#type::pointer(context, 0)])
                .add_results(&[llvm::r#type::pointer(context, 0)])
//                .build()?,
                .build()?,
//        )
        )
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space and write the current gas.
    // Allocate space and write the current gas.
//    let gas_builtin_ptr = helper.init_block().alloca1(
    let gas_builtin_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        IntegerType::new(context, 128).into(),
        IntegerType::new(context, 128).into(),
//        Some(get_integer_layout(128).align()),
        Some(get_integer_layout(128).align()),
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        gas_builtin_ptr,
        gas_builtin_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    let i128_ty = IntegerType::new(context, 128).into();
    let i128_ty = IntegerType::new(context, 128).into();
//

//    // Allocate `x` argument and write the value.
    // Allocate `x` argument and write the value.
//    let x_arg_ptr = helper.init_block().alloca1(
    let x_arg_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
        llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
//        None,
        None,
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(2)?.into(),
        entry.argument(2)?.into(),
//        x_arg_ptr,
        x_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Allocate `y` argument and write the value.
    // Allocate `y` argument and write the value.
//    let y_arg_ptr = helper.init_block().alloca1(
    let y_arg_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
        llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
//        None,
        None,
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(3)?.into(),
        entry.argument(3)?.into(),
//        y_arg_ptr,
        y_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::get_element_ptr(
        .append_operation(llvm::get_element_ptr(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            DenseI32ArrayAttribute::new(
            DenseI32ArrayAttribute::new(
//                context,
                context,
//                &[StarknetSyscallHandlerCallbacks::<()>::SECP256R1_NEW.try_into()?],
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256R1_NEW.try_into()?],
//            ),
            ),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            fn_ptr,
            fn_ptr,
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(
    entry.append_operation(
//        OperationBuilder::new("llvm.call", location)
        OperationBuilder::new("llvm.call", location)
//            .add_operands(&[
            .add_operands(&[
//                fn_ptr,
                fn_ptr,
//                result_ptr,
                result_ptr,
//                ptr,
                ptr,
//                gas_builtin_ptr,
                gas_builtin_ptr,
//                x_arg_ptr,
                x_arg_ptr,
//                y_arg_ptr,
                y_arg_ptr,
//            ])
            ])
//            .build()?,
            .build()?,
//    );
    );
//

//    let result = entry
    let result = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            result_ptr,
            result_ptr,
//            llvm::r#type::r#struct(
            llvm::r#type::r#struct(
//                context,
                context,
//                &[
                &[
//                    result_tag_ty,
                    result_tag_ty,
//                    llvm::r#type::array(
                    llvm::r#type::array(
//                        IntegerType::new(context, 8).into(),
                        IntegerType::new(context, 8).into(),
//                        (result_layout.size() - 1).try_into()?,
                        (result_layout.size() - 1).try_into()?,
//                    ),
                    ),
//                ],
                ],
//                false,
                false,
//            ),
            ),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_tag = entry
    let result_tag = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            result,
            result,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            IntegerType::new(context, 1).into(),
            IntegerType::new(context, 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Load the two variants of the result returned by the syscall handler.
    // Load the two variants of the result returned by the syscall handler.
//    let payload_ok = {
    let payload_ok = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry.load(
        entry.load(
//            context,
            context,
//            location,
            location,
//            ptr,
            ptr,
//            variant_tys[0].0,
            variant_tys[0].0,
//            Some(variant_tys[0].1.align()),
            Some(variant_tys[0].1.align()),
//        )?
        )?
//    };
    };
//    let payload_err = {
    let payload_err = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry
        entry
//            .append_operation(llvm::load(
            .append_operation(llvm::load(
//                context,
                context,
//                ptr,
                ptr,
//                variant_tys[1].0,
                variant_tys[1].0,
//                location,
                location,
//                LoadStoreOptions::default(),
                LoadStoreOptions::default(),
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into()
            .into()
//    };
    };
//

//    let remaining_gas = entry
    let remaining_gas = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            gas_builtin_ptr,
            gas_builtin_ptr,
//            IntegerType::new(context, 128).into(),
            IntegerType::new(context, 128).into(),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        result_tag,
        result_tag,
//        [1, 0],
        [1, 0],
//        [
        [
//            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
//            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
//        ],
        ],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

//pub fn build_r1_add<'ctx, 'this>(
pub fn build_r1_add<'ctx, 'this>(
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
//    // Extract self pointer.
    // Extract self pointer.
//    let ptr = entry
    let ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space for the return value.
    // Allocate space for the return value.
//    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
//        crate::types::r#enum::get_type_for_variants(
        crate::types::r#enum::get_type_for_variants(
//            context,
            context,
//            helper,
            helper,
//            registry,
            registry,
//            metadata,
            metadata,
//            &[
            &[
//                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[0].vars[2].ty.clone(),
//                info.branch_signatures()[1].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
//            ],
            ],
//        )?;
        )?;
//

//    let k1 = helper
    let k1 = helper
//        .init_block()
        .init_block()
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_ptr = helper
    let result_ptr = helper
//        .init_block()
        .init_block()
//        .append_operation(
        .append_operation(
//            OperationBuilder::new("llvm.alloca", location)
            OperationBuilder::new("llvm.alloca", location)
//                .add_attributes(&[
                .add_attributes(&[
//                    (
                    (
//                        Identifier::new(context, "alignment"),
                        Identifier::new(context, "alignment"),
//                        IntegerAttribute::new(
                        IntegerAttribute::new(
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            result_layout.align().try_into()?,
                            result_layout.align().try_into()?,
//                        )
                        )
//                        .into(),
                        .into(),
//                    ),
                    ),
//                    (
                    (
//                        Identifier::new(context, "elem_type"),
                        Identifier::new(context, "elem_type"),
//                        TypeAttribute::new(llvm::r#type::r#struct(
                        TypeAttribute::new(llvm::r#type::r#struct(
//                            context,
                            context,
//                            &[
                            &[
//                                result_tag_ty,
                                result_tag_ty,
//                                llvm::r#type::array(
                                llvm::r#type::array(
//                                    IntegerType::new(context, 8).into(),
                                    IntegerType::new(context, 8).into(),
//                                    (result_layout.size() - 1).try_into()?,
                                    (result_layout.size() - 1).try_into()?,
//                                ),
                                ),
//                            ],
                            ],
//                            false,
                            false,
//                        ))
                        ))
//                        .into(),
                        .into(),
//                    ),
                    ),
//                ])
                ])
//                .add_operands(&[k1])
                .add_operands(&[k1])
//                .add_results(&[llvm::r#type::pointer(context, 0)])
                .add_results(&[llvm::r#type::pointer(context, 0)])
//                .build()?,
                .build()?,
//        )
        )
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space and write the current gas.
    // Allocate space and write the current gas.
//    let gas_builtin_ptr = helper.init_block().alloca1(
    let gas_builtin_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        IntegerType::new(context, 128).into(),
        IntegerType::new(context, 128).into(),
//        Some(get_integer_layout(128).align()),
        Some(get_integer_layout(128).align()),
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        gas_builtin_ptr,
        gas_builtin_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Allocate `p0` argument and write the value.
    // Allocate `p0` argument and write the value.
//    let p0_arg_ptr = helper.init_block().alloca1(
    let p0_arg_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        llvm::r#type::r#struct(
        llvm::r#type::r#struct(
//            context,
            context,
//            &[
            &[
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//            ],
            ],
//            false,
            false,
//        ),
        ),
//        None,
        None,
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(2)?.into(),
        entry.argument(2)?.into(),
//        p0_arg_ptr,
        p0_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Allocate `p1` argument and write the value.
    // Allocate `p1` argument and write the value.
//    let p1_arg_ptr = helper.init_block().alloca1(
    let p1_arg_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        llvm::r#type::r#struct(
        llvm::r#type::r#struct(
//            context,
            context,
//            &[
            &[
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//            ],
            ],
//            false,
            false,
//        ),
        ),
//        None,
        None,
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(3)?.into(),
        entry.argument(3)?.into(),
//        p1_arg_ptr,
        p1_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Extract function pointer.
    // Extract function pointer.
//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::get_element_ptr(
        .append_operation(llvm::get_element_ptr(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            DenseI32ArrayAttribute::new(
            DenseI32ArrayAttribute::new(
//                context,
                context,
//                &[StarknetSyscallHandlerCallbacks::<()>::SECP256R1_ADD.try_into()?],
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256R1_ADD.try_into()?],
//            ),
            ),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            fn_ptr,
            fn_ptr,
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(
    entry.append_operation(
//        OperationBuilder::new("llvm.call", location)
        OperationBuilder::new("llvm.call", location)
//            .add_operands(&[
            .add_operands(&[
//                fn_ptr,
                fn_ptr,
//                result_ptr,
                result_ptr,
//                ptr,
                ptr,
//                gas_builtin_ptr,
                gas_builtin_ptr,
//                p0_arg_ptr,
                p0_arg_ptr,
//                p1_arg_ptr,
                p1_arg_ptr,
//            ])
            ])
//            .build()?,
            .build()?,
//    );
    );
//

//    let result = entry
    let result = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            result_ptr,
            result_ptr,
//            llvm::r#type::r#struct(
            llvm::r#type::r#struct(
//                context,
                context,
//                &[
                &[
//                    result_tag_ty,
                    result_tag_ty,
//                    llvm::r#type::array(
                    llvm::r#type::array(
//                        IntegerType::new(context, 8).into(),
                        IntegerType::new(context, 8).into(),
//                        (result_layout.size() - 1).try_into()?,
                        (result_layout.size() - 1).try_into()?,
//                    ),
                    ),
//                ],
                ],
//                false,
                false,
//            ),
            ),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_tag = entry
    let result_tag = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            result,
            result,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            IntegerType::new(context, 1).into(),
            IntegerType::new(context, 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let payload_ok = {
    let payload_ok = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry
        entry
//            .append_operation(llvm::load(
            .append_operation(llvm::load(
//                context,
                context,
//                ptr,
                ptr,
//                variant_tys[0].0,
                variant_tys[0].0,
//                location,
                location,
//                LoadStoreOptions::default(),
                LoadStoreOptions::default(),
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into()
            .into()
//    };
    };
//    let payload_err = {
    let payload_err = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry
        entry
//            .append_operation(llvm::load(
            .append_operation(llvm::load(
//                context,
                context,
//                ptr,
                ptr,
//                variant_tys[1].0,
                variant_tys[1].0,
//                location,
                location,
//                LoadStoreOptions::default(),
                LoadStoreOptions::default(),
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into()
            .into()
//    };
    };
//

//    let remaining_gas = entry
    let remaining_gas = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            gas_builtin_ptr,
            gas_builtin_ptr,
//            IntegerType::new(context, 128).into(),
            IntegerType::new(context, 128).into(),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        result_tag,
        result_tag,
//        [1, 0],
        [1, 0],
//        [
        [
//            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
//            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
//        ],
        ],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

//pub fn build_r1_mul<'ctx, 'this>(
pub fn build_r1_mul<'ctx, 'this>(
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
//    // Extract self pointer.
    // Extract self pointer.
//    let ptr = entry
    let ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space for the return value.
    // Allocate space for the return value.
//    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
//        crate::types::r#enum::get_type_for_variants(
        crate::types::r#enum::get_type_for_variants(
//            context,
            context,
//            helper,
            helper,
//            registry,
            registry,
//            metadata,
            metadata,
//            &[
            &[
//                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[0].vars[2].ty.clone(),
//                info.branch_signatures()[1].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
//            ],
            ],
//        )?;
        )?;
//

//    let k1 = helper
    let k1 = helper
//        .init_block()
        .init_block()
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_ptr = helper
    let result_ptr = helper
//        .init_block()
        .init_block()
//        .append_operation(
        .append_operation(
//            OperationBuilder::new("llvm.alloca", location)
            OperationBuilder::new("llvm.alloca", location)
//                .add_attributes(&[
                .add_attributes(&[
//                    (
                    (
//                        Identifier::new(context, "alignment"),
                        Identifier::new(context, "alignment"),
//                        IntegerAttribute::new(
                        IntegerAttribute::new(
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            result_layout.align().try_into()?,
                            result_layout.align().try_into()?,
//                        )
                        )
//                        .into(),
                        .into(),
//                    ),
                    ),
//                    (
                    (
//                        Identifier::new(context, "elem_type"),
                        Identifier::new(context, "elem_type"),
//                        TypeAttribute::new(llvm::r#type::r#struct(
                        TypeAttribute::new(llvm::r#type::r#struct(
//                            context,
                            context,
//                            &[
                            &[
//                                result_tag_ty,
                                result_tag_ty,
//                                llvm::r#type::array(
                                llvm::r#type::array(
//                                    IntegerType::new(context, 8).into(),
                                    IntegerType::new(context, 8).into(),
//                                    (result_layout.size() - 1).try_into()?,
                                    (result_layout.size() - 1).try_into()?,
//                                ),
                                ),
//                            ],
                            ],
//                            false,
                            false,
//                        ))
                        ))
//                        .into(),
                        .into(),
//                    ),
                    ),
//                ])
                ])
//                .add_operands(&[k1])
                .add_operands(&[k1])
//                .add_results(&[llvm::r#type::pointer(context, 0)])
                .add_results(&[llvm::r#type::pointer(context, 0)])
//                .build()?,
                .build()?,
//        )
        )
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space and write the current gas.
    // Allocate space and write the current gas.
//    let gas_builtin_ptr = helper.init_block().alloca1(
    let gas_builtin_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        IntegerType::new(context, 128).into(),
        IntegerType::new(context, 128).into(),
//        Some(get_integer_layout(128).align()),
        Some(get_integer_layout(128).align()),
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        gas_builtin_ptr,
        gas_builtin_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Allocate `p` argument and write the value.
    // Allocate `p` argument and write the value.
//    let p_arg_ptr = helper.init_block().alloca1(
    let p_arg_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        llvm::r#type::r#struct(
        llvm::r#type::r#struct(
//            context,
            context,
//            &[
            &[
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//            ],
            ],
//            false,
            false,
//        ),
        ),
//        None,
        None,
//    )?;
    )?;
//

//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(2)?.into(),
        entry.argument(2)?.into(),
//        p_arg_ptr,
        p_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Allocate `scalar` argument and write the value.
    // Allocate `scalar` argument and write the value.
//    let scalar_arg_ptr = helper.init_block().alloca1(
    let scalar_arg_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        llvm::r#type::r#struct(
        llvm::r#type::r#struct(
//            context,
            context,
//            &[
            &[
//                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
//                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
//            ],
            ],
//            false,
            false,
//        ),
        ),
//        None,
        None,
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(3)?.into(),
        entry.argument(3)?.into(),
//        scalar_arg_ptr,
        scalar_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Extract function pointer.
    // Extract function pointer.
//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::get_element_ptr(
        .append_operation(llvm::get_element_ptr(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            DenseI32ArrayAttribute::new(
            DenseI32ArrayAttribute::new(
//                context,
                context,
//                &[StarknetSyscallHandlerCallbacks::<()>::SECP256R1_MUL.try_into()?],
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256R1_MUL.try_into()?],
//            ),
            ),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            fn_ptr,
            fn_ptr,
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(
    entry.append_operation(
//        OperationBuilder::new("llvm.call", location)
        OperationBuilder::new("llvm.call", location)
//            .add_operands(&[
            .add_operands(&[
//                fn_ptr,
                fn_ptr,
//                result_ptr,
                result_ptr,
//                ptr,
                ptr,
//                gas_builtin_ptr,
                gas_builtin_ptr,
//                p_arg_ptr,
                p_arg_ptr,
//                scalar_arg_ptr,
                scalar_arg_ptr,
//            ])
            ])
//            .build()?,
            .build()?,
//    );
    );
//

//    let result = entry
    let result = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            result_ptr,
            result_ptr,
//            llvm::r#type::r#struct(
            llvm::r#type::r#struct(
//                context,
                context,
//                &[
                &[
//                    result_tag_ty,
                    result_tag_ty,
//                    llvm::r#type::array(
                    llvm::r#type::array(
//                        IntegerType::new(context, 8).into(),
                        IntegerType::new(context, 8).into(),
//                        (result_layout.size() - 1).try_into()?,
                        (result_layout.size() - 1).try_into()?,
//                    ),
                    ),
//                ],
                ],
//                false,
                false,
//            ),
            ),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_tag = entry
    let result_tag = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            result,
            result,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            IntegerType::new(context, 1).into(),
            IntegerType::new(context, 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let payload_ok = {
    let payload_ok = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry
        entry
//            .append_operation(llvm::load(
            .append_operation(llvm::load(
//                context,
                context,
//                ptr,
                ptr,
//                variant_tys[0].0,
                variant_tys[0].0,
//                location,
                location,
//                LoadStoreOptions::default(),
                LoadStoreOptions::default(),
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into()
            .into()
//    };
    };
//    let payload_err = {
    let payload_err = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry
        entry
//            .append_operation(llvm::load(
            .append_operation(llvm::load(
//                context,
                context,
//                ptr,
                ptr,
//                variant_tys[1].0,
                variant_tys[1].0,
//                location,
                location,
//                LoadStoreOptions::default(),
                LoadStoreOptions::default(),
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into()
            .into()
//    };
    };
//

//    let remaining_gas = entry
    let remaining_gas = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            gas_builtin_ptr,
            gas_builtin_ptr,
//            IntegerType::new(context, 128).into(),
            IntegerType::new(context, 128).into(),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        result_tag,
        result_tag,
//        [1, 0],
        [1, 0],
//        [
        [
//            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
//            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
//        ],
        ],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

//pub fn build_r1_get_point_from_x<'ctx, 'this>(
pub fn build_r1_get_point_from_x<'ctx, 'this>(
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
//    // Extract self pointer.
    // Extract self pointer.
//    let ptr = entry
    let ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space for the return value.
    // Allocate space for the return value.
//    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
//        crate::types::r#enum::get_type_for_variants(
        crate::types::r#enum::get_type_for_variants(
//            context,
            context,
//            helper,
            helper,
//            registry,
            registry,
//            metadata,
            metadata,
//            &[
            &[
//                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[0].vars[2].ty.clone(),
//                info.branch_signatures()[1].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
//            ],
            ],
//        )?;
        )?;
//

//    let k1 = helper
    let k1 = helper
//        .init_block()
        .init_block()
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_ptr = helper
    let result_ptr = helper
//        .init_block()
        .init_block()
//        .append_operation(
        .append_operation(
//            OperationBuilder::new("llvm.alloca", location)
            OperationBuilder::new("llvm.alloca", location)
//                .add_attributes(&[
                .add_attributes(&[
//                    (
                    (
//                        Identifier::new(context, "alignment"),
                        Identifier::new(context, "alignment"),
//                        IntegerAttribute::new(
                        IntegerAttribute::new(
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            result_layout.align().try_into()?,
                            result_layout.align().try_into()?,
//                        )
                        )
//                        .into(),
                        .into(),
//                    ),
                    ),
//                    (
                    (
//                        Identifier::new(context, "elem_type"),
                        Identifier::new(context, "elem_type"),
//                        TypeAttribute::new(llvm::r#type::r#struct(
                        TypeAttribute::new(llvm::r#type::r#struct(
//                            context,
                            context,
//                            &[
                            &[
//                                result_tag_ty,
                                result_tag_ty,
//                                llvm::r#type::array(
                                llvm::r#type::array(
//                                    IntegerType::new(context, 8).into(),
                                    IntegerType::new(context, 8).into(),
//                                    (result_layout.size() - 1).try_into()?,
                                    (result_layout.size() - 1).try_into()?,
//                                ),
                                ),
//                            ],
                            ],
//                            false,
                            false,
//                        ))
                        ))
//                        .into(),
                        .into(),
//                    ),
                    ),
//                ])
                ])
//                .add_operands(&[k1])
                .add_operands(&[k1])
//                .add_results(&[llvm::r#type::pointer(context, 0)])
                .add_results(&[llvm::r#type::pointer(context, 0)])
//                .build()?,
                .build()?,
//        )
        )
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space and write the current gas.
    // Allocate space and write the current gas.
//    let gas_builtin_ptr = helper.init_block().alloca1(
    let gas_builtin_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        IntegerType::new(context, 128).into(),
        IntegerType::new(context, 128).into(),
//        Some(get_integer_layout(128).align()),
        Some(get_integer_layout(128).align()),
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        gas_builtin_ptr,
        gas_builtin_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Allocate `x` argument and write the value.
    // Allocate `x` argument and write the value.
//    let x_arg_ptr = helper.init_block().alloca1(
    let x_arg_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        llvm::r#type::r#struct(
        llvm::r#type::r#struct(
//            context,
            context,
//            &[
            &[
//                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
//                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
//            ],
            ],
//            false,
            false,
//        ),
        ),
//        None,
        None,
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(2)?.into(),
        entry.argument(2)?.into(),
//        x_arg_ptr,
        x_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Allocate `y_parity` argument and write the value.
    // Allocate `y_parity` argument and write the value.
//    let y_parity_arg_ptr = helper.init_block().alloca_int(context, location, 1)?;
    let y_parity_arg_ptr = helper.init_block().alloca_int(context, location, 1)?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(3)?.into(),
        entry.argument(3)?.into(),
//        y_parity_arg_ptr,
        y_parity_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Extract function pointer.
    // Extract function pointer.
//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::get_element_ptr(
        .append_operation(llvm::get_element_ptr(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            DenseI32ArrayAttribute::new(
            DenseI32ArrayAttribute::new(
//                context,
                context,
//                &[StarknetSyscallHandlerCallbacks::<()>::SECP256R1_GET_POINT_FROM_X.try_into()?],
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256R1_GET_POINT_FROM_X.try_into()?],
//            ),
            ),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            fn_ptr,
            fn_ptr,
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(
    entry.append_operation(
//        OperationBuilder::new("llvm.call", location)
        OperationBuilder::new("llvm.call", location)
//            .add_operands(&[
            .add_operands(&[
//                fn_ptr,
                fn_ptr,
//                result_ptr,
                result_ptr,
//                ptr,
                ptr,
//                gas_builtin_ptr,
                gas_builtin_ptr,
//                x_arg_ptr,
                x_arg_ptr,
//                y_parity_arg_ptr,
                y_parity_arg_ptr,
//            ])
            ])
//            .build()?,
            .build()?,
//    );
    );
//

//    let result = entry
    let result = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            result_ptr,
            result_ptr,
//            llvm::r#type::r#struct(
            llvm::r#type::r#struct(
//                context,
                context,
//                &[
                &[
//                    result_tag_ty,
                    result_tag_ty,
//                    llvm::r#type::array(
                    llvm::r#type::array(
//                        IntegerType::new(context, 8).into(),
                        IntegerType::new(context, 8).into(),
//                        (result_layout.size() - 1).try_into()?,
                        (result_layout.size() - 1).try_into()?,
//                    ),
                    ),
//                ],
                ],
//                false,
                false,
//            ),
            ),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_tag = entry
    let result_tag = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            result,
            result,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            IntegerType::new(context, 1).into(),
            IntegerType::new(context, 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let payload_ok = {
    let payload_ok = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry.load(
        entry.load(
//            context,
            context,
//            location,
            location,
//            ptr,
            ptr,
//            variant_tys[0].0,
            variant_tys[0].0,
//            Some(variant_tys[0].1.align()),
            Some(variant_tys[0].1.align()),
//        )?
        )?
//    };
    };
//    let payload_err = {
    let payload_err = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry
        entry
//            .append_operation(llvm::load(
            .append_operation(llvm::load(
//                context,
                context,
//                ptr,
                ptr,
//                variant_tys[1].0,
                variant_tys[1].0,
//                location,
                location,
//                LoadStoreOptions::default(),
                LoadStoreOptions::default(),
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into()
            .into()
//    };
    };
//

//    let remaining_gas = entry
    let remaining_gas = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            gas_builtin_ptr,
            gas_builtin_ptr,
//            IntegerType::new(context, 128).into(),
            IntegerType::new(context, 128).into(),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        result_tag,
        result_tag,
//        [1, 0],
        [1, 0],
//        [
        [
//            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
//            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
//        ],
        ],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

//pub fn build_r1_get_xy<'ctx, 'this>(
pub fn build_r1_get_xy<'ctx, 'this>(
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
//    // Extract self pointer.
    // Extract self pointer.
//    let ptr = entry
    let ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space for the return value.
    // Allocate space for the return value.
//    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) = {
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) = {
//        // Note: This libfunc has multiple return values when successful, therefore the method used
        // Note: This libfunc has multiple return values when successful, therefore the method used
//        //   for the other libfuncs cannot be reused here.
        //   for the other libfuncs cannot be reused here.
//

//        let u128_layout = get_integer_layout(128);
        let u128_layout = get_integer_layout(128);
//        let u256_layout = u128_layout.extend(u128_layout)?.0;
        let u256_layout = u128_layout.extend(u128_layout)?.0;
//        let u256_ty = llvm::r#type::r#struct(
        let u256_ty = llvm::r#type::r#struct(
//            context,
            context,
//            &[
            &[
//                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
//                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
//            ],
            ],
//            false,
            false,
//        );
        );
//

//        let (ok_ty, ok_layout) = (
        let (ok_ty, ok_layout) = (
//            llvm::r#type::r#struct(context, &[u256_ty, u256_ty], false),
            llvm::r#type::r#struct(context, &[u256_ty, u256_ty], false),
//            u256_layout.extend(u256_layout)?.0,
            u256_layout.extend(u256_layout)?.0,
//        );
        );
//        let (err_ty, err_layout) = registry.build_type_with_layout(
        let (err_ty, err_layout) = registry.build_type_with_layout(
//            context,
            context,
//            helper,
            helper,
//            registry,
            registry,
//            metadata,
            metadata,
//            &info.branch_signatures()[1].vars[2].ty,
            &info.branch_signatures()[1].vars[2].ty,
//        )?;
        )?;
//

//        let (tag_ty, tag_layout) = (IntegerType::new(context, 1).into(), get_integer_layout(1));
        let (tag_ty, tag_layout) = (IntegerType::new(context, 1).into(), get_integer_layout(1));
//

//        (
        (
//            tag_layout
            tag_layout
//                .extend(Layout::from_size_align(
                .extend(Layout::from_size_align(
//                    ok_layout.size().max(err_layout.size()),
                    ok_layout.size().max(err_layout.size()),
//                    ok_layout.align().max(err_layout.align()),
                    ok_layout.align().max(err_layout.align()),
//                )?)?
                )?)?
//                .0,
                .0,
//            (tag_ty, tag_layout),
            (tag_ty, tag_layout),
//            [(ok_ty, ok_layout), (err_ty, err_layout)],
            [(ok_ty, ok_layout), (err_ty, err_layout)],
//        )
        )
//    };
    };
//

//    let k1 = helper
    let k1 = helper
//        .init_block()
        .init_block()
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_ptr = helper
    let result_ptr = helper
//        .init_block()
        .init_block()
//        .append_operation(
        .append_operation(
//            OperationBuilder::new("llvm.alloca", location)
            OperationBuilder::new("llvm.alloca", location)
//                .add_attributes(&[
                .add_attributes(&[
//                    (
                    (
//                        Identifier::new(context, "alignment"),
                        Identifier::new(context, "alignment"),
//                        IntegerAttribute::new(
                        IntegerAttribute::new(
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            result_layout.align().try_into()?,
                            result_layout.align().try_into()?,
//                        )
                        )
//                        .into(),
                        .into(),
//                    ),
                    ),
//                    (
                    (
//                        Identifier::new(context, "elem_type"),
                        Identifier::new(context, "elem_type"),
//                        TypeAttribute::new(llvm::r#type::r#struct(
                        TypeAttribute::new(llvm::r#type::r#struct(
//                            context,
                            context,
//                            &[
                            &[
//                                result_tag_ty,
                                result_tag_ty,
//                                llvm::r#type::array(
                                llvm::r#type::array(
//                                    IntegerType::new(context, 8).into(),
                                    IntegerType::new(context, 8).into(),
//                                    (result_layout.size() - 1).try_into()?,
                                    (result_layout.size() - 1).try_into()?,
//                                ),
                                ),
//                            ],
                            ],
//                            false,
                            false,
//                        ))
                        ))
//                        .into(),
                        .into(),
//                    ),
                    ),
//                ])
                ])
//                .add_operands(&[k1])
                .add_operands(&[k1])
//                .add_results(&[llvm::r#type::pointer(context, 0)])
                .add_results(&[llvm::r#type::pointer(context, 0)])
//                .build()?,
                .build()?,
//        )
        )
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Allocate space and write the current gas.
    // Allocate space and write the current gas.
//    let gas_builtin_ptr = helper.init_block().alloca1(
    let gas_builtin_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        IntegerType::new(context, 128).into(),
        IntegerType::new(context, 128).into(),
//        Some(get_integer_layout(128).align()),
        Some(get_integer_layout(128).align()),
//    )?;
    )?;
//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        gas_builtin_ptr,
        gas_builtin_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Allocate `p` argument and write the value.
    // Allocate `p` argument and write the value.
//    let p_arg_ptr = helper.init_block().alloca1(
    let p_arg_ptr = helper.init_block().alloca1(
//        context,
        context,
//        location,
        location,
//        llvm::r#type::r#struct(
        llvm::r#type::r#struct(
//            context,
            context,
//            &[
            &[
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//            ],
            ],
//            false,
            false,
//        ),
        ),
//        None,
        None,
//    )?;
    )?;
//

//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(2)?.into(),
        entry.argument(2)?.into(),
//        p_arg_ptr,
        p_arg_ptr,
//        location,
        location,
//        LoadStoreOptions::default(),
        LoadStoreOptions::default(),
//    ));
    ));
//

//    // Extract function pointer.
    // Extract function pointer.
//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::get_element_ptr(
        .append_operation(llvm::get_element_ptr(
//            context,
            context,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            DenseI32ArrayAttribute::new(
            DenseI32ArrayAttribute::new(
//                context,
                context,
//                &[StarknetSyscallHandlerCallbacks::<()>::SECP256R1_GET_XY.try_into()?],
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256R1_GET_XY.try_into()?],
//            ),
            ),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let fn_ptr = entry
    let fn_ptr = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            fn_ptr,
            fn_ptr,
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(
    entry.append_operation(
//        OperationBuilder::new("llvm.call", location)
        OperationBuilder::new("llvm.call", location)
//            .add_operands(&[fn_ptr, result_ptr, ptr, gas_builtin_ptr, p_arg_ptr])
            .add_operands(&[fn_ptr, result_ptr, ptr, gas_builtin_ptr, p_arg_ptr])
//            .build()?,
            .build()?,
//    );
    );
//

//    let result = entry
    let result = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            result_ptr,
            result_ptr,
//            llvm::r#type::r#struct(
            llvm::r#type::r#struct(
//                context,
                context,
//                &[
                &[
//                    result_tag_ty,
                    result_tag_ty,
//                    llvm::r#type::array(
                    llvm::r#type::array(
//                        IntegerType::new(context, 8).into(),
                        IntegerType::new(context, 8).into(),
//                        (result_layout.size() - 1).try_into()?,
                        (result_layout.size() - 1).try_into()?,
//                    ),
                    ),
//                ],
                ],
//                false,
                false,
//            ),
            ),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_tag = entry
    let result_tag = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            result,
            result,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            IntegerType::new(context, 1).into(),
            IntegerType::new(context, 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let payload_ok = {
    let payload_ok = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let value = entry
        let value = entry
//            .append_operation(llvm::load(
            .append_operation(llvm::load(
//                context,
                context,
//                ptr,
                ptr,
//                variant_tys[0].0,
                variant_tys[0].0,
//                location,
                location,
//                LoadStoreOptions::default(),
                LoadStoreOptions::default(),
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        let x_value = entry
        let x_value = entry
//            .append_operation(llvm::extract_value(
            .append_operation(llvm::extract_value(
//                context,
                context,
//                value,
                value,
//                DenseI64ArrayAttribute::new(context, &[0]),
                DenseI64ArrayAttribute::new(context, &[0]),
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let y_value = entry
        let y_value = entry
//            .append_operation(llvm::extract_value(
            .append_operation(llvm::extract_value(
//                context,
                context,
//                value,
                value,
//                DenseI64ArrayAttribute::new(context, &[1]),
                DenseI64ArrayAttribute::new(context, &[1]),
//                llvm::r#type::r#struct(
                llvm::r#type::r#struct(
//                    context,
                    context,
//                    &[
                    &[
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
//                    ],
                    ],
//                    false,
                    false,
//                ),
                ),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        (x_value, y_value)
        (x_value, y_value)
//    };
    };
//    let payload_err = {
    let payload_err = {
//        let ptr = entry
        let ptr = entry
//            .append_operation(
            .append_operation(
//                OperationBuilder::new("llvm.getelementptr", location)
                OperationBuilder::new("llvm.getelementptr", location)
//                    .add_attributes(&[
                    .add_attributes(&[
//                        (
                        (
//                            Identifier::new(context, "rawConstantIndices"),
                            Identifier::new(context, "rawConstantIndices"),
//                            DenseI32ArrayAttribute::new(
                            DenseI32ArrayAttribute::new(
//                                context,
                                context,
//                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
//                            )
                            )
//                            .into(),
                            .into(),
//                        ),
                        ),
//                        (
                        (
//                            Identifier::new(context, "elem_type"),
                            Identifier::new(context, "elem_type"),
//                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
//                        ),
                        ),
//                    ])
                    ])
//                    .add_operands(&[result_ptr])
                    .add_operands(&[result_ptr])
//                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .add_results(&[llvm::r#type::pointer(context, 0)])
//                    .build()?,
                    .build()?,
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        entry
        entry
//            .append_operation(llvm::load(
            .append_operation(llvm::load(
//                context,
                context,
//                ptr,
                ptr,
//                variant_tys[1].0,
                variant_tys[1].0,
//                location,
                location,
//                LoadStoreOptions::default(),
                LoadStoreOptions::default(),
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into()
            .into()
//    };
    };
//

//    let remaining_gas = entry
    let remaining_gas = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            gas_builtin_ptr,
            gas_builtin_ptr,
//            IntegerType::new(context, 128).into(),
            IntegerType::new(context, 128).into(),
//            location,
            location,
//            LoadStoreOptions::default(),
            LoadStoreOptions::default(),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        result_tag,
        result_tag,
//        [1, 0],
        [1, 0],
//        [
        [
//            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
//            &[
            &[
//                remaining_gas,
                remaining_gas,
//                entry.argument(1)?.into(),
                entry.argument(1)?.into(),
//                payload_ok.0,
                payload_ok.0,
//                payload_ok.1,
                payload_ok.1,
//            ],
            ],
//        ],
        ],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
