use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    libfuncs::{LibfuncBuilder, LibfuncHelper},
    metadata::MetadataStorage,
    starknet::handler::StarkNetSyscallHandlerCallbacks,
    types::TypeBuilder,
    utils::get_integer_layout,
};
use cairo_lang_sierra::{
    extensions::{
        lib_func::SignatureOnlyConcreteLibfunc,
        starknet::secp256::{Secp256ConcreteLibfunc, Secp256OpConcreteLibfunc},
        ConcreteLibfunc, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith,
        llvm::{self, LoadStoreOptions},
    },
    ir::{
        attribute::{
            DenseI32ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute, TypeAttribute,
        },
        operation::OperationBuilder,
        r#type::IntegerType,
        Block, Identifier, Location,
    },
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Secp256ConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        Secp256ConcreteLibfunc::K1(selector) => match selector {
            Secp256OpConcreteLibfunc::New(info) => {
                build_k1_new(context, registry, entry, location, helper, metadata, info)
            }
            Secp256OpConcreteLibfunc::Add(info) => {
                build_k1_add(context, registry, entry, location, helper, metadata, info)
            }
            Secp256OpConcreteLibfunc::Mul(info) => {
                build_k1_mul(context, registry, entry, location, helper, metadata, info)
            }
            Secp256OpConcreteLibfunc::GetPointFromX(info) => build_k1_get_point_from_x(
                context, registry, entry, location, helper, metadata, info,
            ),
            Secp256OpConcreteLibfunc::GetXy(info) => {
                build_k1_get_xy(context, registry, entry, location, helper, metadata, info)
            }
        },
        Secp256ConcreteLibfunc::R1(selector) => match selector {
            Secp256OpConcreteLibfunc::New(info) => {
                build_r1_new(context, registry, entry, location, helper, metadata, info)
            }
            Secp256OpConcreteLibfunc::Add(info) => {
                build_r1_add(context, registry, entry, location, helper, metadata, info)
            }
            Secp256OpConcreteLibfunc::Mul(info) => {
                build_r1_mul(context, registry, entry, location, helper, metadata, info)
            }
            Secp256OpConcreteLibfunc::GetPointFromX(info) => build_r1_get_point_from_x(
                context, registry, entry, location, helper, metadata, info,
            ),
            Secp256OpConcreteLibfunc::GetXy(info) => {
                build_r1_get_xy(context, registry, entry, location, helper, metadata, info)
            }
        },
    }
}

pub fn build_k1_new<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
        .init_block()
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
            location,
        ))
        .result(0)?
        .into();
    let result_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[
                    (
                        Identifier::new(context, "alignment"),
                        IntegerAttribute::new(
                            result_layout.align().try_into()?,
                            IntegerType::new(context, 64).into(),
                        )
                        .into(),
                    ),
                    (
                        Identifier::new(context, "elem_type"),
                        TypeAttribute::new(llvm::r#type::r#struct(
                            context,
                            &[
                                result_tag_ty,
                                llvm::r#type::array(
                                    IntegerType::new(context, 8).into(),
                                    (result_layout.size() - 1).try_into()?,
                                ),
                            ],
                            false,
                        ))
                        .into(),
                    ),
                ])
                .add_operands(&[k1])
                .add_results(&[llvm::r#type::opaque_pointer(context)])
                .build()?,
        )
        .result(0)?
        .into();

    // Allocate space and write the current gas.
    let gas_builtin_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        result_layout.align().try_into()?,
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[llvm::r#type::pointer(
                    IntegerType::new(context, 128).into(),
                    0,
                )])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let i128_ty = IntegerType::new(context, 128).into();

    // Allocate `x` argument and write the value.
    let x_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
        0,
    );
    let x_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        get_integer_layout(128).align().try_into().unwrap(),
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[x_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        x_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `y` argument and write the value.
    let y_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
        0,
    );
    let y_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        get_integer_layout(128).align().try_into().unwrap(),
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[y_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        y_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Extract function pointer.
    let fn_ptr_ty = llvm::r#type::function(
        llvm::r#type::void(context),
        &[
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::pointer(IntegerType::new(context, 128).into(), 0),
            x_arg_ptr_ty,
            y_arg_ptr_ty,
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarkNetSyscallHandlerCallbacks::<()>::SECP256K1_NEW.try_into()?],
            ),
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::opaque_pointer(context),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(fn_ptr_ty, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                x_arg_ptr,
                y_arg_ptr,
            ])
            .build()?,
    );

    let result = entry
        .append_operation(llvm::load(
            context,
            result_ptr,
            llvm::r#type::r#struct(
                context,
                &[
                    result_tag_ty,
                    llvm::r#type::array(
                        IntegerType::new(context, 8).into(),
                        (result_layout.size() - 1).try_into()?,
                    ),
                ],
                false,
            ),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();
    let result_tag = entry
        .append_operation(llvm::extract_value(
            context,
            result,
            DenseI64ArrayAttribute::new(context, &[0]),
            IntegerType::new(context, 1).into(),
            location,
        ))
        .result(0)?
        .into();

    let payload_ok = {
        let ptr = entry
            .append_operation(
                OperationBuilder::new("llvm.getelementptr", location)
                    .add_attributes(&[
                        (
                            Identifier::new(context, "rawConstantIndices"),
                            DenseI32ArrayAttribute::new(
                                context,
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
                            )
                            .into(),
                        ),
                        (
                            Identifier::new(context, "elem_type"),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                        ),
                    ])
                    .add_operands(&[result_ptr])
                    .add_results(&[llvm::r#type::opaque_pointer(context)])
                    .build()?,
            )
            .result(0)?
            .into();
        // entry
        //     .append_operation(llvm::load(
        //         context,
        //         ptr,
        //         variant_tys[0].0,
        //         location,
        //         LoadStoreOptions::default(),
        //     ))
        //     .result(0)?
        //     .into()
        ptr
    };
    let payload_err = {
        let ptr = entry
            .append_operation(
                OperationBuilder::new("llvm.getelementptr", location)
                    .add_attributes(&[
                        (
                            Identifier::new(context, "rawConstantIndices"),
                            DenseI32ArrayAttribute::new(
                                context,
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
                            )
                            .into(),
                        ),
                        (
                            Identifier::new(context, "elem_type"),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                        ),
                    ])
                    .add_operands(&[result_ptr])
                    .add_results(&[llvm::r#type::opaque_pointer(context)])
                    .build()?,
            )
            .result(0)?
            .into();
        entry
            .append_operation(llvm::load(
                context,
                ptr,
                variant_tys[1].0,
                location,
                LoadStoreOptions::default(),
            ))
            .result(0)?
            .into()
    };

    let remaining_gas = entry
        .append_operation(llvm::load(
            context,
            gas_builtin_ptr,
            IntegerType::new(context, 128).into(),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_k1_add<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
        .init_block()
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
            location,
        ))
        .result(0)?
        .into();
    let result_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[
                    (
                        Identifier::new(context, "alignment"),
                        IntegerAttribute::new(
                            result_layout.align().try_into()?,
                            IntegerType::new(context, 64).into(),
                        )
                        .into(),
                    ),
                    (
                        Identifier::new(context, "elem_type"),
                        TypeAttribute::new(llvm::r#type::r#struct(
                            context,
                            &[
                                result_tag_ty,
                                llvm::r#type::array(
                                    IntegerType::new(context, 8).into(),
                                    (result_layout.size() - 1).try_into()?,
                                ),
                            ],
                            false,
                        ))
                        .into(),
                    ),
                ])
                .add_operands(&[k1])
                .add_results(&[llvm::r#type::opaque_pointer(context)])
                .build()?,
        )
        .result(0)?
        .into();

    // Allocate space and write the current gas.
    let gas_builtin_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        result_layout.align().try_into()?,
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[llvm::r#type::pointer(
                    IntegerType::new(context, 128).into(),
                    0,
                )])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `p0` argument and write the value.
    let p0_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(
            context,
            &[
                llvm::r#type::r#struct(
                    context,
                    &[
                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
                    ],
                    false,
                ),
                llvm::r#type::r#struct(
                    context,
                    &[
                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
                    ],
                    false,
                ),
            ],
            false,
        ),
        0,
    );
    let p0_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        get_integer_layout(128).align().try_into().unwrap(),
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[p0_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        p0_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `p1` argument and write the value.
    let p1_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(
            context,
            &[
                llvm::r#type::r#struct(
                    context,
                    &[
                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
                    ],
                    false,
                ),
                llvm::r#type::r#struct(
                    context,
                    &[
                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
                    ],
                    false,
                ),
            ],
            false,
        ),
        0,
    );
    let p1_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        get_integer_layout(128).align().try_into().unwrap(),
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[p0_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        p1_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Extract function pointer.
    let fn_ptr_ty = llvm::r#type::function(
        llvm::r#type::void(context),
        &[
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::pointer(IntegerType::new(context, 128).into(), 0),
            p0_arg_ptr_ty,
            p1_arg_ptr_ty,
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarkNetSyscallHandlerCallbacks::<()>::SECP256K1_ADD.try_into()?],
            ),
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::opaque_pointer(context),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(fn_ptr_ty, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                p0_arg_ptr,
                p1_arg_ptr,
            ])
            .build()?,
    );

    let result = entry
        .append_operation(llvm::load(
            context,
            result_ptr,
            llvm::r#type::r#struct(
                context,
                &[
                    result_tag_ty,
                    llvm::r#type::array(
                        IntegerType::new(context, 8).into(),
                        (result_layout.size() - 1).try_into()?,
                    ),
                ],
                false,
            ),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();
    let result_tag = entry
        .append_operation(llvm::extract_value(
            context,
            result,
            DenseI64ArrayAttribute::new(context, &[0]),
            IntegerType::new(context, 1).into(),
            location,
        ))
        .result(0)?
        .into();

    let payload_ok = {
        let ptr = entry
            .append_operation(
                OperationBuilder::new("llvm.getelementptr", location)
                    .add_attributes(&[
                        (
                            Identifier::new(context, "rawConstantIndices"),
                            DenseI32ArrayAttribute::new(
                                context,
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
                            )
                            .into(),
                        ),
                        (
                            Identifier::new(context, "elem_type"),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                        ),
                    ])
                    .add_operands(&[result_ptr])
                    .add_results(&[llvm::r#type::opaque_pointer(context)])
                    .build()?,
            )
            .result(0)?
            .into();
        entry
            .append_operation(llvm::load(
                context,
                ptr,
                variant_tys[0].0,
                location,
                LoadStoreOptions::default(),
            ))
            .result(0)?
            .into()
    };
    let payload_err = {
        let ptr = entry
            .append_operation(
                OperationBuilder::new("llvm.getelementptr", location)
                    .add_attributes(&[
                        (
                            Identifier::new(context, "rawConstantIndices"),
                            DenseI32ArrayAttribute::new(
                                context,
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
                            )
                            .into(),
                        ),
                        (
                            Identifier::new(context, "elem_type"),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                        ),
                    ])
                    .add_operands(&[result_ptr])
                    .add_results(&[llvm::r#type::opaque_pointer(context)])
                    .build()?,
            )
            .result(0)?
            .into();
        entry
            .append_operation(llvm::load(
                context,
                ptr,
                variant_tys[1].0,
                location,
                LoadStoreOptions::default(),
            ))
            .result(0)?
            .into()
    };

    let remaining_gas = entry
        .append_operation(llvm::load(
            context,
            gas_builtin_ptr,
            IntegerType::new(context, 128).into(),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_k1_mul<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
        .init_block()
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
            location,
        ))
        .result(0)?
        .into();
    let result_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[
                    (
                        Identifier::new(context, "alignment"),
                        IntegerAttribute::new(
                            result_layout.align().try_into()?,
                            IntegerType::new(context, 64).into(),
                        )
                        .into(),
                    ),
                    (
                        Identifier::new(context, "elem_type"),
                        TypeAttribute::new(llvm::r#type::r#struct(
                            context,
                            &[
                                result_tag_ty,
                                llvm::r#type::array(
                                    IntegerType::new(context, 8).into(),
                                    (result_layout.size() - 1).try_into()?,
                                ),
                            ],
                            false,
                        ))
                        .into(),
                    ),
                ])
                .add_operands(&[k1])
                .add_results(&[llvm::r#type::opaque_pointer(context)])
                .build()?,
        )
        .result(0)?
        .into();

    // Allocate space and write the current gas.
    let gas_builtin_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        result_layout.align().try_into()?,
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[llvm::r#type::pointer(
                    IntegerType::new(context, 128).into(),
                    0,
                )])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `p` argument and write the value.
    let p_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(
            context,
            &[
                llvm::r#type::r#struct(
                    context,
                    &[
                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
                    ],
                    false,
                ),
                llvm::r#type::r#struct(
                    context,
                    &[
                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
                    ],
                    false,
                ),
            ],
            false,
        ),
        0,
    );
    let p_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        get_integer_layout(128).align().try_into().unwrap(),
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[p_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        p_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `scalar` argument and write the value.
    let scalar_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(
            context,
            &[
                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
            ],
            false,
        ),
        0,
    );
    let scalar_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        get_integer_layout(128).align().try_into().unwrap(),
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[scalar_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        scalar_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Extract function pointer.
    let fn_ptr_ty = llvm::r#type::function(
        llvm::r#type::void(context),
        &[
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::pointer(IntegerType::new(context, 128).into(), 0),
            p_arg_ptr_ty,
            scalar_arg_ptr_ty,
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarkNetSyscallHandlerCallbacks::<()>::SECP256K1_MUL.try_into()?],
            ),
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::opaque_pointer(context),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(fn_ptr_ty, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                p_arg_ptr,
                scalar_arg_ptr,
            ])
            .build()?,
    );

    let result = entry
        .append_operation(llvm::load(
            context,
            result_ptr,
            llvm::r#type::r#struct(
                context,
                &[
                    result_tag_ty,
                    llvm::r#type::array(
                        IntegerType::new(context, 8).into(),
                        (result_layout.size() - 1).try_into()?,
                    ),
                ],
                false,
            ),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();
    let result_tag = entry
        .append_operation(llvm::extract_value(
            context,
            result,
            DenseI64ArrayAttribute::new(context, &[0]),
            IntegerType::new(context, 1).into(),
            location,
        ))
        .result(0)?
        .into();

    let payload_ok = {
        let ptr = entry
            .append_operation(
                OperationBuilder::new("llvm.getelementptr", location)
                    .add_attributes(&[
                        (
                            Identifier::new(context, "rawConstantIndices"),
                            DenseI32ArrayAttribute::new(
                                context,
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
                            )
                            .into(),
                        ),
                        (
                            Identifier::new(context, "elem_type"),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                        ),
                    ])
                    .add_operands(&[result_ptr])
                    .add_results(&[llvm::r#type::opaque_pointer(context)])
                    .build()?,
            )
            .result(0)?
            .into();
        entry
            .append_operation(llvm::load(
                context,
                ptr,
                variant_tys[0].0,
                location,
                LoadStoreOptions::default(),
            ))
            .result(0)?
            .into()
    };
    let payload_err = {
        let ptr = entry
            .append_operation(
                OperationBuilder::new("llvm.getelementptr", location)
                    .add_attributes(&[
                        (
                            Identifier::new(context, "rawConstantIndices"),
                            DenseI32ArrayAttribute::new(
                                context,
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
                            )
                            .into(),
                        ),
                        (
                            Identifier::new(context, "elem_type"),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                        ),
                    ])
                    .add_operands(&[result_ptr])
                    .add_results(&[llvm::r#type::opaque_pointer(context)])
                    .build()?,
            )
            .result(0)?
            .into();
        entry
            .append_operation(llvm::load(
                context,
                ptr,
                variant_tys[1].0,
                location,
                LoadStoreOptions::default(),
            ))
            .result(0)?
            .into()
    };

    let remaining_gas = entry
        .append_operation(llvm::load(
            context,
            gas_builtin_ptr,
            IntegerType::new(context, 128).into(),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_k1_get_point_from_x<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
        .init_block()
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
            location,
        ))
        .result(0)?
        .into();
    let result_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[
                    (
                        Identifier::new(context, "alignment"),
                        IntegerAttribute::new(
                            result_layout.align().try_into()?,
                            IntegerType::new(context, 64).into(),
                        )
                        .into(),
                    ),
                    (
                        Identifier::new(context, "elem_type"),
                        TypeAttribute::new(llvm::r#type::r#struct(
                            context,
                            &[
                                result_tag_ty,
                                llvm::r#type::array(
                                    IntegerType::new(context, 8).into(),
                                    (result_layout.size() - 1).try_into()?,
                                ),
                            ],
                            false,
                        ))
                        .into(),
                    ),
                ])
                .add_operands(&[k1])
                .add_results(&[llvm::r#type::opaque_pointer(context)])
                .build()?,
        )
        .result(0)?
        .into();

    // Allocate space and write the current gas.
    let gas_builtin_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        result_layout.align().try_into()?,
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[llvm::r#type::pointer(
                    IntegerType::new(context, 128).into(),
                    0,
                )])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `x` argument and write the value.
    let x_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(
            context,
            &[
                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
            ],
            false,
        ),
        0,
    );
    let x_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        get_integer_layout(128).align().try_into().unwrap(),
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[x_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        x_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `y_parity` argument and write the value.
    let y_parity_arg_ptr_ty = llvm::r#type::pointer(IntegerType::new(context, 1).into(), 0);
    let y_parity_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_operands(&[k1])
                .add_results(&[y_parity_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        y_parity_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Extract function pointer.
    let fn_ptr_ty = llvm::r#type::function(
        llvm::r#type::void(context),
        &[
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::pointer(IntegerType::new(context, 128).into(), 0),
            x_arg_ptr_ty,
            y_parity_arg_ptr_ty,
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarkNetSyscallHandlerCallbacks::<()>::SECP256K1_GET_POINT_FROM_X.try_into()?],
            ),
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::opaque_pointer(context),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(fn_ptr_ty, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                x_arg_ptr,
                y_parity_arg_ptr,
            ])
            .build()?,
    );

    let result = entry
        .append_operation(llvm::load(
            context,
            result_ptr,
            llvm::r#type::r#struct(
                context,
                &[
                    result_tag_ty,
                    llvm::r#type::array(
                        IntegerType::new(context, 8).into(),
                        (result_layout.size() - 1).try_into()?,
                    ),
                ],
                false,
            ),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();
    let result_tag = entry
        .append_operation(llvm::extract_value(
            context,
            result,
            DenseI64ArrayAttribute::new(context, &[0]),
            IntegerType::new(context, 1).into(),
            location,
        ))
        .result(0)?
        .into();

    let payload_ok = {
        let ptr = entry
            .append_operation(
                OperationBuilder::new("llvm.getelementptr", location)
                    .add_attributes(&[
                        (
                            Identifier::new(context, "rawConstantIndices"),
                            DenseI32ArrayAttribute::new(
                                context,
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
                            )
                            .into(),
                        ),
                        (
                            Identifier::new(context, "elem_type"),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                        ),
                    ])
                    .add_operands(&[result_ptr])
                    .add_results(&[llvm::r#type::opaque_pointer(context)])
                    .build()?,
            )
            .result(0)?
            .into();
        // entry
        //     .append_operation(llvm::load(
        //         context,
        //         ptr,
        //         variant_tys[0].0,
        //         location,
        //         LoadStoreOptions::default(),
        //     ))
        //     .result(0)?
        //     .into()
        ptr
    };
    let payload_err = {
        let ptr = entry
            .append_operation(
                OperationBuilder::new("llvm.getelementptr", location)
                    .add_attributes(&[
                        (
                            Identifier::new(context, "rawConstantIndices"),
                            DenseI32ArrayAttribute::new(
                                context,
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
                            )
                            .into(),
                        ),
                        (
                            Identifier::new(context, "elem_type"),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                        ),
                    ])
                    .add_operands(&[result_ptr])
                    .add_results(&[llvm::r#type::opaque_pointer(context)])
                    .build()?,
            )
            .result(0)?
            .into();
        entry
            .append_operation(llvm::load(
                context,
                ptr,
                variant_tys[1].0,
                location,
                LoadStoreOptions::default(),
            ))
            .result(0)?
            .into()
    };

    let remaining_gas = entry
        .append_operation(llvm::load(
            context,
            gas_builtin_ptr,
            IntegerType::new(context, 128).into(),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_k1_get_xy<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
        .init_block()
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
            location,
        ))
        .result(0)?
        .into();
    let result_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[
                    (
                        Identifier::new(context, "alignment"),
                        IntegerAttribute::new(
                            result_layout.align().try_into()?,
                            IntegerType::new(context, 64).into(),
                        )
                        .into(),
                    ),
                    (
                        Identifier::new(context, "elem_type"),
                        TypeAttribute::new(llvm::r#type::r#struct(
                            context,
                            &[
                                result_tag_ty,
                                llvm::r#type::array(
                                    IntegerType::new(context, 8).into(),
                                    (result_layout.size() - 1).try_into()?,
                                ),
                            ],
                            false,
                        ))
                        .into(),
                    ),
                ])
                .add_operands(&[k1])
                .add_results(&[llvm::r#type::opaque_pointer(context)])
                .build()?,
        )
        .result(0)?
        .into();

    // Allocate space and write the current gas.
    let gas_builtin_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        result_layout.align().try_into()?,
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[llvm::r#type::pointer(
                    IntegerType::new(context, 128).into(),
                    0,
                )])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `p` argument and write the value.
    let p_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(
            context,
            &[
                llvm::r#type::r#struct(
                    context,
                    &[
                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
                    ],
                    false,
                ),
                llvm::r#type::r#struct(
                    context,
                    &[
                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
                    ],
                    false,
                ),
            ],
            false,
        ),
        0,
    );
    let p_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        get_integer_layout(128).align().try_into().unwrap(),
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[p_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        p_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Extract function pointer.
    let fn_ptr_ty = llvm::r#type::function(
        llvm::r#type::void(context),
        &[
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::pointer(IntegerType::new(context, 128).into(), 0),
            p_arg_ptr_ty,
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarkNetSyscallHandlerCallbacks::<()>::SECP256K1_GET_POINT_FROM_X.try_into()?],
            ),
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::opaque_pointer(context),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(fn_ptr_ty, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[fn_ptr, result_ptr, ptr, gas_builtin_ptr, p_arg_ptr])
            .build()?,
    );

    let result = entry
        .append_operation(llvm::load(
            context,
            result_ptr,
            llvm::r#type::r#struct(
                context,
                &[
                    result_tag_ty,
                    llvm::r#type::array(
                        IntegerType::new(context, 8).into(),
                        (result_layout.size() - 1).try_into()?,
                    ),
                ],
                false,
            ),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();
    let result_tag = entry
        .append_operation(llvm::extract_value(
            context,
            result,
            DenseI64ArrayAttribute::new(context, &[0]),
            IntegerType::new(context, 1).into(),
            location,
        ))
        .result(0)?
        .into();

    let payload_ok = {
        let ptr = entry
            .append_operation(
                OperationBuilder::new("llvm.getelementptr", location)
                    .add_attributes(&[
                        (
                            Identifier::new(context, "rawConstantIndices"),
                            DenseI32ArrayAttribute::new(
                                context,
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
                            )
                            .into(),
                        ),
                        (
                            Identifier::new(context, "elem_type"),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                        ),
                    ])
                    .add_operands(&[result_ptr])
                    .add_results(&[llvm::r#type::opaque_pointer(context)])
                    .build()?,
            )
            .result(0)?
            .into();
        let value = entry
            .append_operation(llvm::load(
                context,
                ptr,
                variant_tys[0].0,
                location,
                LoadStoreOptions::default(),
            ))
            .result(0)?
            .into();

        let x_value = entry
            .append_operation(llvm::extract_value(
                context,
                value,
                DenseI64ArrayAttribute::new(context, &[0]),
                llvm::r#type::r#struct(
                    context,
                    &[
                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
                    ],
                    false,
                ),
                location,
            ))
            .result(0)?
            .into();
        let y_value = entry
            .append_operation(llvm::extract_value(
                context,
                value,
                DenseI64ArrayAttribute::new(context, &[0]),
                llvm::r#type::r#struct(
                    context,
                    &[
                        IntegerType::new(context, 128).into(),
                        IntegerType::new(context, 128).into(),
                    ],
                    false,
                ),
                location,
            ))
            .result(0)?
            .into();

        (x_value, y_value)
    };
    let payload_err = {
        let ptr = entry
            .append_operation(
                OperationBuilder::new("llvm.getelementptr", location)
                    .add_attributes(&[
                        (
                            Identifier::new(context, "rawConstantIndices"),
                            DenseI32ArrayAttribute::new(
                                context,
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
                            )
                            .into(),
                        ),
                        (
                            Identifier::new(context, "elem_type"),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                        ),
                    ])
                    .add_operands(&[result_ptr])
                    .add_results(&[llvm::r#type::opaque_pointer(context)])
                    .build()?,
            )
            .result(0)?
            .into();
        entry
            .append_operation(llvm::load(
                context,
                ptr,
                variant_tys[1].0,
                location,
                LoadStoreOptions::default(),
            ))
            .result(0)?
            .into()
    };

    let remaining_gas = entry
        .append_operation(llvm::load(
            context,
            gas_builtin_ptr,
            IntegerType::new(context, 128).into(),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[
                remaining_gas,
                entry.argument(1)?.into(),
                payload_ok.0,
                payload_ok.1,
            ],
        ],
        location,
    ));
    Ok(())
}

pub fn build_r1_new<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
        .init_block()
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
            location,
        ))
        .result(0)?
        .into();
    let result_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[
                    (
                        Identifier::new(context, "alignment"),
                        IntegerAttribute::new(
                            result_layout.align().try_into()?,
                            IntegerType::new(context, 64).into(),
                        )
                        .into(),
                    ),
                    (
                        Identifier::new(context, "elem_type"),
                        TypeAttribute::new(llvm::r#type::r#struct(
                            context,
                            &[
                                result_tag_ty,
                                llvm::r#type::array(
                                    IntegerType::new(context, 8).into(),
                                    (result_layout.size() - 1).try_into()?,
                                ),
                            ],
                            false,
                        ))
                        .into(),
                    ),
                ])
                .add_operands(&[k1])
                .add_results(&[llvm::r#type::opaque_pointer(context)])
                .build()?,
        )
        .result(0)?
        .into();

    // Allocate space and write the current gas.
    let gas_builtin_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        result_layout.align().try_into()?,
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[llvm::r#type::pointer(
                    IntegerType::new(context, 128).into(),
                    0,
                )])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let i128_ty = IntegerType::new(context, 128).into();

    // Allocate `x` argument and write the value.
    let x_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
        0,
    );
    let x_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        get_integer_layout(128).align().try_into().unwrap(),
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[x_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        x_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `y` argument and write the value.
    let y_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
        0,
    );
    let y_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        get_integer_layout(128).align().try_into().unwrap(),
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[y_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        y_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Extract function pointer.
    let fn_ptr_ty = llvm::r#type::function(
        llvm::r#type::void(context),
        &[
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::pointer(IntegerType::new(context, 128).into(), 0),
            x_arg_ptr_ty,
            y_arg_ptr_ty,
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarkNetSyscallHandlerCallbacks::<()>::SECP256R1_NEW.try_into()?],
            ),
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::opaque_pointer(context),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(fn_ptr_ty, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                x_arg_ptr,
                y_arg_ptr,
            ])
            .build()?,
    );

    let result = entry
        .append_operation(llvm::load(
            context,
            result_ptr,
            llvm::r#type::r#struct(
                context,
                &[
                    result_tag_ty,
                    llvm::r#type::array(
                        IntegerType::new(context, 8).into(),
                        (result_layout.size() - 1).try_into()?,
                    ),
                ],
                false,
            ),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();
    let result_tag = entry
        .append_operation(llvm::extract_value(
            context,
            result,
            DenseI64ArrayAttribute::new(context, &[0]),
            IntegerType::new(context, 1).into(),
            location,
        ))
        .result(0)?
        .into();

    let payload_ok = {
        let ptr = entry
            .append_operation(
                OperationBuilder::new("llvm.getelementptr", location)
                    .add_attributes(&[
                        (
                            Identifier::new(context, "rawConstantIndices"),
                            DenseI32ArrayAttribute::new(
                                context,
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
                            )
                            .into(),
                        ),
                        (
                            Identifier::new(context, "elem_type"),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                        ),
                    ])
                    .add_operands(&[result_ptr])
                    .add_results(&[llvm::r#type::opaque_pointer(context)])
                    .build()?,
            )
            .result(0)?
            .into();
        entry
            .append_operation(llvm::load(
                context,
                ptr,
                variant_tys[0].0,
                location,
                LoadStoreOptions::default(),
            ))
            .result(0)?
            .into()
    };
    let payload_err = {
        let ptr = entry
            .append_operation(
                OperationBuilder::new("llvm.getelementptr", location)
                    .add_attributes(&[
                        (
                            Identifier::new(context, "rawConstantIndices"),
                            DenseI32ArrayAttribute::new(
                                context,
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
                            )
                            .into(),
                        ),
                        (
                            Identifier::new(context, "elem_type"),
                            TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                        ),
                    ])
                    .add_operands(&[result_ptr])
                    .add_results(&[llvm::r#type::opaque_pointer(context)])
                    .build()?,
            )
            .result(0)?
            .into();
        entry
            .append_operation(llvm::load(
                context,
                ptr,
                variant_tys[1].0,
                location,
                LoadStoreOptions::default(),
            ))
            .result(0)?
            .into()
    };

    let remaining_gas = entry
        .append_operation(llvm::load(
            context,
            gas_builtin_ptr,
            IntegerType::new(context, 128).into(),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_r1_add<'ctx, 'this, TType, TLibfunc>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _entry: &'this Block<'ctx>,
    _location: Location<'ctx>,
    _helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    todo!()
}

pub fn build_r1_mul<'ctx, 'this, TType, TLibfunc>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _entry: &'this Block<'ctx>,
    _location: Location<'ctx>,
    _helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    todo!()
}

pub fn build_r1_get_point_from_x<'ctx, 'this, TType, TLibfunc>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _entry: &'this Block<'ctx>,
    _location: Location<'ctx>,
    _helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    todo!()
}

pub fn build_r1_get_xy<'ctx, 'this, TType, TLibfunc>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _entry: &'this Block<'ctx>,
    _location: Location<'ctx>,
    _helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    todo!()
}
