//! # Function call libfuncs
//!
//! Includes logic for handling direct tail recursive function calls. More information on this topic
//! at the [tail recursive metadata](crate::metadata::tail_recursion).

use super::LibfuncHelper;
use crate::{
    error::builders::Result,
    metadata::{tail_recursion::TailRecursionMeta, MetadataStorage},
    types::TypeBuilder,
    utils::generate_function_name,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        function_call::FunctionCallConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith, cf, func, index,
        llvm::{self, AllocaOptions, LoadStoreOptions},
        memref,
    },
    ir::{
        attribute::{
            DenseI32ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute, TypeAttribute,
        },
        r#type::IntegerType,
        Block, Location, Type, Value,
    },
    Context,
};
use std::alloc::Layout;

/// Generate MLIR operations for the `function_call` libfunc.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &FunctionCallConcreteLibfunc,
) -> Result<()> {
    let mut arguments = Vec::new();
    let mut result_types = Vec::new();

    for (idx, type_id) in info.function.signature.param_types.iter().enumerate() {
        let type_info = registry.get_type(type_id)?;

        if !(type_info.is_builtin() && type_info.is_zst(registry)) {
            arguments.push(entry.argument(idx)?.into());
        }
    }

    let return_types = info
        .function
        .signature
        .ret_types
        .iter()
        .filter_map(|type_id| {
            let type_info = registry.get_type(type_id).unwrap();
            if type_info.is_builtin() && type_info.is_zst(registry) {
                None
            } else {
                Some((type_id, type_info))
            }
        })
        .collect::<Vec<_>>();
    let has_return_ptr = if return_types.len() > 1 {
        result_types.extend(
            return_types
                .iter()
                .map(|(type_id, type_info)| {
                    type_info.build(context, helper, registry, metadata, type_id)
                })
                .collect::<std::result::Result<Vec<_>, _>>()?,
        );

        Some(false)
    } else if return_types
        .first()
        .is_some_and(|(_, type_info)| type_info.is_memory_allocated(registry))
    {
        let (type_id, type_info) = return_types[0];
        let layout = type_info.layout(registry)?;

        let k1 = helper
            .init_block()
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
                location,
            ))
            .result(0)?
            .into();
        let stack_ptr = helper
            .init_block()
            .append_operation(llvm::alloca(
                context,
                k1,
                llvm::r#type::opaque_pointer(context),
                location,
                AllocaOptions::new()
                    .align(Some(IntegerAttribute::new(
                        layout.align() as i64,
                        IntegerType::new(context, 64).into(),
                    )))
                    .elem_type(Some(TypeAttribute::new(
                        type_info.build(context, helper, registry, metadata, type_id)?,
                    ))),
            ))
            .result(0)?
            .into();

        arguments.insert(0, stack_ptr);

        Some(true)
    } else {
        let (type_id, type_info) = return_types[0];
        result_types.push(type_info.build(context, helper, registry, metadata, type_id)?);

        None
    };

    if let Some(tailrec_meta) = metadata.get_mut::<TailRecursionMeta>() {
        let op0 = entry.append_operation(memref::load(tailrec_meta.depth_counter(), &[], location));
        let op1 = entry.append_operation(index::constant(
            context,
            IntegerAttribute::new(1, Type::index(context)),
            location,
        ));
        let op2 = entry.append_operation(index::add(
            op0.result(0)?.into(),
            op1.result(0)?.into(),
            location,
        ));
        entry.append_operation(memref::store(
            op2.result(0)?.into(),
            tailrec_meta.depth_counter(),
            &[],
            location,
        ));

        entry.append_operation(cf::br(
            &tailrec_meta.recursion_target(),
            &arguments
                .iter()
                .skip(has_return_ptr.is_some_and(|x| x) as usize)
                .copied()
                .collect::<Vec<_>>(),
            location,
        ));

        let cont_block = helper.append_block(Block::new(
            &result_types
                .iter()
                .copied()
                .map(|ty| (ty, location))
                .collect::<Vec<_>>(),
        ));
        tailrec_meta.set_return_target(cont_block);

        let mut results = Vec::<Value>::new();
        let mut count = 0;
        for var_info in &info.signature.branch_signatures[0].vars {
            let type_info = registry.get_type(&var_info.ty)?;

            if type_info.is_builtin() && type_info.is_zst(registry) {
                results.push(
                    cont_block
                        .append_operation(llvm::undef(
                            type_info.build(context, helper, registry, metadata, &var_info.ty)?,
                            location,
                        ))
                        .result(0)?
                        .into(),
                );
            } else {
                let val = cont_block.argument(count)?.into();
                count += 1;

                results.push(if type_info.is_memory_allocated(registry) {
                    let ty = type_info.build(context, helper, registry, metadata, &var_info.ty)?;
                    let layout = type_info.layout(registry)?;

                    let k1 = helper
                        .init_block()
                        .append_operation(arith::constant(
                            context,
                            IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
                            location,
                        ))
                        .result(0)?
                        .into();
                    let stack_ptr = helper
                        .init_block()
                        .append_operation(llvm::alloca(
                            context,
                            k1,
                            llvm::r#type::opaque_pointer(context),
                            location,
                            AllocaOptions::new()
                                .align(Some(IntegerAttribute::new(
                                    layout.align() as i64,
                                    IntegerType::new(context, 64).into(),
                                )))
                                .elem_type(Some(TypeAttribute::new(ty))),
                        ))
                        .result(0)?
                        .into();

                    cont_block.append_operation(llvm::store(
                        context,
                        val,
                        stack_ptr,
                        location,
                        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                            layout.align() as i64,
                            IntegerType::new(context, 64).into(),
                        ))),
                    ));

                    stack_ptr
                } else {
                    val
                });
            }
        }

        cont_block.append_operation(helper.br(0, &results, location));
    } else {
        let op0 = entry.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, &generate_function_name(&info.function.id)),
            &arguments,
            &result_types,
            location,
        ));

        let mut results = Vec::new();
        match has_return_ptr {
            Some(true) => {
                // Manual return type.

                let mut layout = Layout::new::<()>();
                for (idx, type_id) in info.function.signature.ret_types.iter().enumerate() {
                    let type_info = registry.get_type(type_id)?;

                    if type_info.is_builtin() && type_info.is_zst(registry) {
                        results.push(entry.argument(idx)?.into());
                    } else {
                        let val = arguments[0];

                        let offset;
                        let ret_layout = type_info.layout(registry)?;
                        (layout, offset) = layout.extend(ret_layout)?;

                        let pointer_val = entry
                            .append_operation(llvm::get_element_ptr(
                                context,
                                val,
                                DenseI32ArrayAttribute::new(context, &[offset as i32]),
                                IntegerType::new(context, 8).into(),
                                llvm::r#type::opaque_pointer(context),
                                location,
                            ))
                            .result(0)?
                            .into();

                        results.push(if type_info.is_memory_allocated(registry) {
                            pointer_val
                        } else {
                            entry
                                .append_operation(llvm::load(
                                    context,
                                    pointer_val,
                                    type_info
                                        .build(context, helper, registry, metadata, type_id)?,
                                    location,
                                    LoadStoreOptions::new(),
                                ))
                                .result(0)?
                                .into()
                        });
                    }
                }
            }
            Some(false) => {
                // Complex return type. Just extract the values from the struct, since LLVM will
                // handle the rest.

                let mut count = 0;
                for (idx, type_id) in info.function.signature.ret_types.iter().enumerate() {
                    let type_info = registry.get_type(type_id)?;

                    if type_info.is_builtin() && type_info.is_zst(registry) {
                        results.push(entry.argument(idx)?.into());
                    } else {
                        let val = op0.result(count)?.into();
                        count += 1;

                        results.push(if type_info.is_memory_allocated(registry) {
                            let ty =
                                type_info.build(context, helper, registry, metadata, type_id)?;
                            let layout = type_info.layout(registry)?;

                            let k1 = helper
                                .init_block()
                                .append_operation(arith::constant(
                                    context,
                                    IntegerAttribute::new(1, IntegerType::new(context, 64).into())
                                        .into(),
                                    location,
                                ))
                                .result(0)?
                                .into();
                            let stack_ptr = helper
                                .init_block()
                                .append_operation(llvm::alloca(
                                    context,
                                    k1,
                                    llvm::r#type::opaque_pointer(context),
                                    location,
                                    AllocaOptions::new()
                                        .align(Some(IntegerAttribute::new(
                                            layout.align() as i64,
                                            IntegerType::new(context, 64).into(),
                                        )))
                                        .elem_type(Some(TypeAttribute::new(ty))),
                                ))
                                .result(0)?
                                .into();

                            entry.append_operation(llvm::store(
                                context,
                                val,
                                stack_ptr,
                                location,
                                LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                                    layout.align() as i64,
                                    IntegerType::new(context, 64).into(),
                                ))),
                            ));

                            stack_ptr
                        } else {
                            val
                        });
                    }
                }
            }
            None => {
                // Returned data is simple.

                let mut count = 0;
                for (idx, type_id) in info.function.signature.ret_types.iter().enumerate() {
                    let type_info = registry.get_type(type_id)?;
                    assert!(!type_info.is_memory_allocated(registry));

                    if type_info.is_builtin() && type_info.is_zst(registry) {
                        results.push(entry.argument(idx)?.into());
                    } else {
                        let value = op0.result(count)?.into();
                        count += 1;

                        results.push(value);
                    }
                }
            }
        }

        entry.append_operation(helper.br(0, &results, location));
    }

    Ok(())
}
