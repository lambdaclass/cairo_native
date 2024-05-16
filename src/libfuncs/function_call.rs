//! # Function call libfuncs
//!
//! Includes logic for handling direct tail recursive function calls. More information on this topic
//! at the [tail recursive metadata](crate::metadata::tail_recursion).

use super::LibfuncHelper;
use crate::{
    block_ext::BlockExt,
    error::Result,
    metadata::{tail_recursion::TailRecursionMeta, MetadataStorage},
    types::TypeBuilder,
    utils::generate_function_name,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        function_call::SignatureAndFunctionConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{cf, func, index, llvm, memref},
    ir::{
        attribute::{DenseI32ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute},
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
    info: &SignatureAndFunctionConcreteLibfunc,
) -> Result<()> {
    let mut tailrec_meta = metadata.remove::<TailRecursionMeta>();

    let mut arguments = Vec::new();
    for (idx, type_id) in info.function.signature.param_types.iter().enumerate() {
        let type_info = registry.get_type(type_id)?;

        if !(type_info.is_builtin() && type_info.is_zst(registry)) {
            arguments.push(
                if tailrec_meta.is_none() && type_info.is_memory_allocated(registry) {
                    let elem_ty = type_info.build(context, helper, registry, metadata, type_id)?;
                    let stack_ptr = helper.init_block().alloca1(
                        context,
                        location,
                        elem_ty,
                        Some(type_info.layout(registry)?.align()),
                    )?;

                    entry.store(
                        context,
                        location,
                        stack_ptr,
                        entry.argument(idx)?.into(),
                        Some(type_info.layout(registry)?.align()),
                    );

                    stack_ptr
                } else {
                    entry.argument(idx)?.into()
                },
            );
        }
    }

    if let Some(tailrec_meta) = &mut tailrec_meta {
        let depth_counter =
            entry.append_op_result(memref::load(tailrec_meta.depth_counter(), &[], location))?;

        let index1 = entry.append_op_result(index::constant(
            context,
            IntegerAttribute::new(Type::index(context), 1),
            location,
        ))?;

        let depth_counter_plus_1 =
            entry.append_op_result(index::add(depth_counter, index1, location))?;

        entry.append_operation(memref::store(
            depth_counter_plus_1,
            tailrec_meta.depth_counter(),
            &[],
            location,
        ));

        entry.append_operation(cf::br(
            &tailrec_meta.recursion_target(),
            &arguments,
            location,
        ));

        let cont_block = helper.append_block(Block::new(
            &info
                .function
                .signature
                .ret_types
                .iter()
                .map(|ty| {
                    (
                        registry
                            .get_type(ty)
                            .unwrap()
                            .build(context, helper, registry, metadata, ty)
                            .unwrap(),
                        location,
                    )
                })
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

                results.push(val);
            }
        }

        cont_block.append_operation(helper.br(0, &results, location));
    } else {
        let mut result_types = Vec::new();
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
        // A function has a return pointer if either:
        //   - There are multiple return values.
        //   - The return value is memory allocated.
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

            let stack_ptr = helper.init_block().alloca1(
                context,
                location,
                type_info.build(context, helper, registry, metadata, type_id)?,
                Some(layout.align()),
            )?;

            arguments.insert(0, stack_ptr);

            Some(true)
        } else {
            let (type_id, type_info) = return_types[0];
            result_types.push(type_info.build(context, helper, registry, metadata, type_id)?);

            None
        };

        let function_call_result = entry.append_operation(func::call(
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

                        results.push(entry.load(
                            context,
                            location,
                            pointer_val,
                            type_info.build(context, helper, registry, metadata, type_id)?,
                            None,
                        )?);
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
                        let val = function_call_result.result(count)?.into();
                        count += 1;

                        results.push(val);
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
                        let value = function_call_result.result(count)?.into();
                        count += 1;

                        results.push(value);
                    }
                }
            }
        }

        entry.append_operation(helper.br(0, &results, location));
    }

    if let Some(tailrec_meta) = tailrec_meta {
        metadata.insert(tailrec_meta);
    }

    Ok(())
}
