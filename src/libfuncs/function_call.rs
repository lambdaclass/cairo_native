////! # Function call libfuncs
//! # Function call libfuncs
////!
//!
////! Includes logic for handling direct tail recursive function calls. More information on this topic
//! Includes logic for handling direct tail recursive function calls. More information on this topic
////! at the [tail recursive metadata](crate::metadata::tail_recursion).
//! at the [tail recursive metadata](crate::metadata::tail_recursion).
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    block_ext::BlockExt,
    block_ext::BlockExt,
//    error::Result,
    error::Result,
//    metadata::{tail_recursion::TailRecursionMeta, MetadataStorage},
    metadata::{tail_recursion::TailRecursionMeta, MetadataStorage},
//    types::TypeBuilder,
    types::TypeBuilder,
//    utils::generate_function_name,
    utils::generate_function_name,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        function_call::SignatureAndFunctionConcreteLibfunc,
        function_call::SignatureAndFunctionConcreteLibfunc,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::{cf, func, index, llvm, memref},
    dialect::{cf, func, index, llvm, memref},
//    ir::{
    ir::{
//        attribute::{DenseI32ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute},
        attribute::{DenseI32ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute},
//        r#type::IntegerType,
        r#type::IntegerType,
//        Block, Location, Type, Value,
        Block, Location, Type, Value,
//    },
    },
//    Context,
    Context,
//};
};
//use std::alloc::Layout;
use std::alloc::Layout;
//

///// Generate MLIR operations for the `function_call` libfunc.
/// Generate MLIR operations for the `function_call` libfunc.
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
//    info: &SignatureAndFunctionConcreteLibfunc,
    info: &SignatureAndFunctionConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let mut tailrec_meta = metadata.remove::<TailRecursionMeta>();
    let mut tailrec_meta = metadata.remove::<TailRecursionMeta>();
//

//    let mut arguments = Vec::new();
    let mut arguments = Vec::new();
//    for (idx, type_id) in info.function.signature.param_types.iter().enumerate() {
    for (idx, type_id) in info.function.signature.param_types.iter().enumerate() {
//        let type_info = registry.get_type(type_id)?;
        let type_info = registry.get_type(type_id)?;
//

//        if !(type_info.is_builtin() && type_info.is_zst(registry)) {
        if !(type_info.is_builtin() && type_info.is_zst(registry)) {
//            arguments.push(
            arguments.push(
//                if tailrec_meta.is_none() && type_info.is_memory_allocated(registry) {
                if tailrec_meta.is_none() && type_info.is_memory_allocated(registry) {
//                    let elem_ty = type_info.build(context, helper, registry, metadata, type_id)?;
                    let elem_ty = type_info.build(context, helper, registry, metadata, type_id)?;
//                    let stack_ptr = helper.init_block().alloca1(
                    let stack_ptr = helper.init_block().alloca1(
//                        context,
                        context,
//                        location,
                        location,
//                        elem_ty,
                        elem_ty,
//                        Some(type_info.layout(registry)?.align()),
                        Some(type_info.layout(registry)?.align()),
//                    )?;
                    )?;
//

//                    entry.store(
                    entry.store(
//                        context,
                        context,
//                        location,
                        location,
//                        stack_ptr,
                        stack_ptr,
//                        entry.argument(idx)?.into(),
                        entry.argument(idx)?.into(),
//                        Some(type_info.layout(registry)?.align()),
                        Some(type_info.layout(registry)?.align()),
//                    );
                    );
//

//                    stack_ptr
                    stack_ptr
//                } else {
                } else {
//                    entry.argument(idx)?.into()
                    entry.argument(idx)?.into()
//                },
                },
//            );
            );
//        }
        }
//    }
    }
//

//    if let Some(tailrec_meta) = &mut tailrec_meta {
    if let Some(tailrec_meta) = &mut tailrec_meta {
//        let depth_counter =
        let depth_counter =
//            entry.append_op_result(memref::load(tailrec_meta.depth_counter(), &[], location))?;
            entry.append_op_result(memref::load(tailrec_meta.depth_counter(), &[], location))?;
//

//        let index1 = entry.append_op_result(index::constant(
        let index1 = entry.append_op_result(index::constant(
//            context,
            context,
//            IntegerAttribute::new(Type::index(context), 1),
            IntegerAttribute::new(Type::index(context), 1),
//            location,
            location,
//        ))?;
        ))?;
//

//        let depth_counter_plus_1 =
        let depth_counter_plus_1 =
//            entry.append_op_result(index::add(depth_counter, index1, location))?;
            entry.append_op_result(index::add(depth_counter, index1, location))?;
//

//        entry.append_operation(memref::store(
        entry.append_operation(memref::store(
//            depth_counter_plus_1,
            depth_counter_plus_1,
//            tailrec_meta.depth_counter(),
            tailrec_meta.depth_counter(),
//            &[],
            &[],
//            location,
            location,
//        ));
        ));
//

//        entry.append_operation(cf::br(
        entry.append_operation(cf::br(
//            &tailrec_meta.recursion_target(),
            &tailrec_meta.recursion_target(),
//            &arguments,
            &arguments,
//            location,
            location,
//        ));
        ));
//

//        let cont_block = helper.append_block(Block::new(
        let cont_block = helper.append_block(Block::new(
//            &info
            &info
//                .function
                .function
//                .signature
                .signature
//                .ret_types
                .ret_types
//                .iter()
                .iter()
//                .map(|ty| {
                .map(|ty| {
//                    (
                    (
//                        registry
                        registry
//                            .get_type(ty)
                            .get_type(ty)
//                            .unwrap()
                            .unwrap()
//                            .build(context, helper, registry, metadata, ty)
                            .build(context, helper, registry, metadata, ty)
//                            .unwrap(),
                            .unwrap(),
//                        location,
                        location,
//                    )
                    )
//                })
                })
//                .collect::<Vec<_>>(),
                .collect::<Vec<_>>(),
//        ));
        ));
//        tailrec_meta.set_return_target(cont_block);
        tailrec_meta.set_return_target(cont_block);
//

//        let mut results = Vec::<Value>::new();
        let mut results = Vec::<Value>::new();
//        let mut count = 0;
        let mut count = 0;
//        for var_info in &info.signature.branch_signatures[0].vars {
        for var_info in &info.signature.branch_signatures[0].vars {
//            let type_info = registry.get_type(&var_info.ty)?;
            let type_info = registry.get_type(&var_info.ty)?;
//

//            if type_info.is_builtin() && type_info.is_zst(registry) {
            if type_info.is_builtin() && type_info.is_zst(registry) {
//                results.push(
                results.push(
//                    cont_block
                    cont_block
//                        .append_operation(llvm::undef(
                        .append_operation(llvm::undef(
//                            type_info.build(context, helper, registry, metadata, &var_info.ty)?,
                            type_info.build(context, helper, registry, metadata, &var_info.ty)?,
//                            location,
                            location,
//                        ))
                        ))
//                        .result(0)?
                        .result(0)?
//                        .into(),
                        .into(),
//                );
                );
//            } else {
            } else {
//                let val = cont_block.argument(count)?.into();
                let val = cont_block.argument(count)?.into();
//                count += 1;
                count += 1;
//

//                results.push(val);
                results.push(val);
//            }
            }
//        }
        }
//

//        cont_block.append_operation(helper.br(0, &results, location));
        cont_block.append_operation(helper.br(0, &results, location));
//    } else {
    } else {
//        let mut result_types = Vec::new();
        let mut result_types = Vec::new();
//        let return_types = info
        let return_types = info
//            .function
            .function
//            .signature
            .signature
//            .ret_types
            .ret_types
//            .iter()
            .iter()
//            .filter_map(|type_id| {
            .filter_map(|type_id| {
//                let type_info = registry.get_type(type_id).unwrap();
                let type_info = registry.get_type(type_id).unwrap();
//                if type_info.is_builtin() && type_info.is_zst(registry) {
                if type_info.is_builtin() && type_info.is_zst(registry) {
//                    None
                    None
//                } else {
                } else {
//                    Some((type_id, type_info))
                    Some((type_id, type_info))
//                }
                }
//            })
            })
//            .collect::<Vec<_>>();
            .collect::<Vec<_>>();
//        // A function has a return pointer if either:
        // A function has a return pointer if either:
//        //   - There are multiple return values.
        //   - There are multiple return values.
//        //   - The return value is memory allocated.
        //   - The return value is memory allocated.
//        let has_return_ptr = if return_types.len() > 1 {
        let has_return_ptr = if return_types.len() > 1 {
//            result_types.extend(
            result_types.extend(
//                return_types
                return_types
//                    .iter()
                    .iter()
//                    .map(|(type_id, type_info)| {
                    .map(|(type_id, type_info)| {
//                        type_info.build(context, helper, registry, metadata, type_id)
                        type_info.build(context, helper, registry, metadata, type_id)
//                    })
                    })
//                    .collect::<std::result::Result<Vec<_>, _>>()?,
                    .collect::<std::result::Result<Vec<_>, _>>()?,
//            );
            );
//

//            Some(false)
            Some(false)
//        } else if return_types
        } else if return_types
//            .first()
            .first()
//            .is_some_and(|(_, type_info)| type_info.is_memory_allocated(registry))
            .is_some_and(|(_, type_info)| type_info.is_memory_allocated(registry))
//        {
        {
//            let (type_id, type_info) = return_types[0];
            let (type_id, type_info) = return_types[0];
//            let layout = type_info.layout(registry)?;
            let layout = type_info.layout(registry)?;
//

//            let stack_ptr = helper.init_block().alloca1(
            let stack_ptr = helper.init_block().alloca1(
//                context,
                context,
//                location,
                location,
//                type_info.build(context, helper, registry, metadata, type_id)?,
                type_info.build(context, helper, registry, metadata, type_id)?,
//                Some(layout.align()),
                Some(layout.align()),
//            )?;
            )?;
//

//            arguments.insert(0, stack_ptr);
            arguments.insert(0, stack_ptr);
//

//            Some(true)
            Some(true)
//        } else if return_types.first().is_some() {
        } else if return_types.first().is_some() {
//            let (type_id, type_info) = return_types[0];
            let (type_id, type_info) = return_types[0];
//            result_types.push(type_info.build(context, helper, registry, metadata, type_id)?);
            result_types.push(type_info.build(context, helper, registry, metadata, type_id)?);
//

//            None
            None
//        } else {
        } else {
//            None
            None
//        };
        };
//

//        let function_call_result = entry.append_operation(func::call(
        let function_call_result = entry.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, &generate_function_name(&info.function.id)),
            FlatSymbolRefAttribute::new(context, &generate_function_name(&info.function.id)),
//            &arguments,
            &arguments,
//            &result_types,
            &result_types,
//            location,
            location,
//        ));
        ));
//

//        let mut results = Vec::new();
        let mut results = Vec::new();
//        match has_return_ptr {
        match has_return_ptr {
//            Some(true) => {
            Some(true) => {
//                // Manual return type.
                // Manual return type.
//

//                let mut layout = Layout::new::<()>();
                let mut layout = Layout::new::<()>();
//                for (idx, type_id) in info.function.signature.ret_types.iter().enumerate() {
                for (idx, type_id) in info.function.signature.ret_types.iter().enumerate() {
//                    let type_info = registry.get_type(type_id)?;
                    let type_info = registry.get_type(type_id)?;
//

//                    if type_info.is_builtin() && type_info.is_zst(registry) {
                    if type_info.is_builtin() && type_info.is_zst(registry) {
//                        results.push(entry.argument(idx)?.into());
                        results.push(entry.argument(idx)?.into());
//                    } else {
                    } else {
//                        let val = arguments[0];
                        let val = arguments[0];
//

//                        let offset;
                        let offset;
//                        let ret_layout = type_info.layout(registry)?;
                        let ret_layout = type_info.layout(registry)?;
//                        (layout, offset) = layout.extend(ret_layout)?;
                        (layout, offset) = layout.extend(ret_layout)?;
//

//                        let pointer_val = entry
                        let pointer_val = entry
//                            .append_operation(llvm::get_element_ptr(
                            .append_operation(llvm::get_element_ptr(
//                                context,
                                context,
//                                val,
                                val,
//                                DenseI32ArrayAttribute::new(context, &[offset as i32]),
                                DenseI32ArrayAttribute::new(context, &[offset as i32]),
//                                IntegerType::new(context, 8).into(),
                                IntegerType::new(context, 8).into(),
//                                llvm::r#type::pointer(context, 0),
                                llvm::r#type::pointer(context, 0),
//                                location,
                                location,
//                            ))
                            ))
//                            .result(0)?
                            .result(0)?
//                            .into();
                            .into();
//

//                        results.push(entry.load(
                        results.push(entry.load(
//                            context,
                            context,
//                            location,
                            location,
//                            pointer_val,
                            pointer_val,
//                            type_info.build(context, helper, registry, metadata, type_id)?,
                            type_info.build(context, helper, registry, metadata, type_id)?,
//                            None,
                            None,
//                        )?);
                        )?);
//                    }
                    }
//                }
                }
//            }
            }
//            Some(false) => {
            Some(false) => {
//                // Complex return type. Just extract the values from the struct, since LLVM will
                // Complex return type. Just extract the values from the struct, since LLVM will
//                // handle the rest.
                // handle the rest.
//

//                let mut count = 0;
                let mut count = 0;
//                for (idx, type_id) in info.function.signature.ret_types.iter().enumerate() {
                for (idx, type_id) in info.function.signature.ret_types.iter().enumerate() {
//                    let type_info = registry.get_type(type_id)?;
                    let type_info = registry.get_type(type_id)?;
//

//                    if type_info.is_builtin() && type_info.is_zst(registry) {
                    if type_info.is_builtin() && type_info.is_zst(registry) {
//                        results.push(entry.argument(idx)?.into());
                        results.push(entry.argument(idx)?.into());
//                    } else {
                    } else {
//                        let val = function_call_result.result(count)?.into();
                        let val = function_call_result.result(count)?.into();
//                        count += 1;
                        count += 1;
//

//                        results.push(val);
                        results.push(val);
//                    }
                    }
//                }
                }
//            }
            }
//            None => {
            None => {
//                // Returned data is simple.
                // Returned data is simple.
//

//                let mut count = 0;
                let mut count = 0;
//                for (idx, type_id) in info.function.signature.ret_types.iter().enumerate() {
                for (idx, type_id) in info.function.signature.ret_types.iter().enumerate() {
//                    let type_info = registry.get_type(type_id)?;
                    let type_info = registry.get_type(type_id)?;
//                    assert!(!type_info.is_memory_allocated(registry));
                    assert!(!type_info.is_memory_allocated(registry));
//

//                    if type_info.is_builtin() && type_info.is_zst(registry) {
                    if type_info.is_builtin() && type_info.is_zst(registry) {
//                        results.push(entry.argument(idx)?.into());
                        results.push(entry.argument(idx)?.into());
//                    } else {
                    } else {
//                        let value = function_call_result.result(count)?.into();
                        let value = function_call_result.result(count)?.into();
//                        count += 1;
                        count += 1;
//

//                        results.push(value);
                        results.push(value);
//                    }
                    }
//                }
                }
//            }
            }
//        }
        }
//

//        entry.append_operation(helper.br(0, &results, location));
        entry.append_operation(helper.br(0, &results, location));
//    }
    }
//

//    if let Some(tailrec_meta) = tailrec_meta {
    if let Some(tailrec_meta) = tailrec_meta {
//        metadata.insert(tailrec_meta);
        metadata.insert(tailrec_meta);
//    }
    }
//

//    Ok(())
    Ok(())
//}
}
