//! # Array type
//!
//! An array type is a dynamically allocated list of items.
//!
//! ## Layout
//!
//! Being dynamically allocated, we just need to keep the pointer to the data, its length and
//! its capacity:
//!
//! | Index | Type           | Description              |
//! | ----- | -------------- | ------------------------ |
//! |   0   | `!llvm.ptr<T>` | Pointer to the data[^1]. |
//! |   1   | `i32`          | Array start offset[^2].  |
//! |   1   | `i32`          | Array end offset[^2].    |
//! |   2   | `i32`          | Allocated capacity[^2].  |
//!
//! [^1]: When capacity is zero, this field is not guaranteed to be valid.
//! [^2]: Those numbers are number of items, **not bytes**.

use super::{TypeBuilder, WithSelf};
use crate::{
    error::{libfuncs, types::Result},
    libfuncs::LibfuncHelper,
    metadata::{
        realloc_bindings::ReallocBindingsMeta, snapshot_clones::SnapshotClonesMeta, MetadataStorage,
    },
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, llvm},
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute, StringAttribute},
        r#type::IntegerType,
        Block, Location, Module, Type, Value,
    },
    Context,
};

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoAndTypeConcreteType>,
) -> Result<Type<'ctx>> {
    metadata
        .get_or_insert_with::<SnapshotClonesMeta>(SnapshotClonesMeta::default)
        .register(
            info.self_ty().clone(),
            snapshot_take,
            InfoAndTypeConcreteType {
                info: info.info.clone(),
                ty: info.ty.clone(),
            },
        );

    let ptr_ty = llvm::r#type::opaque_pointer(context);
    let len_ty = IntegerType::new(context, 32).into();

    Ok(llvm::r#type::r#struct(
        context,
        &[ptr_ty, len_ty, len_ty, len_ty],
        false,
    ))
}

#[allow(clippy::too_many_arguments)]
fn snapshot_take<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoAndTypeConcreteType>,
    src_value: Value<'ctx, 'this>,
) -> libfuncs::Result<Value<'ctx, 'this>> {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let elem_snapshot_take = metadata
        .get::<SnapshotClonesMeta>()
        .and_then(|meta| meta.wrap_invoke(&info.ty));

    let elem_ty = registry.get_type(&info.ty)?;
    let elem_layout = elem_ty.layout(registry)?;
    let elem_stride = elem_layout.pad_to_align().size();

    let src_ptr = entry
        .append_operation(llvm::extract_value(
            context,
            src_value,
            DenseI64ArrayAttribute::new(context, &[0]),
            llvm::r#type::opaque_pointer(context),
            location,
        ))
        .result(0)?
        .into();
    let array_start = entry
        .append_operation(llvm::extract_value(
            context,
            src_value,
            DenseI64ArrayAttribute::new(context, &[1]),
            IntegerType::new(context, 32).into(),
            location,
        ))
        .result(0)?
        .into();
    let array_end = entry
        .append_operation(llvm::extract_value(
            context,
            src_value,
            DenseI64ArrayAttribute::new(context, &[2]),
            IntegerType::new(context, 32).into(),
            location,
        ))
        .result(0)?
        .into();

    let elem_stride = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(
                elem_stride.try_into()?,
                IntegerType::new(context, 64).into(),
            )
            .into(),
            location,
        ))
        .result(0)?
        .into();

    let array_ty = registry.build_type(context, helper, registry, metadata, info.self_ty())?;

    let array_len = entry
        .append_operation(arith::subi(array_end, array_start, location))
        .result(0)?
        .into();
    let dst_len_bytes = {
        let array_len = entry
            .append_operation(arith::extui(
                array_len,
                IntegerType::new(context, 64).into(),
                location,
            ))
            .result(0)?
            .into();

        entry
            .append_operation(arith::muli(array_len, elem_stride, location))
            .result(0)?
            .into()
    };

    let dst_ptr = {
        let dst_ptr = entry
            .append_operation(llvm::nullptr(
                llvm::r#type::opaque_pointer(context),
                location,
            ))
            .result(0)?
            .into();

        entry
            .append_operation(ReallocBindingsMeta::realloc(
                context,
                dst_ptr,
                dst_len_bytes,
                location,
            ))
            .result(0)?
            .into()
    };

    match elem_snapshot_take {
        Some(_) => todo!(),
        None => {
            let is_volatile = entry
                .append_operation(arith::constant(
                    context,
                    IntegerAttribute::new(0, IntegerType::new(context, 1).into()).into(),
                    location,
                ))
                .result(0)?
                .into();

            let src_ptr_offset = {
                let array_start = entry
                    .append_operation(arith::extui(
                        array_start,
                        IntegerType::new(context, 64).into(),
                        location,
                    ))
                    .result(0)?
                    .into();

                entry
                    .append_operation(arith::muli(array_start, elem_stride, location))
                    .result(0)?
                    .into()
            };
            let src_ptr = entry
                .append_operation(llvm::get_element_ptr_dynamic(
                    context,
                    src_ptr,
                    &[src_ptr_offset],
                    IntegerType::new(context, 8).into(),
                    llvm::r#type::opaque_pointer(context),
                    location,
                ))
                .result(0)?
                .into();

            entry.append_operation(llvm::call_intrinsic(
                context,
                StringAttribute::new(context, "llvm.memcpy"),
                &[dst_ptr, src_ptr, dst_len_bytes, is_volatile],
                &[],
                location,
            ));
        }
    }

    let dst_value = entry
        .append_operation(llvm::undef(array_ty, location))
        .result(0)?
        .into();

    let k0 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(0, IntegerType::new(context, 32).into()).into(),
            location,
        ))
        .result(0)?
        .into();
    let dst_value = entry
        .append_operation(llvm::insert_value(
            context,
            dst_value,
            DenseI64ArrayAttribute::new(context, &[0]),
            dst_ptr,
            location,
        ))
        .result(0)?
        .into();
    let dst_value = entry
        .append_operation(llvm::insert_value(
            context,
            dst_value,
            DenseI64ArrayAttribute::new(context, &[1]),
            k0,
            location,
        ))
        .result(0)?
        .into();
    let dst_value = entry
        .append_operation(llvm::insert_value(
            context,
            dst_value,
            DenseI64ArrayAttribute::new(context, &[2]),
            array_len,
            location,
        ))
        .result(0)?
        .into();
    let dst_value = entry
        .append_operation(llvm::insert_value(
            context,
            dst_value,
            DenseI64ArrayAttribute::new(context, &[3]),
            array_len,
            location,
        ))
        .result(0)?
        .into();

    Ok(dst_value)
}
