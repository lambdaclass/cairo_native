////! # Nullable type
//! # Nullable type
////!
//!
////! Nullable is represented as a pointer, usually the null value will point to a alloca in the stack.
//! Nullable is represented as a pointer, usually the null value will point to a alloca in the stack.
////!
//!
////! A nullable is functionally equivalent to Rust's `Option<Box<T>>`. Since it's always paired with
//! A nullable is functionally equivalent to Rust's `Option<Box<T>>`. Since it's always paired with
////! `Box<T>` we can reuse its pointer, just leaving it null when there's no value.
//! `Box<T>` we can reuse its pointer, just leaving it null when there's no value.
//

//use super::{TypeBuilder, WithSelf};
use super::{TypeBuilder, WithSelf};
//use crate::{
use crate::{
//    error::Result,
    error::Result,
//    libfuncs::LibfuncHelper,
    libfuncs::LibfuncHelper,
//    metadata::{
    metadata::{
//        realloc_bindings::ReallocBindingsMeta, snapshot_clones::SnapshotClonesMeta, MetadataStorage,
        realloc_bindings::ReallocBindingsMeta, snapshot_clones::SnapshotClonesMeta, MetadataStorage,
//    },
    },
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        types::InfoAndTypeConcreteType,
        types::InfoAndTypeConcreteType,
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
//        llvm::{self, r#type::pointer},
        llvm::{self, r#type::pointer},
//        ods, scf,
        ods, scf,
//    },
    },
//    ir::{
    ir::{
//        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Block,
        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Block,
//        Location, Module, Region, Type, Value,
        Location, Module, Region, Type, Value,
//    },
    },
//    Context,
    Context,
//};
};
//

///// Build the MLIR type.
/// Build the MLIR type.
/////
///
///// Check out [the module](self) for more info.
/// Check out [the module](self) for more info.
//pub fn build<'ctx>(
pub fn build<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _module: &Module<'ctx>,
    _module: &Module<'ctx>,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: WithSelf<InfoAndTypeConcreteType>,
    info: WithSelf<InfoAndTypeConcreteType>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    metadata
    metadata
//        .get_or_insert_with::<SnapshotClonesMeta>(SnapshotClonesMeta::default)
        .get_or_insert_with::<SnapshotClonesMeta>(SnapshotClonesMeta::default)
//        .register(
        .register(
//            info.self_ty().clone(),
            info.self_ty().clone(),
//            snapshot_take,
            snapshot_take,
//            InfoAndTypeConcreteType {
            InfoAndTypeConcreteType {
//                info: info.info.clone(),
                info: info.info.clone(),
//                ty: info.ty.clone(),
                ty: info.ty.clone(),
//            },
            },
//        );
        );
//

//    // nullable is represented as a pointer, like a box, used to check if its null (when it can be null).
    // nullable is represented as a pointer, like a box, used to check if its null (when it can be null).
//    Ok(llvm::r#type::pointer(context, 0))
    Ok(llvm::r#type::pointer(context, 0))
//}
}
//

//#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
//fn snapshot_take<'ctx, 'this>(
fn snapshot_take<'ctx, 'this>(
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
//    info: WithSelf<InfoAndTypeConcreteType>,
    info: WithSelf<InfoAndTypeConcreteType>,
//    src_value: Value<'ctx, 'this>,
    src_value: Value<'ctx, 'this>,
//) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)> {
) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)> {
//    if metadata.get::<ReallocBindingsMeta>().is_none() {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
//        metadata.insert(ReallocBindingsMeta::new(context, helper));
        metadata.insert(ReallocBindingsMeta::new(context, helper));
//    }
    }
//

//    let elem_layout = registry.get_type(&info.ty)?.layout(registry)?;
    let elem_layout = registry.get_type(&info.ty)?.layout(registry)?;
//

//    let k0 = entry
    let k0 = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let null_ptr = entry
    let null_ptr = entry
//        .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
        .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let ptr_value = entry
    let ptr_value = entry
//        .append_operation(
        .append_operation(
//            OperationBuilder::new("llvm.ptrtoint", location)
            OperationBuilder::new("llvm.ptrtoint", location)
//                .add_operands(&[src_value])
                .add_operands(&[src_value])
//                .add_results(&[IntegerType::new(context, 64).into()])
                .add_results(&[IntegerType::new(context, 64).into()])
//                .build()?,
                .build()?,
//        )
        )
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let is_null = entry
    let is_null = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Eq,
            CmpiPredicate::Eq,
//            ptr_value,
            ptr_value,
//            k0,
            k0,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let value = entry
    let value = entry
//        .append_operation(scf::r#if(
        .append_operation(scf::r#if(
//            is_null,
            is_null,
//            &[llvm::r#type::pointer(context, 0)],
            &[llvm::r#type::pointer(context, 0)],
//            {
            {
//                let region = Region::new();
                let region = Region::new();
//                let block = region.append_block(Block::new(&[]));
                let block = region.append_block(Block::new(&[]));
//

//                block.append_operation(scf::r#yield(&[null_ptr], location));
                block.append_operation(scf::r#yield(&[null_ptr], location));
//                region
                region
//            },
            },
//            {
            {
//                let region = Region::new();
                let region = Region::new();
//                let block = region.append_block(Block::new(&[]));
                let block = region.append_block(Block::new(&[]));
//

//                let alloc_len = block
                let alloc_len = block
//                    .append_operation(arith::constant(
                    .append_operation(arith::constant(
//                        context,
                        context,
//                        IntegerAttribute::new(
                        IntegerAttribute::new(
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            elem_layout.size() as i64,
                            elem_layout.size() as i64,
//                        )
                        )
//                        .into(),
                        .into(),
//                        location,
                        location,
//                    ))
                    ))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//

//                let cloned_ptr = block
                let cloned_ptr = block
//                    .append_operation(ReallocBindingsMeta::realloc(
                    .append_operation(ReallocBindingsMeta::realloc(
//                        context, null_ptr, alloc_len, location,
                        context, null_ptr, alloc_len, location,
//                    ))
                    ))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//

//                block.append_operation(
                block.append_operation(
//                    ods::llvm::intr_memcpy(
                    ods::llvm::intr_memcpy(
//                        context,
                        context,
//                        cloned_ptr,
                        cloned_ptr,
//                        src_value,
                        src_value,
//                        alloc_len,
                        alloc_len,
//                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
//                        location,
                        location,
//                    )
                    )
//                    .into(),
                    .into(),
//                );
                );
//

//                block.append_operation(scf::r#yield(&[cloned_ptr], location));
                block.append_operation(scf::r#yield(&[cloned_ptr], location));
//                region
                region
//            },
            },
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    Ok((entry, value))
    Ok((entry, value))
//}
}
