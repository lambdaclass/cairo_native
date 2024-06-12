////! # Starknet types
//! # Starknet types
////!
//!
////! ## ClassHash
//! ## ClassHash
////! Type for Starknet class hash, a value in the range [0, 2 ** 251).
//! Type for Starknet class hash, a value in the range [0, 2 ** 251).
////!
//!
////! ## ContractAddress
//! ## ContractAddress
////! Type for Starknet contract address, a value in the range [0, 2 ** 251).
//! Type for Starknet contract address, a value in the range [0, 2 ** 251).
////!
//!
////! ## StorageBaseAddress
//! ## StorageBaseAddress
////! Type for Starknet storage base address, a value in the range [0, 2 ** 251 - 256).
//! Type for Starknet storage base address, a value in the range [0, 2 ** 251 - 256).
////!
//!
////! ## StorageAddress
//! ## StorageAddress
////! Type for Starknet storage base address, a value in the range [0, 2 ** 251).
//! Type for Starknet storage base address, a value in the range [0, 2 ** 251).
////!
//!
////! ## System
//! ## System
////! Type for Starknet system object.
//! Type for Starknet system object.
////! Used to make system calls.
//! Used to make system calls.
////!
//!
////! ## Secp256Point
//! ## Secp256Point
////! TODO
//! TODO
//

//// TODO: Maybe the types used here can be i251 instead of i252.
// TODO: Maybe the types used here can be i251 instead of i252.
//

//use super::WithSelf;
use super::WithSelf;
//use crate::{error::Result, metadata::MetadataStorage};
use crate::{error::Result, metadata::MetadataStorage};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        starknet::{secp256::Secp256PointTypeConcrete, StarkNetTypeConcrete},
        starknet::{secp256::Secp256PointTypeConcrete, StarkNetTypeConcrete},
//        types::InfoOnlyConcreteType,
        types::InfoOnlyConcreteType,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::llvm,
    dialect::llvm,
//    ir::{r#type::IntegerType, Module, Type},
    ir::{r#type::IntegerType, Module, Type},
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
//    module: &Module<'ctx>,
    module: &Module<'ctx>,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    selector: WithSelf<StarkNetTypeConcrete>,
    selector: WithSelf<StarkNetTypeConcrete>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    match &*selector {
    match &*selector {
//        StarkNetTypeConcrete::ClassHash(info) => build_class_hash(
        StarkNetTypeConcrete::ClassHash(info) => build_class_hash(
//            context,
            context,
//            module,
            module,
//            registry,
            registry,
//            metadata,
            metadata,
//            WithSelf::new(selector.self_ty(), info),
            WithSelf::new(selector.self_ty(), info),
//        ),
        ),
//        StarkNetTypeConcrete::ContractAddress(info) => build_contract_address(
        StarkNetTypeConcrete::ContractAddress(info) => build_contract_address(
//            context,
            context,
//            module,
            module,
//            registry,
            registry,
//            metadata,
            metadata,
//            WithSelf::new(selector.self_ty(), info),
            WithSelf::new(selector.self_ty(), info),
//        ),
        ),
//        StarkNetTypeConcrete::StorageBaseAddress(info) => build_storage_base_address(
        StarkNetTypeConcrete::StorageBaseAddress(info) => build_storage_base_address(
//            context,
            context,
//            module,
            module,
//            registry,
            registry,
//            metadata,
            metadata,
//            WithSelf::new(selector.self_ty(), info),
            WithSelf::new(selector.self_ty(), info),
//        ),
        ),
//        StarkNetTypeConcrete::StorageAddress(info) => build_storage_address(
        StarkNetTypeConcrete::StorageAddress(info) => build_storage_address(
//            context,
            context,
//            module,
            module,
//            registry,
            registry,
//            metadata,
            metadata,
//            WithSelf::new(selector.self_ty(), info),
            WithSelf::new(selector.self_ty(), info),
//        ),
        ),
//        StarkNetTypeConcrete::System(info) => build_system(
        StarkNetTypeConcrete::System(info) => build_system(
//            context,
            context,
//            module,
            module,
//            registry,
            registry,
//            metadata,
            metadata,
//            WithSelf::new(selector.self_ty(), info),
            WithSelf::new(selector.self_ty(), info),
//        ),
        ),
//        StarkNetTypeConcrete::Secp256Point(info) => build_secp256_point(
        StarkNetTypeConcrete::Secp256Point(info) => build_secp256_point(
//            context,
            context,
//            module,
            module,
//            registry,
            registry,
//            metadata,
            metadata,
//            WithSelf::new(selector.self_ty(), info),
            WithSelf::new(selector.self_ty(), info),
//        ),
        ),
//    }
    }
//}
}
//

//pub fn build_class_hash<'ctx>(
pub fn build_class_hash<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    module: &Module<'ctx>,
    module: &Module<'ctx>,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: WithSelf<InfoOnlyConcreteType>,
    info: WithSelf<InfoOnlyConcreteType>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    // it's a felt252 value
    // it's a felt252 value
//    super::felt252::build(context, module, registry, metadata, info)
    super::felt252::build(context, module, registry, metadata, info)
//}
}
//

//pub fn build_contract_address<'ctx>(
pub fn build_contract_address<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    module: &Module<'ctx>,
    module: &Module<'ctx>,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: WithSelf<InfoOnlyConcreteType>,
    info: WithSelf<InfoOnlyConcreteType>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    // it's a felt252 value
    // it's a felt252 value
//    super::felt252::build(context, module, registry, metadata, info)
    super::felt252::build(context, module, registry, metadata, info)
//}
}
//

//pub fn build_storage_base_address<'ctx>(
pub fn build_storage_base_address<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    module: &Module<'ctx>,
    module: &Module<'ctx>,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: WithSelf<InfoOnlyConcreteType>,
    info: WithSelf<InfoOnlyConcreteType>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    // it's a felt252 value
    // it's a felt252 value
//    super::felt252::build(context, module, registry, metadata, info)
    super::felt252::build(context, module, registry, metadata, info)
//}
}
//

//pub fn build_storage_address<'ctx>(
pub fn build_storage_address<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    module: &Module<'ctx>,
    module: &Module<'ctx>,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: WithSelf<InfoOnlyConcreteType>,
    info: WithSelf<InfoOnlyConcreteType>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    // it's a felt252 value
    // it's a felt252 value
//    super::felt252::build(context, module, registry, metadata, info)
    super::felt252::build(context, module, registry, metadata, info)
//}
}
//

//pub fn build_system<'ctx>(
pub fn build_system<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _module: &Module<'ctx>,
    _module: &Module<'ctx>,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: WithSelf<InfoOnlyConcreteType>,
    _info: WithSelf<InfoOnlyConcreteType>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    Ok(llvm::r#type::pointer(context, 0))
    Ok(llvm::r#type::pointer(context, 0))
//}
}
//

//pub fn build_secp256_point<'ctx>(
pub fn build_secp256_point<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _module: &Module<'ctx>,
    _module: &Module<'ctx>,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: WithSelf<Secp256PointTypeConcrete>,
    _info: WithSelf<Secp256PointTypeConcrete>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    Ok(llvm::r#type::r#struct(
    Ok(llvm::r#type::r#struct(
//        context,
        context,
//        &[
        &[
//            llvm::r#type::r#struct(
            llvm::r#type::r#struct(
//                context,
                context,
//                &[
                &[
//                    IntegerType::new(context, 128).into(),
                    IntegerType::new(context, 128).into(),
//                    IntegerType::new(context, 128).into(),
                    IntegerType::new(context, 128).into(),
//                ],
                ],
//                false,
                false,
//            ),
            ),
//            llvm::r#type::r#struct(
            llvm::r#type::r#struct(
//                context,
                context,
//                &[
                &[
//                    IntegerType::new(context, 128).into(),
                    IntegerType::new(context, 128).into(),
//                    IntegerType::new(context, 128).into(),
                    IntegerType::new(context, 128).into(),
//                ],
                ],
//                false,
                false,
//            ),
            ),
//        ],
        ],
//        false,
        false,
//    ))
    ))
//}
}
