////! # Squashed `Felt` dictionary type
//! # Squashed `Felt` dictionary type
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
//    dialect::llvm,
    dialect::llvm,
//    ir::{Module, Type},
    ir::{Module, Type},
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
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: WithSelf<InfoAndTypeConcreteType>,
    _info: WithSelf<InfoAndTypeConcreteType>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    //let inner = registry.get_type(&info.ty)?;
    //let inner = registry.get_type(&info.ty)?;
//    //let layout = inner.layout(registry)?;
    //let layout = inner.layout(registry)?;
//

//    Ok(llvm::r#type::pointer(context, 0))
    Ok(llvm::r#type::pointer(context, 0))
//}
}
