//! # StarkNet libfuncs
//!
//! TODO

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::MetadataStorage,
    types::TypeBuilder,
};
use cairo_lang_sierra::{
    extensions::{starknet::StarkNetConcreteLibfunc, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Block, Location},
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _entry: &'this Block<'ctx>,
    _location: Location<'ctx>,
    _helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    selector: &StarkNetConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        StarkNetConcreteLibfunc::CallContract(_) => todo!(),
        StarkNetConcreteLibfunc::ClassHashConst(_) => todo!(),
        StarkNetConcreteLibfunc::ClassHashTryFromFelt252(_) => todo!(),
        StarkNetConcreteLibfunc::ClassHashToFelt252(_) => todo!(),
        StarkNetConcreteLibfunc::ContractAddressConst(_) => todo!(),
        StarkNetConcreteLibfunc::ContractAddressTryFromFelt252(_) => todo!(),
        StarkNetConcreteLibfunc::ContractAddressToFelt252(_) => todo!(),
        StarkNetConcreteLibfunc::StorageRead(_) => todo!(),
        StarkNetConcreteLibfunc::StorageWrite(_) => todo!(),
        StarkNetConcreteLibfunc::StorageBaseAddressConst(_) => todo!(),
        StarkNetConcreteLibfunc::StorageBaseAddressFromFelt252(_) => todo!(),
        StarkNetConcreteLibfunc::StorageAddressFromBase(_) => todo!(),
        StarkNetConcreteLibfunc::StorageAddressFromBaseAndOffset(_) => todo!(),
        StarkNetConcreteLibfunc::StorageAddressToFelt252(_) => todo!(),
        StarkNetConcreteLibfunc::StorageAddressTryFromFelt252(_) => todo!(),
        StarkNetConcreteLibfunc::EmitEvent(_) => todo!(),
        StarkNetConcreteLibfunc::GetBlockHash(_) => todo!(),
        StarkNetConcreteLibfunc::GetExecutionInfo(_) => todo!(),
        StarkNetConcreteLibfunc::Deploy(_) => todo!(),
        StarkNetConcreteLibfunc::Keccak(_) => todo!(),
        StarkNetConcreteLibfunc::LibraryCall(_) => todo!(),
        StarkNetConcreteLibfunc::ReplaceClass(_) => todo!(),
        StarkNetConcreteLibfunc::SendMessageToL1(_) => todo!(),
        StarkNetConcreteLibfunc::Testing(_) => todo!(),
        StarkNetConcreteLibfunc::Secp256(_) => todo!(),
    }
}
