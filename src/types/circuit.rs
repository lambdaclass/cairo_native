//! # `Circuit` type

use super::WithSelf;
use crate::{error::Result, metadata::MetadataStorage};
use cairo_lang_sierra::{
    extensions::{
        circuit::CircuitTypeConcrete,
        core::{CoreLibfunc, CoreType},
        types::InfoOnlyConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{r#type::IntegerType, Module, Type},
    Context,
};

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    selector: WithSelf<CircuitTypeConcrete>,
) -> Result<Type<'ctx>> {
    match &*selector {
        CircuitTypeConcrete::AddMod(_)
        | CircuitTypeConcrete::MulMod(_)
        | CircuitTypeConcrete::CircuitDescriptor(_) => Ok(IntegerType::new(context, 64).into()),
        CircuitTypeConcrete::CircuitModulus(_) => Ok(IntegerType::new(context, 384).into()),
        CircuitTypeConcrete::U96Guarantee(_) => Ok(IntegerType::new(context, 96).into()),
        CircuitTypeConcrete::CircuitInputAccumulator(info) => build_circuit_accumulator(
            context,
            module,
            registry,
            metadata,
            WithSelf::new(selector.self_ty(), info),
        ),
        CircuitTypeConcrete::CircuitData(info) => build_circuit_data(
            context,
            module,
            registry,
            metadata,
            WithSelf::new(selector.self_ty(), info),
        ),
        CircuitTypeConcrete::Circuit(_)
        | CircuitTypeConcrete::AddModGate(_) 
        | CircuitTypeConcrete::SubModGate(_)
        | CircuitTypeConcrete::MulModGate(_)
        | CircuitTypeConcrete::InverseGate(_)
        | CircuitTypeConcrete::CircuitFailureGuarantee(_)
        | CircuitTypeConcrete::CircuitOutputs(_)
        | CircuitTypeConcrete::CircuitPartialOutputs(_)
        | CircuitTypeConcrete::CircuitInput(_) 
        | CircuitTypeConcrete::U96LimbsLessThanGuarantee(_) => todo!(),
    }
}

pub fn build_circuit_accumulator<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoOnlyConcreteType>,
) -> Result<Type<'ctx>> {
    Ok(IntegerType::new(context, 64).into())
}

pub fn build_circuit_data<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoOnlyConcreteType>,
) -> Result<Type<'ctx>> {
    Ok(IntegerType::new(context, 64).into())
}
