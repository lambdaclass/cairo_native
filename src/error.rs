pub use self::{libfuncs::Error as CoreLibfuncBuilderError, types::Error as CoreTypeBuilderError};
use crate::{libfuncs::LibfuncBuilder, types::TypeBuilder};
use cairo_lang_sierra::{
    edit_state::EditStateError,
    extensions::{GenericLibfunc, GenericType},
    ids::{ConcreteLibfuncId, ConcreteTypeId},
    program_registry::ProgramRegistryError,
};
use serde::{Deserializer, Serializer};
use std::{alloc::LayoutError, fmt};
use thiserror::Error;

pub mod libfuncs;
pub mod types;

#[derive(Error)]
pub enum CompileError<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
{
    #[error(transparent)]
    EditStateError(#[from] EditStateError),
    #[error(transparent)]
    MlirError(#[from] melior::Error),
    #[error(transparent)]
    ProgramRegistryError(#[from] Box<ProgramRegistryError>),

    #[error("Error building type '{type_id}': {error}")]
    TypeBuilderError {
        type_id: ConcreteTypeId,
        error: <<TType as GenericType>::Concrete as TypeBuilder<TType, TLibfunc>>::Error,
    },
    #[error("Error building type '{libfunc_id}': {error}")]
    LibfuncBuilderError {
        libfunc_id: ConcreteLibfuncId,
        error: <<TLibfunc as GenericLibfunc>::Concrete as LibfuncBuilder<TType, TLibfunc>>::Error,
    },
}

// Manual implementation necessary because `#[derive(Debug)]` requires that `TType` and `TLibfunc`
// both implement `Debug`, which isn't the case.
impl<TType, TLibfunc> fmt::Debug for CompileError<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EditStateError(arg0) => f.debug_tuple("EditStateError").field(arg0).finish(),
            Self::MlirError(arg0) => f.debug_tuple("MlirError").field(arg0).finish(),
            Self::ProgramRegistryError(arg0) => {
                f.debug_tuple("ProgramRegistryError").field(arg0).finish()
            }
            Self::TypeBuilderError { type_id, error } => f
                .debug_struct("TypeBuilderError")
                .field("type_id", type_id)
                .field("error", error)
                .finish(),
            Self::LibfuncBuilderError { libfunc_id, error } => f
                .debug_struct("LibfuncBuilderError")
                .field("libfunc_id", libfunc_id)
                .field("error", error)
                .finish(),
        }
    }
}

#[derive(Error)]
pub enum JitRunnerError<'de, TType, TLibfunc, D, S>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
    S: Serializer,
{
    #[error(transparent)]
    LayoutError(#[from] LayoutError),
    #[error(transparent)]
    MlirError(#[from] melior::Error),
    #[error(transparent)]
    ProgramRegistryError(#[from] Box<ProgramRegistryError>),

    #[error("Error building type '{type_id}': {error}")]
    TypeBuilderError {
        type_id: ConcreteTypeId,
        error: <<TType as GenericType>::Concrete as TypeBuilder<TType, TLibfunc>>::Error,
    },

    #[error(transparent)]
    DeserializeError(D::Error),
    #[error(transparent)]
    SerializeError(S::Error),
}

impl<'de, TType, TLibfunc, D, S> fmt::Debug for JitRunnerError<'de, TType, TLibfunc, D, S>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
    S: Serializer,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayoutError(arg0) => f.debug_tuple("LayoutError").field(arg0).finish(),
            Self::MlirError(arg0) => f.debug_tuple("MlirError").field(arg0).finish(),
            Self::ProgramRegistryError(arg0) => {
                f.debug_tuple("ProgramRegistryError").field(arg0).finish()
            }
            Self::TypeBuilderError { type_id, error } => f
                .debug_struct("TypeBuilderError")
                .field("type_id", type_id)
                .field("error", error)
                .finish(),
            Self::DeserializeError(arg0) => f.debug_tuple("DeserializeError").field(arg0).finish(),
            Self::SerializeError(arg0) => f.debug_tuple("SerializeError").field(arg0).finish(),
        }
    }
}
