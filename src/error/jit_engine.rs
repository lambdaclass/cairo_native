use crate::{libfuncs::LibfuncBuilder, types::TypeBuilder};
use cairo_lang_sierra::{
    extensions::{GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistryError,
};
use serde::{Deserializer, Serializer};
use std::{alloc::LayoutError, backtrace::Backtrace, fmt, ops::Deref};
use thiserror::Error;

#[derive(Debug, Error)]
pub struct Error<'de, TType, TLibfunc, D, S>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
    S: Serializer,
{
    pub backtrace: Backtrace,
    pub source: ErrorImpl<'de, TType, TLibfunc, D, S>,
}

impl<'de, TType, TLibfunc, D, S> fmt::Display for Error<'de, TType, TLibfunc, D, S>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
    S: Serializer,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.source, f)
    }
}

impl<'de, TType, TLibfunc, D, S> Deref for Error<'de, TType, TLibfunc, D, S>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
    S: Serializer,
{
    type Target = ErrorImpl<'de, TType, TLibfunc, D, S>;

    fn deref(&self) -> &Self::Target {
        &self.source
    }
}

impl<'de, TType, TLibfunc, D, S, E> From<E> for Error<'de, TType, TLibfunc, D, S>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
    S: Serializer,
    ErrorImpl<'de, TType, TLibfunc, D, S>: From<E>,
{
    fn from(error: E) -> Self {
        Self {
            backtrace: Backtrace::capture(),
            source: error.into(),
        }
    }
}

#[derive(Error)]
pub enum ErrorImpl<'de, TType, TLibfunc, D, S>
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

impl<'de, TType, TLibfunc, D, S> fmt::Debug for ErrorImpl<'de, TType, TLibfunc, D, S>
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
