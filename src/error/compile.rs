use crate::{libfuncs::LibfuncBuilder, types::TypeBuilder};
use cairo_lang_sierra::{
    edit_state::EditStateError,
    extensions::{GenericLibfunc, GenericType},
    ids::{ConcreteLibfuncId, ConcreteTypeId},
    program_registry::ProgramRegistryError,
};
use std::{backtrace::Backtrace, fmt, ops::Deref};
use thiserror::Error;

#[derive(Debug, Error)]
pub struct Error<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
{
    pub backtrace: Backtrace,
    pub source: ErrorImpl<TType, TLibfunc>,
}

impl<TType, TLibfunc> fmt::Display for Error<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.source, f)
    }
}

impl<TType, TLibfunc> Deref for Error<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
{
    type Target = ErrorImpl<TType, TLibfunc>;

    fn deref(&self) -> &Self::Target {
        &self.source
    }
}

impl<TType, TLibfunc, E> From<E> for Error<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
    ErrorImpl<TType, TLibfunc>: From<E>,
{
    fn from(error: E) -> Self {
        Self {
            backtrace: Backtrace::capture(),
            source: error.into(),
        }
    }
}

#[derive(Error)]
pub enum ErrorImpl<TType, TLibfunc>
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
impl<TType, TLibfunc> fmt::Debug for ErrorImpl<TType, TLibfunc>
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
