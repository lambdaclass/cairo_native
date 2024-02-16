use cairo_lang_sierra::{
    edit_state::EditStateError,
    ids::{ConcreteLibfuncId, ConcreteTypeId},
    program_registry::ProgramRegistryError,
};
use std::{fmt, ops::Deref};
use thiserror::Error;

use super::{CoreLibfuncBuilderError, CoreTypeBuilderError};

pub type CompileError = Box<Error>;

#[derive(Error)]
pub struct Error {
    // TODO: enable once its stable in rust
    // pub backtrace: Backtrace,
    pub source: ErrorImpl,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.source, f)
    }
}

impl Deref for Error {
    type Target = ErrorImpl;

    fn deref(&self) -> &Self::Target {
        &self.source
    }
}

impl<E> From<E> for Error
where
    ErrorImpl: From<E>,
{
    fn from(error: E) -> Self {
        Self {
            // backtrace: Backtrace::capture(),
            source: error.into(),
        }
    }
}

impl<E> From<E> for Box<Error>
where
    ErrorImpl: From<E>,
{
    fn from(error: E) -> Self {
        Self::new(Error::from(error))
    }
}

// Manual implementation necessary because `#[derive(Debug)]` requires that `TType` and `TLibfunc`
// both implement `Debug`, which isn't the case.
impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Error")
            // .field("backtrace", &self.backtrace)
            .field("source", &self.source)
            .finish()
    }
}

#[derive(Error)]
pub enum ErrorImpl {
    #[error(transparent)]
    EditStateError(#[from] EditStateError),
    #[error(transparent)]
    MlirError(#[from] melior::Error),
    #[error(transparent)]
    ProgramRegistryError(#[from] Box<ProgramRegistryError>),

    #[error("Error building type '{type_id}': {error}")]
    TypeBuilderError {
        type_id: ConcreteTypeId,
        error: CoreTypeBuilderError,
    },
    #[error("Error building type '{libfunc_id}': {error}")]
    LibfuncBuilderError {
        libfunc_id: ConcreteLibfuncId,
        error: CoreLibfuncBuilderError,
    },
}

// Manual implementation necessary because `#[derive(Debug)]` requires that `TType` and `TLibfunc`
// both implement `Debug`, which isn't the case.
impl fmt::Debug for ErrorImpl {
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

pub fn make_type_builder_error(
    id: &ConcreteTypeId,
) -> impl '_ + FnOnce(CoreTypeBuilderError) -> Error
where
{
    move |source| {
        ErrorImpl::TypeBuilderError {
            type_id: id.clone(),
            error: source,
        }
        .into()
    }
}

pub fn make_libfunc_builder_error(
    id: &ConcreteLibfuncId,
) -> impl '_ + FnOnce(CoreLibfuncBuilderError) -> Error {
    move |source| {
        ErrorImpl::LibfuncBuilderError {
            libfunc_id: id.clone(),
            error: source,
        }
        .into()
    }
}
