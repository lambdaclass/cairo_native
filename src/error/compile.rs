use cairo_lang_sierra::{
    edit_state::EditStateError,
    ids::{ConcreteLibfuncId, ConcreteTypeId},
    program_registry::ProgramRegistryError,
};
use std::{fmt, ops::Deref};
use thiserror::Error;

use crate::metadata::gas::GasMetadataError;

use super::{CoreLibfuncBuilderError, CoreTypeBuilderError};

pub type CompileError = Box<Error>;

#[derive(Error, Debug)]
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

#[derive(Error, Debug)]
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
    #[error("gas metadata error")]
    GasMetadataError(#[from] GasMetadataError),
    #[error("llvm error")]
    LLVMCompileError(String),
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
