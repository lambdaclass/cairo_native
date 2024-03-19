//! Various error types used thorough the crate.
use cairo_lang_sierra::{
    edit_state::EditStateError, ids::ConcreteTypeId, program_registry::ProgramRegistryError,
};

use std::{alloc::LayoutError, fmt, num::TryFromIntError, ops::Deref};
use thiserror::Error;

use crate::metadata::gas::GasMetadataError;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub struct Error {
    pub source: Box<ErrorImpl>,
}

#[derive(Error, Debug)]
pub enum ErrorImpl {
    #[error(transparent)]
    LayoutError(#[from] LayoutError),

    #[error(transparent)]
    MlirError(#[from] melior::Error),

    #[error("missing parameter of type '{0}'")]
    MissingParameter(String),

    #[error("unexpected value, expected value of type '{0}'")]
    UnexpectedValue(String),

    #[error("not enough gas to run")]
    InsufficientGasError,

    #[error("a syscall handler was expected but was not provided")]
    MissingSyscallHandler,

    #[error(transparent)]
    LayoutErrorPolyfill(#[from] crate::utils::LayoutError),

    #[error(transparent)]
    ProgramRegistryError(#[from] ProgramRegistryError),

    #[error(transparent)]
    ProgramRegistryErrorBoxed(#[from] Box<ProgramRegistryError>),

    #[error(transparent)]
    TryFromIntError(#[from] TryFromIntError),

    #[error("error parsing attribute")]
    ParseAttributeError,

    #[error("missing metadata")]
    MissingMetadata,

    #[error("a cairo-native sierra related assert failed: {0}")]
    SierraAssert(String),

    #[error("a compiler related error happened: {0}")]
    Error(String),

    #[error(transparent)]
    EditStateError(#[from] EditStateError),

    #[error("gas metadata error")]
    GasMetadataError(#[from] GasMetadataError),

    #[error("llvm error")]
    LLVMCompileError(String),
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
            source: Box::new(error.into()),
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

pub fn make_unexpected_value_error(expected: String) -> Error {
    ErrorImpl::UnexpectedValue(expected).into()
}

pub fn make_missing_parameter(ty: &ConcreteTypeId) -> Error {
    ErrorImpl::MissingParameter(
        ty.debug_name
            .as_ref()
            .map(|x| x.to_string())
            .unwrap_or_default(),
    )
    .into()
}
