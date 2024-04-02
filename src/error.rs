//! Various error types used thorough the crate.
use cairo_lang_sierra::{
    edit_state::EditStateError, ids::ConcreteTypeId, program_registry::ProgramRegistryError,
};

use std::{alloc::LayoutError, num::TryFromIntError};
use thiserror::Error;

use crate::metadata::gas::GasMetadataError;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
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

    #[error("cairo const data mismatch")]
    ConstDataMismatch,
}

pub fn make_unexpected_value_error(expected: String) -> Error {
    Error::UnexpectedValue(expected)
}

pub fn make_missing_parameter(ty: &ConcreteTypeId) -> Error {
    Error::MissingParameter(
        ty.debug_name
            .as_ref()
            .map(|x| x.to_string())
            .unwrap_or_default(),
    )
}
