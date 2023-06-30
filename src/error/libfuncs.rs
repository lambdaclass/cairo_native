use cairo_lang_sierra::program_registry::ProgramRegistryError;
use std::{alloc::LayoutError, num::TryFromIntError};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    LayoutError(#[from] LayoutError),
    #[error(transparent)]
    MlirError(#[from] melior::Error),
    #[error(transparent)]
    ProgramRegistryError(#[from] Box<ProgramRegistryError>),
    #[error(transparent)]
    TryFromIntError(#[from] TryFromIntError),
}

impl From<super::CoreTypeBuilderError> for Error {
    fn from(_value: super::CoreTypeBuilderError) -> Self {
        todo!()
    }
}
