use cairo_lang_sierra::program_registry::ProgramRegistryError;
use std::{alloc::LayoutError, fmt, num::TryFromIntError, ops::Deref};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub struct Error {
    // TODO: enable once its stable in rust
    // pub backtrace: Backtrace,
    pub source: Box<ErrorImpl>,
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
            source: Box::new(error.into()),
        }
    }
}

#[derive(Debug, Error)]
pub enum ErrorImpl {
    #[error(transparent)]
    LayoutError(#[from] LayoutError),
    #[error(transparent)]
    MlirError(#[from] melior::Error),
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
    CompileError(String),
}

impl From<super::CoreTypeBuilderError> for ErrorImpl {
    fn from(value: super::CoreTypeBuilderError) -> Self {
        match *value.source {
            super::types::ErrorImpl::LayoutError(e) => Self::LayoutError(e),
            super::types::ErrorImpl::ProgramRegistryError(e) => Self::ProgramRegistryError(e),
            super::types::ErrorImpl::TryFromIntError(e) => Self::TryFromIntError(e),
            super::types::ErrorImpl::MlirError(e) => Self::MlirError(e),
            super::types::ErrorImpl::LibFuncError(e) => *e.source,
            super::types::ErrorImpl::ProgramRegistryErrorBoxed(e) => {
                Self::ProgramRegistryErrorBoxed(e)
            }
        }
    }
}
