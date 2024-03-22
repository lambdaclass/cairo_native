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
    ProgramRegistryError(#[from] ProgramRegistryError),
    #[error(transparent)]
    ProgramRegistryErrorBoxed(#[from] Box<ProgramRegistryError>),
    #[error(transparent)]
    TryFromIntError(#[from] TryFromIntError),
    #[error(transparent)]
    MlirError(#[from] melior::Error),
    #[error(transparent)]
    LibFuncError(#[from] crate::error::libfuncs::Error),
}
