//! Various error types used thorough the crate.
use crate::metadata::gas::GasMetadataError;
use cairo_lang_sierra::extensions::modules::utils::Range;
use cairo_lang_sierra::{
    edit_state::EditStateError, ids::ConcreteTypeId, program_registry::ProgramRegistryError,
};
use num_bigint::BigInt;
use std::backtrace::{Backtrace, BacktraceStatus};
use std::panic::Location;
use std::{alloc::LayoutError, num::TryFromIntError};
use thiserror::Error;

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

    #[error(transparent)]
    SierraAssert(#[from] SierraAssertError),

    #[error(transparent)]
    NativeAssert(#[from] NativeAssertError),

    #[error(transparent)]
    Compiler(#[from] CompilerError),

    #[error(transparent)]
    EditStateError(#[from] EditStateError),

    #[error(transparent)]
    GasMetadataError(#[from] GasMetadataError),

    #[error("llvm compile error: {0}")]
    LLVMCompileError(String),

    #[error("ld link error: {0}")]
    LinkError(String),

    #[error("cairo const data mismatch")]
    ConstDataMismatch,

    #[error("expected an integer-like type")]
    IntegerLikeTypeExpected,

    #[error("integer conversion failed")]
    IntegerConversion,

    #[error("missing BuiltinCosts global symbol, should never happen, this is a bug")]
    MissingBuiltinCostsSymbol,

    #[error("selector not found in the AotContractExecutor mappings")]
    SelectorNotFound,

    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[error(transparent)]
    LibraryLoadError(#[from] libloading::Error),

    #[error(transparent)]
    SerdeJsonError(#[from] serde_json::Error),
}

impl Error {
    pub fn make_missing_parameter(ty: &ConcreteTypeId) -> Self {
        Self::MissingParameter(
            ty.debug_name
                .as_ref()
                .map(|x| x.to_string())
                .unwrap_or_default(),
        )
    }
}

#[derive(Error, Debug)]
pub enum SierraAssertError {
    #[error("casts always happen between numerical types")]
    Cast,
    #[error("range should always intersect, from {:?} to {:?}", ranges.0, ranges.1)]
    Range { ranges: Box<(Range, Range)> },
    #[error("type {:?} should never be initialized", .0)]
    BadTypeInit(ConcreteTypeId),
    #[error("expected type information was missing")]
    BadTypeInfo,
    #[error("circuit cannot be evaluated")]
    ImpossibleCircuit,
}

#[derive(Debug)]
pub struct NativeAssertError {
    msg: String,
    info: BacktraceOrLocation,
}

impl std::error::Error for NativeAssertError {}

impl NativeAssertError {
    pub fn new(msg: String) -> Self {
        let backtrace = Backtrace::capture();
        let info = if let BacktraceStatus::Captured = backtrace.status() {
            BacktraceOrLocation::Backtrace(backtrace)
        } else {
            BacktraceOrLocation::Location(std::panic::Location::caller())
        };

        Self { msg, info }
    }
}

pub trait ToNativeExpect<T> {
    fn native_expect(self, msg: &str) -> Result<T>;
}

impl<T> ToNativeExpect<T> for Option<T> {
    fn native_expect(self, msg: &str) -> Result<T> {
        self.ok_or_else(|| Error::NativeAssert(NativeAssertError::new(msg.to_string())))
    }
}

impl<T> ToNativeExpect<T> for Result<T> {
    fn native_expect(self, msg: &str) -> Result<T> {
        self.map_err(|_| Error::NativeAssert(NativeAssertError::new(msg.to_string())))
    }
}

#[macro_export]
macro_rules! native_panic {
    ($arg:tt) => {
        return Err($crate::error::Error::NativeAssert(
            $crate::error::NativeAssertError::new(format!($arg)),
        ))
    };
}

#[derive(Debug)]
enum BacktraceOrLocation {
    Backtrace(Backtrace),
    Location(&'static Location<'static>),
}

impl std::fmt::Display for NativeAssertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", &self.msg)?;
        match &self.info {
            BacktraceOrLocation::Backtrace(backtrace) => {
                writeln!(f, "Stack backtrace:\n{}", backtrace)
            }
            BacktraceOrLocation::Location(location) => {
                writeln!(f, "Location: {}", location)
            }
        }
    }
}

#[derive(Error, Debug)]
pub enum CompilerError {
    #[error("BoundedInt value is out of range: {:?} not within [{:?}, {:?})", value, range.0, range.1)]
    BoundedIntOutOfRange {
        value: Box<BigInt>,
        range: Box<(BigInt, BigInt)>,
    },
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_make_missing_parameter() {
        // Test with a type ID that has a debug name
        let ty_with_debug_name = ConcreteTypeId {
            debug_name: Some("u32".into()),
            id: 10,
        };

        assert_eq!(
            Error::make_missing_parameter(&ty_with_debug_name).to_string(),
            "missing parameter of type 'u32'"
        );

        // Test with a type ID that does not have a debug name
        let ty_without_debug_name = ConcreteTypeId {
            debug_name: None,
            id: 10,
        };

        assert_eq!(
            Error::make_missing_parameter(&ty_without_debug_name).to_string(),
            "missing parameter of type ''"
        );
    }
}
