//! # Executor caching

pub use self::{aot::AotProgramCache, jit::JitProgramCache};
use std::hash::Hash;

mod aot;
mod jit;

/// A cache to store compiled programs.
#[derive(Debug)]
pub enum ProgramCache<'a, K>
where
    K: PartialEq + Eq + Hash,
{
    /// Cache is for AOT-compiled programs.
    Aot(AotProgramCache<'a, K>),
    /// Cache is for JIT-compiled programs.
    Jit(JitProgramCache<'a, K>),
}

impl<'a, K> From<AotProgramCache<'a, K>> for ProgramCache<'a, K>
where
    K: PartialEq + Eq + Hash,
{
    fn from(value: AotProgramCache<'a, K>) -> Self {
        Self::Aot(value)
    }
}

impl<'a, K> From<JitProgramCache<'a, K>> for ProgramCache<'a, K>
where
    K: PartialEq + Eq + Hash,
{
    fn from(value: JitProgramCache<'a, K>) -> Self {
        Self::Jit(value)
    }
}
