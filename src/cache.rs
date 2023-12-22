pub use self::{aot::AotProgramCache, jit::JitProgramCache};
use std::hash::Hash;

pub mod aot;
pub mod jit;

#[derive(Debug)]
pub enum ProgramCache<'a, K>
where
    K: PartialEq + Eq + Hash,
{
    Aot(AotProgramCache<'a, K>),
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
