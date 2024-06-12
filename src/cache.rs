//pub use self::{aot::AotProgramCache, jit::JitProgramCache};
pub use self::{aot::AotProgramCache, jit::JitProgramCache};
//use std::hash::Hash;
use std::hash::Hash;
//

//pub mod aot;
pub mod aot;
//pub mod jit;
pub mod jit;
//

//#[derive(Debug)]
#[derive(Debug)]
//pub enum ProgramCache<'a, K>
pub enum ProgramCache<'a, K>
//where
where
//    K: PartialEq + Eq + Hash,
    K: PartialEq + Eq + Hash,
//{
{
//    Aot(AotProgramCache<'a, K>),
    Aot(AotProgramCache<'a, K>),
//    Jit(JitProgramCache<'a, K>),
    Jit(JitProgramCache<'a, K>),
//}
}
//

//impl<'a, K> From<AotProgramCache<'a, K>> for ProgramCache<'a, K>
impl<'a, K> From<AotProgramCache<'a, K>> for ProgramCache<'a, K>
//where
where
//    K: PartialEq + Eq + Hash,
    K: PartialEq + Eq + Hash,
//{
{
//    fn from(value: AotProgramCache<'a, K>) -> Self {
    fn from(value: AotProgramCache<'a, K>) -> Self {
//        Self::Aot(value)
        Self::Aot(value)
//    }
    }
//}
}
//

//impl<'a, K> From<JitProgramCache<'a, K>> for ProgramCache<'a, K>
impl<'a, K> From<JitProgramCache<'a, K>> for ProgramCache<'a, K>
//where
where
//    K: PartialEq + Eq + Hash,
    K: PartialEq + Eq + Hash,
//{
{
//    fn from(value: JitProgramCache<'a, K>) -> Self {
    fn from(value: JitProgramCache<'a, K>) -> Self {
//        Self::Jit(value)
        Self::Jit(value)
//    }
    }
//}
}
