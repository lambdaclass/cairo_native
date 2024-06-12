//use crate::{context::NativeContext, executor::JitNativeExecutor, OptLevel};
use crate::{context::NativeContext, executor::JitNativeExecutor, OptLevel};
//use cairo_lang_sierra::program::Program;
use cairo_lang_sierra::program::Program;
//use std::{
use std::{
//    collections::HashMap,
    collections::HashMap,
//    fmt::{self, Debug},
    fmt::{self, Debug},
//    hash::Hash,
    hash::Hash,
//    rc::Rc,
    rc::Rc,
//};
};
//

///// A Cache for programs with the same context.
/// A Cache for programs with the same context.
//pub struct JitProgramCache<'a, K>
pub struct JitProgramCache<'a, K>
//where
where
//    K: Eq + Hash + PartialEq,
    K: Eq + Hash + PartialEq,
//{
{
//    context: &'a NativeContext,
    context: &'a NativeContext,
//    // Since we already hold a reference to the Context, it doesn't make sense to use thread-safe
    // Since we already hold a reference to the Context, it doesn't make sense to use thread-safe
//    // reference counting. Using a Arc<RwLock<T>> here is useless because NativeExecutor is neither
    // reference counting. Using a Arc<RwLock<T>> here is useless because NativeExecutor is neither
//    // Send nor Sync.
    // Send nor Sync.
//    cache: HashMap<K, Rc<JitNativeExecutor<'a>>>,
    cache: HashMap<K, Rc<JitNativeExecutor<'a>>>,
//}
}
//

//impl<'a, K> JitProgramCache<'a, K>
impl<'a, K> JitProgramCache<'a, K>
//where
where
//    K: Eq + Hash + PartialEq,
    K: Eq + Hash + PartialEq,
//{
{
//    pub fn new(context: &'a NativeContext) -> Self {
    pub fn new(context: &'a NativeContext) -> Self {
//        Self {
        Self {
//            context,
            context,
//            cache: Default::default(),
            cache: Default::default(),
//        }
        }
//    }
    }
//

//    // Return the native context.
    // Return the native context.
//    pub const fn context(&self) -> &'a NativeContext {
    pub const fn context(&self) -> &'a NativeContext {
//        self.context
        self.context
//    }
    }
//

//    pub fn get(&self, key: &K) -> Option<Rc<JitNativeExecutor<'a>>> {
    pub fn get(&self, key: &K) -> Option<Rc<JitNativeExecutor<'a>>> {
//        self.cache.get(key).cloned()
        self.cache.get(key).cloned()
//    }
    }
//

//    pub fn compile_and_insert(
    pub fn compile_and_insert(
//        &mut self,
        &mut self,
//        key: K,
        key: K,
//        program: &Program,
        program: &Program,
//        opt_level: OptLevel,
        opt_level: OptLevel,
//    ) -> Rc<JitNativeExecutor<'a>> {
    ) -> Rc<JitNativeExecutor<'a>> {
//        let module = self.context.compile(program, None).expect("should compile");
        let module = self.context.compile(program, None).expect("should compile");
//        let executor = JitNativeExecutor::from_native_module(module, opt_level);
        let executor = JitNativeExecutor::from_native_module(module, opt_level);
//

//        let executor = Rc::new(executor);
        let executor = Rc::new(executor);
//        self.cache.insert(key, executor.clone());
        self.cache.insert(key, executor.clone());
//

//        executor
        executor
//    }
    }
//}
}
//

//impl<'a, K> Debug for JitProgramCache<'a, K>
impl<'a, K> Debug for JitProgramCache<'a, K>
//where
where
//    K: Eq + Hash + PartialEq,
    K: Eq + Hash + PartialEq,
//{
{
//    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//        f.write_str("JitProgramCache")
        f.write_str("JitProgramCache")
//    }
    }
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use super::*;
    use super::*;
//    use crate::utils::test::load_cairo;
    use crate::utils::test::load_cairo;
//    use std::time::Instant;
    use std::time::Instant;
//

//    #[test]
    #[test]
//    fn test_cache() {
    fn test_cache() {
//        let (_, program1) = load_cairo!(
        let (_, program1) = load_cairo!(
//            fn main(lhs: felt252, rhs: felt252) -> felt252 {
            fn main(lhs: felt252, rhs: felt252) -> felt252 {
//                lhs + rhs
                lhs + rhs
//            }
            }
//        );
        );
//

//        let (_, program2) = load_cairo!(
        let (_, program2) = load_cairo!(
//            fn main(lhs: felt252, rhs: felt252) -> felt252 {
            fn main(lhs: felt252, rhs: felt252) -> felt252 {
//                lhs - rhs
                lhs - rhs
//            }
            }
//        );
        );
//

//        let context = NativeContext::new();
        let context = NativeContext::new();
//        let mut cache: JitProgramCache<&'static str> = JitProgramCache::new(&context);
        let mut cache: JitProgramCache<&'static str> = JitProgramCache::new(&context);
//

//        let start = Instant::now();
        let start = Instant::now();
//        cache.compile_and_insert("program1", &program1, Default::default());
        cache.compile_and_insert("program1", &program1, Default::default());
//        let diff_1 = Instant::now().duration_since(start);
        let diff_1 = Instant::now().duration_since(start);
//

//        let start = Instant::now();
        let start = Instant::now();
//        cache.get(&"program1").expect("exists");
        cache.get(&"program1").expect("exists");
//        let diff_2 = Instant::now().duration_since(start);
        let diff_2 = Instant::now().duration_since(start);
//

//        assert!(diff_2 < diff_1);
        assert!(diff_2 < diff_1);
//

//        let start = Instant::now();
        let start = Instant::now();
//        cache.compile_and_insert("program2", &program2, Default::default());
        cache.compile_and_insert("program2", &program2, Default::default());
//        let diff_1 = Instant::now().duration_since(start);
        let diff_1 = Instant::now().duration_since(start);
//

//        let start = Instant::now();
        let start = Instant::now();
//        cache.get(&"program2").expect("exists");
        cache.get(&"program2").expect("exists");
//        let diff_2 = Instant::now().duration_since(start);
        let diff_2 = Instant::now().duration_since(start);
//

//        assert!(diff_2 < diff_1);
        assert!(diff_2 < diff_1);
//    }
    }
//}
}
