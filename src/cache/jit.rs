use crate::{context::NativeContext, executor::JitNativeExecutor, OptLevel};
use cairo_lang_sierra::program::Program;
use std::{
    collections::HashMap,
    fmt::{self, Debug},
    hash::Hash,
    rc::Rc,
};

/// A Cache for programs with the same context.
pub struct JitProgramCache<'a, K>
where
    K: Eq + Hash + PartialEq,
{
    context: &'a NativeContext,
    // Since we already hold a reference to the Context, it doesn't make sense to use thread-safe
    // reference counting. Using a Arc<RwLock<T>> here is useless because NativeExecutor is neither
    // Send nor Sync.
    cache: HashMap<K, Rc<JitNativeExecutor<'a>>>,
}

impl<'a, K> JitProgramCache<'a, K>
where
    K: Eq + Hash + PartialEq,
{
    pub fn new(context: &'a NativeContext) -> Self {
        Self {
            context,
            cache: Default::default(),
        }
    }

    // Return the native context.
    pub const fn context(&self) -> &'a NativeContext {
        self.context
    }

    pub fn get(&self, key: &K) -> Option<Rc<JitNativeExecutor<'a>>> {
        self.cache.get(key).cloned()
    }

    pub fn compile_and_insert(
        &mut self,
        key: K,
        program: &Program,
        opt_level: OptLevel,
    ) -> Rc<JitNativeExecutor<'a>> {
        let module = self.context.compile(program, None).expect("should compile");
        let executor = JitNativeExecutor::from_native_module(module, opt_level);

        let executor = Rc::new(executor);
        self.cache.insert(key, executor.clone());

        executor
    }
}

impl<'a, K> Debug for JitProgramCache<'a, K>
where
    K: Eq + Hash + PartialEq,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("JitProgramCache")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::test::load_cairo;
    use std::time::Instant;

    #[test]
    fn test_cache() {
        let (_, program1) = load_cairo!(
            fn main(lhs: felt252, rhs: felt252) -> felt252 {
                lhs + rhs
            }
        );

        let (_, program2) = load_cairo!(
            fn main(lhs: felt252, rhs: felt252) -> felt252 {
                lhs - rhs
            }
        );

        let context = NativeContext::new();
        let mut cache: JitProgramCache<&'static str> = JitProgramCache::new(&context);

        let start = Instant::now();
        cache.compile_and_insert("program1", &program1, Default::default());
        let diff_1 = Instant::now().duration_since(start);

        let start = Instant::now();
        cache.get(&"program1").expect("exists");
        let diff_2 = Instant::now().duration_since(start);

        assert!(diff_2 < diff_1);

        let start = Instant::now();
        cache.compile_and_insert("program2", &program2, Default::default());
        let diff_1 = Instant::now().duration_since(start);

        let start = Instant::now();
        cache.get(&"program2").expect("exists");
        let diff_2 = Instant::now().duration_since(start);

        assert!(diff_2 < diff_1);
    }
}
