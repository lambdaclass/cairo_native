use crate::{context::NativeContext, executor::NativeExecutor};
use cairo_lang_sierra::program::Program;
use std::{cell::RefCell, collections::HashMap, fmt::Debug, hash::Hash, rc::Rc};

/// A Cache for programs with the same context.
pub struct ProgramCache<'a, K: PartialEq + Eq + Hash> {
    context: &'a NativeContext,
    // Since we already hold a reference to the Context, it doesn't make sense to use thread-safe
    // reference counting. Using a Arc<RwLock<T>> here is useless because NativeExecutor is neither
    // Send nor Sync.
    cache: HashMap<K, Rc<RefCell<NativeExecutor<'a>>>>,
}

impl<'a, K> ProgramCache<'a, K>
where
    K: Eq + Hash + PartialEq,
{
    // Return the native context.
    pub const fn context(&self) -> &'a NativeContext {
        self.context
    }
}

impl<'a, K: PartialEq + Eq + Hash> Debug for ProgramCache<'a, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ProgramCache")
    }
}

impl<'a, K: Clone + PartialEq + Eq + Hash> ProgramCache<'a, K> {
    pub fn new(context: &'a NativeContext) -> Self {
        Self {
            context,
            cache: Default::default(),
        }
    }

    pub fn get(&self, key: K) -> Option<Rc<RefCell<NativeExecutor<'a>>>> {
        self.cache.get(&key).cloned()
    }

    pub fn compile_and_insert(
        &mut self,
        key: K,
        program: &Program,
    ) -> Rc<RefCell<NativeExecutor<'a>>> {
        let module = self.context.compile(program).expect("should compile");
        let executor = NativeExecutor::new(module);
        self.cache
            .insert(key.clone(), Rc::new(RefCell::new(executor)));
        self.cache.get_mut(&key).cloned().unwrap()
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
        let mut cache: ProgramCache<&'static str> = ProgramCache::new(&context);

        let start = Instant::now();
        cache.compile_and_insert("program1", &program1);
        let diff_1 = Instant::now().duration_since(start);

        let start = Instant::now();
        cache.get("program1").expect("exists");
        let diff_2 = Instant::now().duration_since(start);

        assert!(diff_2 < diff_1);

        let start = Instant::now();
        cache.compile_and_insert("program2", &program2);
        let diff_1 = Instant::now().duration_since(start);

        let start = Instant::now();
        cache.get("program2").expect("exists");
        let diff_2 = Instant::now().duration_since(start);

        assert!(diff_2 < diff_1);
    }
}
