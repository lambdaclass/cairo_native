use std::{cell::RefCell, collections::HashMap, fmt::Debug, hash::Hash, rc::Rc};

use cairo_lang_sierra::program::Program;

use crate::{context::NativeContext, module::NativeModule};

/// A Cache for programs with the same context.
pub struct ProgramCache<'a, K: PartialEq + Eq + Hash> {
    context: &'a NativeContext,
    // Since we already hold a reference to the Context, it doesn't make sense to use thread-safe refcounting.
    cache: HashMap<K, Rc<RefCell<NativeModule<'a>>>>,
}

impl<'a, K: PartialEq + Eq + Hash> Debug for ProgramCache<'a, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ProgramCache")
    }
}

impl<'a, K: PartialEq + Eq + Hash> ProgramCache<'a, K> {
    pub fn new(context: &'a NativeContext) -> Self {
        Self {
            context,
            cache: Default::default(),
        }
    }

    /// Checks if the program identified by the given key is already compiled and returns it or compiles it.
    pub fn compile_or_get(&mut self, key: K, program: &Program) -> Rc<RefCell<NativeModule<'a>>> {
        let module = self
            .cache
            .entry(key)
            .or_insert_with(|| {
                Rc::new(RefCell::new(
                    self.context.compile(program).expect("should compile"),
                ))
            })
            .clone();

        module
    }
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    use crate::utils::test::load_cairo;

    use super::*;

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
        cache.compile_or_get("program1", &program1);
        let diff_1 = Instant::now().duration_since(start);

        let start = Instant::now();
        cache.compile_or_get("program1", &program1);
        let diff_2 = Instant::now().duration_since(start);

        assert!(diff_2 < diff_1);

        let start = Instant::now();
        cache.compile_or_get("program2", &program2);
        let diff_1 = Instant::now().duration_since(start);

        let start = Instant::now();
        cache.compile_or_get("program2", &program2);
        let diff_2 = Instant::now().duration_since(start);

        assert!(diff_2 < diff_1);
    }
}
