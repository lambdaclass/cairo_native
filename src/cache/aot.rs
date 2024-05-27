use crate::{context::NativeContext, executor::AotNativeExecutor, OptLevel};
use cairo_lang_sierra::program::Program;
use std::{
    collections::HashMap,
    fmt::{self, Debug},
    hash::Hash,
    rc::Rc,
};

pub struct AotProgramCache<'a, K>
where
    K: PartialEq + Eq + Hash,
{
    context: &'a NativeContext,
    cache: HashMap<K, Rc<AotNativeExecutor<'a>>>,
}

impl<'a, K> AotProgramCache<'a, K>
where
    K: PartialEq + Eq + Hash,
{
    pub fn new(context: &'a NativeContext) -> Self {
        Self {
            context,
            cache: Default::default(),
        }
    }

    pub fn get(&self, key: &K) -> Option<Rc<AotNativeExecutor<'a>>> {
        self.cache.get(key).cloned()
    }

    pub fn compile_and_insert(
        &mut self,
        key: K,
        program: &Program,
        opt_level: OptLevel,
    ) -> Rc<AotNativeExecutor<'a>> {
        let module = self.context.compile(program, None).expect("should compile");
        let executor = Rc::new(AotNativeExecutor::from_native_module(
            self.context.context(),
            module,
            opt_level,
        ));

        self.cache.insert(key, executor.clone());
        executor
    }
}

impl<'a, K> Debug for AotProgramCache<'a, K>
where
    K: PartialEq + Eq + Hash,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("AotProgramCache")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{utils::test::load_cairo, values::JitValue};
    use starknet_types_core::felt::Felt;

    #[test]
    fn test_aot_compile_and_insert() {
        let native_context = NativeContext::new();
        let mut cache = AotProgramCache::new(&native_context);

        let (_, program) = load_cairo! {
            fn run_test() -> felt252 {
                42
            }
        };

        let function_id = &program.funcs.first().expect("should have a function").id;
        let executor = cache.compile_and_insert((), &program, OptLevel::default());
        let res = executor
            .invoke_dynamic(function_id, &[], Some(u128::MAX))
            .expect("should run");

        // After compiling and inserting the program, we should be able to run it.
        assert_eq!(res.return_value, JitValue::Felt252(Felt::from(42)));
    }
}
