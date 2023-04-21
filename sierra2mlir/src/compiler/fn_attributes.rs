/// Function attributes to fine tune the generated definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FnAttributes {
    /// the function linkage
    pub public: bool,
    pub emit_c_interface: bool,
    /// true = dso_local https://llvm.org/docs/LangRef.html#runtime-preemption-specifiers
    pub local: bool,
    pub inline: bool,
    /// This function attribute indicates that the function does not call itself either directly or indirectly down any possible call path. This produces undefined behavior at runtime if the function ever does recurse.
    pub norecurse: bool,
    /// This function attribute indicates that the function never raises an exception. If the function does raise an exception, its runtime behavior is undefined.
    pub nounwind: bool, // never raises exception
}

impl Default for FnAttributes {
    fn default() -> Self {
        Self {
            public: true,
            emit_c_interface: false,
            local: true,
            inline: false,
            norecurse: false,
            nounwind: false,
        }
    }
}

impl FnAttributes {
    pub const fn libfunc(panics: bool, inline: bool) -> Self {
        Self {
            public: false,
            emit_c_interface: false,
            local: true,
            inline,
            norecurse: true,
            nounwind: !panics,
        }
    }
}
