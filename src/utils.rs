use cairo_lang_sierra::ids::FunctionId;
use std::borrow::Cow;

pub fn generate_function_name(function_id: &FunctionId) -> Cow<str> {
    function_id
        .debug_name
        .as_deref()
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(format!("f{}", function_id.id)))
}
