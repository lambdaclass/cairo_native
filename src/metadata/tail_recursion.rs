#[derive(Debug)]
pub struct TailRecursionMeta {
    is_tail_recursive: bool,
}

impl TailRecursionMeta {
    pub fn new(is_tail_recursive: bool) -> Self {
        Self { is_tail_recursive }
    }

    pub fn is_tail_recursive(&self) -> bool {
        self.is_tail_recursive
    }
}
