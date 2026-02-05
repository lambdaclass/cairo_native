use core::array::SpanTrait;
fn pop_elem(mut self: Span<u64>) -> Option<@u64> {
    let x = self.pop_back();
    x
}

fn run_test() -> Option<@u64> {
    let mut data = array![2].span();
    let x = pop_elem(data);
    x
}
