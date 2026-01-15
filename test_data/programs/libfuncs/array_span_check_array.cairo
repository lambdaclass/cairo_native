use core::array::SpanTrait;
fn pop_elem(mut self: Span<u64>) -> Option<@u64> {
    let x = self.pop_back();
    x
}

fn run_test() -> Array<u64> {
    let mut data = array![1, 2];
    let _x = pop_elem(data.span());
    data
}