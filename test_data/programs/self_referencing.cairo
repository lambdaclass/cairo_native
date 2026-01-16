#[derive(Drop, Copy, PartialEq)]
enum ArrayItem {
    Span: Span<u8>,
    Recursive: Span<ArrayItem>
}

fn recursion(input: Span<u8>) -> Span<ArrayItem> {
    let mut output: Array<ArrayItem> = Default::default();

    let index = (*input.at(0));
    if index < 5 {
        output.append(ArrayItem::Span(input));
    } else {
        let res = recursion(input.slice(1, input.len() - 1));
        output.append(ArrayItem::Recursive(res));
    }

    return output.span();
}

fn run_test() -> Span<ArrayItem> {
    let arr = array![10, 9, 8, 7, 6, 4];
    recursion(arr.span())
}
