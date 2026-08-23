fn run_test(data: Array<bool>) -> u32 {
    let mut count: u32 = 0;
    for value in data {
        if value {
            count += 1;
        }
    }
    count
}
