use gas::withdraw_gas;

fn run_test() {
    let mut i = 10;

    loop {
        if i == 0 {
            break;
        }

        match withdraw_gas() {
            Option::Some(()) => {
                i = i - 1;
            },
            Option::None(()) => {
                break;
            }
        };
        i = i - 1;
    }
}
