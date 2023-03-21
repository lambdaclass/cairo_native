; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define internal { i256, i256 } @"dup<felt252>"(i256 %0) {
  %2 = insertvalue { i256, i256 } undef, i256 %0, 0
  %3 = insertvalue { i256, i256 } %2, i256 %0, 1
  ret { i256, i256 } %3
}

define internal i256 @felt252_add(i256 %0, i256 %1) {
  %3 = sext i256 %0 to i512
  %4 = add i512 %3, %3
  %5 = srem i512 %4, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  %6 = trunc i512 %5 to i256
  ret i256 %6
}

define internal i256 @felt252_sub(i256 %0, i256 %1) {
  %3 = sext i256 %0 to i512
  %4 = sub i512 %3, %3
  %5 = srem i512 %4, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  %6 = trunc i512 %5 to i256
  ret i256 %6
}

define internal { i256, i256 } @"struct_construct<Tuple<felt252, felt252>>"(i256 %0, i256 %1) {
  %3 = insertvalue { i256, i256 } undef, i256 %0, 0
  %4 = insertvalue { i256, i256 } %3, i256 %1, 1
  ret { i256, i256 } %4
}

define internal { i256, i256 } @"store_temp<Tuple<felt252, felt252>>"({ i256, i256 } %0) {
  ret { i256, i256 } %0
}

define { i256, i256 } @simple_simple_something(i256 %0) {
  %2 = sext i256 %0 to i512
  %3 = add i512 %2, %2
  %4 = srem i512 %3, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  %5 = trunc i512 %4 to i256
  %6 = sub i512 %2, %2
  %7 = srem i512 %6, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  %8 = trunc i512 %7 to i256
  %9 = insertvalue { i256, i256 } undef, i256 %5, 0
  %10 = insertvalue { i256, i256 } %9, i256 %8, 1
  ret { i256, i256 } %10
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
