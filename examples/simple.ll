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
  %4 = sext i256 %0 to i512
  %5 = add i512 %3, %4
  %6 = srem i512 %5, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  %7 = trunc i512 %6 to i256
  ret i256 %7
}

define internal i256 @felt252_sub(i256 %0, i256 %1) {
  %3 = sext i256 %0 to i512
  %4 = sext i256 %0 to i512
  %5 = sub i512 %3, %4
  %6 = srem i512 %5, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  %7 = trunc i512 %6 to i256
  ret i256 %7
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
  %2 = call { i256, i256 } @"dup<felt252>"(i256 %0)
  %3 = extractvalue { i256, i256 } %2, 0
  %4 = extractvalue { i256, i256 } %2, 1
  %5 = call i256 @felt252_add(i256 %4, i256 2)
  %6 = call i256 @felt252_sub(i256 %3, i256 2)
  %7 = call { i256, i256 } @"struct_construct<Tuple<felt252, felt252>>"(i256 %5, i256 %6)
  %8 = call { i256, i256 } @"store_temp<Tuple<felt252, felt252>>"({ i256, i256 } %7)
  ret { i256, i256 } %8
}

define void @_mlir_ciface_simple_simple_something(ptr %0, i256 %1) {
  %3 = call { i256, i256 } @simple_simple_something(i256 %1)
  store { i256, i256 } %3, ptr %0, align 4
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
