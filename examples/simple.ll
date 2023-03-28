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
  %3 = add i256 %0, %1
  %4 = icmp uge i256 %3, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  br i1 %4, label %7, label %5

5:                                                ; preds = %7, %2
  %6 = phi i256 [ %8, %7 ], [ %3, %2 ]
  ret i256 %6

7:                                                ; preds = %2
  %8 = sub i256 %3, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  br label %5
}

define internal i256 @felt252_sub(i256 %0, i256 %1) {
  %3 = sub i256 %0, %1
  %4 = icmp ult i256 %0, %1
  br i1 %4, label %7, label %5

5:                                                ; preds = %7, %2
  %6 = phi i256 [ %8, %7 ], [ %3, %2 ]
  ret i256 %6

7:                                                ; preds = %2
  %8 = sub i256 %3, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  br label %5
}

define internal { i256, i256 } @"struct_construct<Tuple<felt252, felt252>>"(i256 %0, i256 %1) {
  %3 = insertvalue { i256, i256 } undef, i256 %0, 0
  %4 = insertvalue { i256, i256 } %3, i256 %1, 1
  ret { i256, i256 } %4
}

define internal { i256, i256 } @"store_temp<Tuple<felt252, felt252>>"({ i256, i256 } %0) {
  ret { i256, i256 } %0
}

define internal { i256, i256 } @simple_simple_something(i256 %0) {
  br label %2

2:                                                ; preds = %1
  %3 = phi i256 [ %0, %1 ]
  %4 = call { i256, i256 } @"dup<felt252>"(i256 %3)
  %5 = extractvalue { i256, i256 } %4, 0
  %6 = extractvalue { i256, i256 } %4, 1
  %7 = call i256 @felt252_add(i256 %6, i256 2)
  %8 = call i256 @felt252_sub(i256 %5, i256 2)
  %9 = call { i256, i256 } @"struct_construct<Tuple<felt252, felt252>>"(i256 %7, i256 %8)
  %10 = call { i256, i256 } @"store_temp<Tuple<felt252, felt252>>"({ i256, i256 } %9)
  ret { i256, i256 } %10
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
