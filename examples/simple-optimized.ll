; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @dprintf(i32, ptr, ...)

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

define { i256, i256 } @simple_simple_something(i256 %0) {
  %2 = add i256 %0, 2
  %3 = icmp uge i256 %2, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  %5 = add i256 %0, 1
  br label %6

6:                                                ; preds = %4, %1
  %7 = phi i256 [ %5, %4 ], [ %2, %1 ]
  %8 = sub i256 %0, 2
  %9 = icmp ult i256 %0, 2
  br i1 %9, label %10, label %12

10:                                               ; preds = %6
  %11 = sub i256 %0, 3
  br label %12

12:                                               ; preds = %10, %6
  %13 = phi i256 [ %11, %10 ], [ %8, %6 ]
  %14 = insertvalue { i256, i256 } undef, i256 %7, 0
  %15 = insertvalue { i256, i256 } %14, i256 %13, 1
  ret { i256, i256 } %15
}

define void @_mlir_ciface_simple_simple_something(ptr %0, i256 %1) {
  %3 = call { i256, i256 } @simple_simple_something(i256 %1)
  store { i256, i256 } %3, ptr %0, align 4
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
