; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @dprintf(i32, ptr, ...)

define internal i256 @felt252_add(i256 %0, i256 %1) {
  %3 = add i256 %0, %1
  %4 = icmp uge i256 %3, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  br i1 %4, label %6, label %5

5:                                                ; preds = %2
  ret i256 %3

6:                                                ; preds = %2
  %7 = sub i256 %3, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  ret i256 %7
}

define internal i256 @felt252_sub(i256 %0, i256 %1) {
  %3 = sub i256 %0, %1
  %4 = icmp slt i256 %3, 0
  br i1 %4, label %6, label %5

5:                                                ; preds = %2
  ret i256 %3

6:                                                ; preds = %2
  %7 = add i256 %3, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  ret i256 %7
}

define internal { i256, i256 } @"struct_construct<Tuple<felt252, felt252>>"(i256 %0, i256 %1) {
  %3 = insertvalue { i256, i256 } undef, i256 %0, 0
  %4 = insertvalue { i256, i256 } %3, i256 %1, 1
  ret { i256, i256 } %4
}

define { i256, i256 } @"simple::simple::something"(i256 %0) {
  br label %2

2:                                                ; preds = %1
  %3 = phi i256 [ %0, %1 ]
  %4 = add i256 %3, 2
  %5 = icmp uge i256 %4, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  br i1 %5, label %7, label %6

6:                                                ; preds = %2
  br label %9

7:                                                ; preds = %2
  %8 = sub i256 %4, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  br label %9

9:                                                ; preds = %7, %6
  %10 = phi i256 [ %8, %7 ], [ %4, %6 ]
  %11 = sub i256 %3, 2
  %12 = icmp slt i256 %11, 0
  br i1 %12, label %14, label %13

13:                                               ; preds = %9
  br label %16

14:                                               ; preds = %9
  %15 = add i256 %11, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  br label %16

16:                                               ; preds = %14, %13
  %17 = phi i256 [ %15, %14 ], [ %11, %13 ]
  %18 = insertvalue { i256, i256 } undef, i256 %10, 0
  %19 = insertvalue { i256, i256 } %18, i256 %17, 1
  ret { i256, i256 } %19
}

define void @"_mlir_ciface_simple::simple::something"(ptr %0, i256 %1) {
  br label %3

3:                                                ; preds = %2
  %4 = phi i256 [ %1, %2 ]
  %5 = add i256 %4, 2
  %6 = icmp uge i256 %5, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  br i1 %6, label %8, label %7

7:                                                ; preds = %3
  br label %10

8:                                                ; preds = %3
  %9 = sub i256 %5, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  br label %10

10:                                               ; preds = %8, %7
  %11 = phi i256 [ %9, %8 ], [ %5, %7 ]
  %12 = sub i256 %4, 2
  %13 = icmp slt i256 %12, 0
  br i1 %13, label %15, label %14

14:                                               ; preds = %10
  br label %17

15:                                               ; preds = %10
  %16 = add i256 %12, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  br label %17

17:                                               ; preds = %15, %14
  %18 = phi i256 [ %16, %15 ], [ %12, %14 ]
  %19 = insertvalue { i256, i256 } undef, i256 %11, 0
  %20 = insertvalue { i256, i256 } %19, i256 %18, 1
  br label %21

21:                                               ; preds = %17
  %22 = phi { i256, i256 } [ %20, %17 ]
  store { i256, i256 } %22, ptr %0, align 4
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
