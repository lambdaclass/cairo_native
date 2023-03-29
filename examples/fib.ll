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

define internal i256 @"store_temp<felt252>"(i256 %0) {
  ret i256 %0
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

define internal i256 @"rename<felt252>"(i256 %0) {
  ret i256 %0
}

define internal {} @"struct_construct<Unit>"() {
  ret {} undef
}

define internal {} @"store_temp<Unit>"({} %0) {
  ret {} %0
}

define internal void @print_Unit({} %0) {
  ret void
}

define void @main(i256 %0) {
  %2 = call {} @fib_fib_main(i256 %0)
  call void @print_Unit({} %2)
  ret void
}

define void @_mlir_ciface_main(i256 %0) {
  call void @main(i256 %0)
  ret void
}

define internal i256 @fib_fib_fib(i256 %0, i256 %1, i256 %2) {
  br label %4

4:                                                ; preds = %3
  %5 = phi i256 [ %0, %3 ]
  %6 = phi i256 [ %1, %3 ]
  %7 = phi i256 [ %2, %3 ]
  %8 = call { i256, i256 } @"dup<felt252>"(i256 %7)
  %9 = extractvalue { i256, i256 } %8, 0
  %10 = extractvalue { i256, i256 } %8, 1
  %11 = icmp eq i256 %10, 0
  br i1 %11, label %12, label %15

12:                                               ; preds = %4
  %13 = phi i256 [ %5, %4 ]
  %14 = call i256 @"store_temp<felt252>"(i256 %13)
  br label %29

15:                                               ; preds = %4
  %16 = phi i256 [ %5, %4 ]
  %17 = phi i256 [ %6, %4 ]
  %18 = phi i256 [ %9, %4 ]
  %19 = call { i256, i256 } @"dup<felt252>"(i256 %17)
  %20 = extractvalue { i256, i256 } %19, 0
  %21 = extractvalue { i256, i256 } %19, 1
  %22 = call i256 @felt252_add(i256 %16, i256 %21)
  %23 = call i256 @felt252_sub(i256 %18, i256 1)
  %24 = call i256 @"store_temp<felt252>"(i256 %20)
  %25 = call i256 @"store_temp<felt252>"(i256 %22)
  %26 = call i256 @"store_temp<felt252>"(i256 %23)
  %27 = call i256 @fib_fib_fib(i256 %24, i256 %25, i256 %26)
  %28 = call i256 @"rename<felt252>"(i256 %27)
  br label %29

29:                                               ; preds = %12, %15
  %30 = phi i256 [ %28, %15 ], [ %14, %12 ]
  %31 = call i256 @"rename<felt252>"(i256 %30)
  ret i256 %31
}

define internal {} @fib_fib_fib_mid(i256 %0) {
  br label %2

2:                                                ; preds = %1
  %3 = phi i256 [ %0, %1 ]
  %4 = call { i256, i256 } @"dup<felt252>"(i256 %3)
  %5 = extractvalue { i256, i256 } %4, 0
  %6 = extractvalue { i256, i256 } %4, 1
  %7 = icmp eq i256 %6, 0
  br i1 %7, label %8, label %9

8:                                                ; preds = %2
  br label %18

9:                                                ; preds = %2
  %10 = phi i256 [ %5, %2 ]
  %11 = call i256 @"store_temp<felt252>"(i256 0)
  %12 = call i256 @"store_temp<felt252>"(i256 1)
  %13 = call i256 @"store_temp<felt252>"(i256 500)
  %14 = call i256 @fib_fib_fib(i256 %11, i256 %12, i256 %13)
  %15 = call i256 @felt252_sub(i256 %10, i256 1)
  %16 = call i256 @"store_temp<felt252>"(i256 %15)
  %17 = call {} @fib_fib_fib_mid(i256 %16)
  br label %18

18:                                               ; preds = %8, %9
  %19 = call {} @"struct_construct<Unit>"()
  %20 = call {} @"store_temp<Unit>"({} %19)
  ret {} %20
}

define internal {} @fib_fib_main(i256 %0) {
  br label %2

2:                                                ; preds = %1
  %3 = call i256 @"store_temp<felt252>"(i256 100)
  %4 = call {} @fib_fib_fib_mid(i256 %3)
  %5 = call {} @"struct_construct<Unit>"()
  %6 = call {} @"store_temp<Unit>"({} %5)
  ret {} %6
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
