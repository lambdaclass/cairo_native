; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @dprintf(i32, ptr, ...)

; Function Attrs: alwaysinline norecurse nounwind
define internal { i256, i256 } @"dup<felt252>"(i256 %0) #0 {
  %2 = insertvalue { i256, i256 } undef, i256 %0, 0
  %3 = insertvalue { i256, i256 } %2, i256 %0, 1
  ret { i256, i256 } %3
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i256 @"store_temp<felt252>"(i256 %0) #0 {
  ret i256 %0
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i256 @felt252_add(i256 %0, i256 %1) #0 {
  %3 = add i256 %0, %1
  %4 = icmp uge i256 %3, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  br i1 %4, label %6, label %5

5:                                                ; preds = %2
  ret i256 %3

6:                                                ; preds = %2
  %7 = sub i256 %3, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  ret i256 %7
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i256 @felt252_sub(i256 %0, i256 %1) #0 {
  %3 = sub i256 %0, %1
  %4 = icmp slt i256 %3, 0
  br i1 %4, label %6, label %5

5:                                                ; preds = %2
  ret i256 %3

6:                                                ; preds = %2
  %7 = add i256 %3, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  ret i256 %7
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i256 @"rename<felt252>"(i256 %0) #0 {
  ret i256 %0
}

; Function Attrs: alwaysinline norecurse nounwind
define internal {} @"struct_construct<Unit>"() #0 {
  ret {} undef
}

; Function Attrs: alwaysinline norecurse nounwind
define internal {} @"store_temp<Unit>"({} %0) #0 {
  ret {} %0
}

define i256 @"fib::fib::fib"(i256 %0, i256 %1, i256 %2) {
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
  %27 = call i256 @"fib::fib::fib"(i256 %24, i256 %25, i256 %26)
  %28 = call i256 @"rename<felt252>"(i256 %27)
  br label %29

29:                                               ; preds = %12, %15
  %30 = phi i256 [ %28, %15 ], [ %14, %12 ]
  %31 = call i256 @"rename<felt252>"(i256 %30)
  ret i256 %31
}

define i256 @"_mlir_ciface_fib::fib::fib"(i256 %0, i256 %1, i256 %2) {
  %4 = call i256 @"fib::fib::fib"(i256 %0, i256 %1, i256 %2)
  ret i256 %4
}

define {} @"fib::fib::fib_mid"(i256 %0) {
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
  %14 = call i256 @"fib::fib::fib"(i256 %11, i256 %12, i256 %13)
  %15 = call i256 @felt252_sub(i256 %10, i256 1)
  %16 = call i256 @"store_temp<felt252>"(i256 %15)
  %17 = call {} @"fib::fib::fib_mid"(i256 %16)
  br label %18

18:                                               ; preds = %8, %9
  %19 = call {} @"struct_construct<Unit>"()
  %20 = call {} @"store_temp<Unit>"({} %19)
  ret {} %20
}

define void @"_mlir_ciface_fib::fib::fib_mid"(ptr %0, i256 %1) {
  %3 = call {} @"fib::fib::fib_mid"(i256 %1)
  store {} %3, ptr %0, align 1
  ret void
}

define {} @"fib::fib::main"(i256 %0) {
  br label %2

2:                                                ; preds = %1
  %3 = call i256 @"store_temp<felt252>"(i256 100)
  %4 = call {} @"fib::fib::fib_mid"(i256 %3)
  %5 = call {} @"struct_construct<Unit>"()
  %6 = call {} @"store_temp<Unit>"({} %5)
  ret {} %6
}

define void @"_mlir_ciface_fib::fib::main"(ptr %0, i256 %1) {
  %3 = call {} @"fib::fib::main"(i256 %1)
  store {} %3, ptr %0, align 1
  ret void
}

attributes #0 = { alwaysinline norecurse nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
