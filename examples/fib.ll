; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @realloc(ptr, i64)

declare i32 @dprintf(i32, ptr, ...)

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
define internal {} @"struct_construct<Unit>"() #0 {
  ret {} undef
}

define i256 @"fib::fib::fib"(i256 %0, i256 %1, i256 %2) {
  br label %4

4:                                                ; preds = %3
  %5 = phi i256 [ %0, %3 ]
  %6 = phi i256 [ %1, %3 ]
  %7 = phi i256 [ %2, %3 ]
  %8 = icmp eq i256 %7, 0
  br i1 %8, label %9, label %11

9:                                                ; preds = %4
  %10 = phi i256 [ %5, %4 ]
  br label %18

11:                                               ; preds = %4
  %12 = phi i256 [ %5, %4 ]
  %13 = phi i256 [ %6, %4 ]
  %14 = phi i256 [ %7, %4 ]
  %15 = call i256 @felt252_add(i256 %12, i256 %13)
  %16 = call i256 @felt252_sub(i256 %14, i256 1)
  %17 = call i256 @"fib::fib::fib"(i256 %13, i256 %15, i256 %16)
  br label %18

18:                                               ; preds = %9, %11
  %19 = phi i256 [ %17, %11 ], [ %10, %9 ]
  ret i256 %19
}

define i256 @"_mlir_ciface_fib::fib::fib"(i256 %0, i256 %1, i256 %2) {
  %4 = call i256 @"fib::fib::fib"(i256 %0, i256 %1, i256 %2)
  ret i256 %4
}

define {} @"fib::fib::fib_mid"(i256 %0) {
  br label %2

2:                                                ; preds = %1
  %3 = phi i256 [ %0, %1 ]
  %4 = icmp eq i256 %3, 0
  br i1 %4, label %5, label %6

5:                                                ; preds = %2
  br label %11

6:                                                ; preds = %2
  %7 = phi i256 [ %3, %2 ]
  %8 = call i256 @"fib::fib::fib"(i256 0, i256 1, i256 500)
  %9 = call i256 @felt252_sub(i256 %7, i256 1)
  %10 = call {} @"fib::fib::fib_mid"(i256 %9)
  br label %11

11:                                               ; preds = %5, %6
  %12 = call {} @"struct_construct<Unit>"()
  ret {} %12
}

define void @"_mlir_ciface_fib::fib::fib_mid"(ptr %0, i256 %1) {
  %3 = call {} @"fib::fib::fib_mid"(i256 %1)
  store {} %3, ptr %0, align 1
  ret void
}

define {} @"fib::fib::main"(i256 %0) {
  br label %2

2:                                                ; preds = %1
  %3 = call {} @"fib::fib::fib_mid"(i256 100)
  %4 = call {} @"struct_construct<Unit>"()
  ret {} %4
}

define void @"_mlir_ciface_fib::fib::main"(ptr %0, i256 %1) {
  %3 = call {} @"fib::fib::main"(i256 %1)
  store {} %3, ptr %0, align 1
  ret void
}

attributes #0 = { alwaysinline norecurse nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
