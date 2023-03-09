; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define { i256, i256 } @fib(i256 %0, i256 %1, i256 %2) {
  %4 = icmp eq i256 %2, 0
  br i1 %4, label %5, label %6

5:                                                ; preds = %3
  br label %16

6:                                                ; preds = %3
  %7 = add i256 %0, %1
  %8 = srem i256 %7, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  %9 = sub i256 %2, 1
  %10 = srem i256 %9, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  %11 = call { i256, i256 } @fib(i256 %1, i256 %8, i256 %10)
  %12 = extractvalue { i256, i256 } %11, 0
  %13 = extractvalue { i256, i256 } %11, 1
  %14 = add i256 %13, 1
  %15 = srem i256 %14, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  br label %16

16:                                               ; preds = %5, %6
  %17 = phi i256 [ %12, %6 ], [ %0, %5 ]
  %18 = phi i256 [ %15, %6 ], [ 0, %5 ]
  br label %19

19:                                               ; preds = %16
  %20 = insertvalue { i256, i256 } undef, i256 %17, 0
  %21 = insertvalue { i256, i256 } %20, i256 %18, 1
  ret { i256, i256 } %21
}

define void @_mlir_ciface_fib(ptr %0, i256 %1, i256 %2, i256 %3) {
  %5 = call { i256, i256 } @fib(i256 %1, i256 %2, i256 %3)
  store { i256, i256 } %5, ptr %0, align 4
  ret void
}

define void @fib_mid(i256 %0) {
  %2 = icmp eq i256 %0, 0
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  br label %10

4:                                                ; preds = %1
  %5 = call { i256, i256 } @fib(i256 0, i256 1, i256 500)
  %6 = extractvalue { i256, i256 } %5, 0
  %7 = extractvalue { i256, i256 } %5, 1
  %8 = sub i256 %0, 1
  %9 = srem i256 %8, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  call void @fib_mid(i256 %9)
  br label %10

10:                                               ; preds = %3, %4
  ret void
}

define void @_mlir_ciface_fib_mid(i256 %0) {
  call void @fib_mid(i256 %0)
  ret void
}

define i32 @main() {
  call void @fib_mid(i256 100)
  ret i32 0
}

define i32 @_mlir_ciface_main() {
  %1 = call i32 @main()
  ret i32 %1
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
