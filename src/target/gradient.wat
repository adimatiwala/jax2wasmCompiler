(module
  ;; Generated from Gradient Pass HLO
  (memory (export "memory") 2)

  (func $matmul (param $lhs i32) (param $rhs i32) (param $out i32)
    (local $i i32) (local $j i32) (local $k i32) (local $sum f32)
    ;; Simple 4x4 matrix multiplication for demonstration
    (local.set $i (i32.const 0))
    (loop $loop_i
      (br_if $exit_i (i32.ge_s (local.get $i) (i32.const 4)))
      (local.set $j (i32.const 0))
      (loop $loop_j
        (br_if $exit_j (i32.ge_s (local.get $j) (i32.const 4)))
        (local.set $sum (f32.const 0))
        (local.set $k (i32.const 0))
        (loop $loop_k
          (br_if $exit_k (i32.ge_s (local.get $k) (i32.const 4)))
          (local.set $sum
            (f32.add (local.get $sum)
              (f32.mul
                (f32.load (i32.add (local.get $lhs)
                  (i32.mul (i32.add (i32.mul (local.get $i) (i32.const 4)) (local.get $k)) (i32.const 4))))
                (f32.load (i32.add (local.get $rhs)
                  (i32.mul (i32.add (i32.mul (local.get $k) (i32.const 4)) (local.get $j)) (i32.const 4))))
              )
            )
          )
          (local.set $k (i32.add (local.get $k) (i32.const 1)))
          (br $loop_k)
        )
        (f32.store
          (i32.add (local.get $out)
            (i32.mul (i32.add (i32.mul (local.get $i) (i32.const 4)) (local.get $j)) (i32.const 4)))
          (local.get $sum)
        )
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (br $loop_j)
        $exit_j
      )
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $loop_i)
      $exit_i
    )
  )
    

  (func $reduce_sum (param $ptr i32) (param $len i32) (result f32)
    (local $i i32) (local $sum f32)
    (local.set $i (i32.const 0))
    (local.set $sum (f32.const 0))
    (loop $reduce_loop
      (br_if $exit (i32.ge_s (local.get $i) (local.get $len)))
      (local.set $sum
        (f32.add (local.get $sum)
          (f32.load (i32.add (local.get $ptr) (i32.mul (local.get $i) (i32.const 4)))))
      )
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $reduce_loop)
    )
    (return (local.get $sum))
    $exit
  )
    

  (func $add_arrays (param $a i32) (param $b i32) (param $out i32) (param $len i32)
    (local $i i32)
    (local.set $i (i32.const 0))
    (loop $loop
      (br_if $exit (i32.ge_s (local.get $i) (local.get $len)))
      (f32.store
        (i32.add (local.get $out) (i32.mul (local.get $i) (i32.const 4)))
        (f32.add
          (f32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4))))
          (f32.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4))))
        )
      )
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $loop)
    )
    $exit
  )
    

  (func $exp (param $x f32) (result f32)
    ;; Simplified exp approximation
    (local $result f32)
    (local.set $result (f32.const 2.718281828))
    ;; This would need a proper implementation
    (return (local.get $result))
  )

  (func $log (param $x f32) (result f32)
    ;; Simplified log approximation
    (return (f32.const 0.693147)) ;; ln(2) as placeholder
  )

  (func $tanh (param $x f32) (result f32)
    ;; Simplified tanh approximation
    (local $exp_pos f32)
    (local $exp_neg f32)
    (local.set $exp_pos (call $exp (local.get $x)))
    (local.set $exp_neg (call $exp (f32.neg (local.get $x))))
    (return 
      (f32.div
        (f32.sub (local.get $exp_pos) (local.get $exp_neg))
        (f32.add (local.get $exp_pos) (local.get $exp_neg))
      )
    )
  )
    

  (func $simd_add (param $a i32) (param $b i32) (param $out i32) (param $len i32)
    (local $i i32)
    (local.set $i (i32.const 0))
    (loop $simd_loop
      (br_if $exit (i32.ge_s (local.get $i) (local.get $len)))
      ;; Process 4 floats at once using SIMD
      (v128.store 
        (i32.add (local.get $out) (i32.mul (local.get $i) (i32.const 4)))
        (f32x4.add
          (v128.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4))))
          (v128.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4))))
        )
      )
      (local.set $i (i32.add (local.get $i) (i32.const 4))) ;; Skip by 4 floats
      (br $simd_loop)
    )
    $exit
  )
    

  (func $simd_mul (param $a i32) (param $b i32) (param $out i32) (param $len i32)
    (local $i i32)
    (local.set $i (i32.const 0))
    (loop $simd_loop
      (br_if $exit (i32.ge_s (local.get $i) (local.get $len)))
      (v128.store 
        (i32.add (local.get $out) (i32.mul (local.get $i) (i32.const 4)))
        (f32x4.mul
          (v128.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4))))
          (v128.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4))))
        )
      )
      (local.set $i (i32.add (local.get $i) (i32.const 4)))
      (br $simd_loop)
    )
    $exit
  )

  (func $simd_reduce_sum (param $ptr i32) (param $len i32) (result f32)
    (local $i i32) (local $sum v128) (local $scalar_sum f32)
    (local.set $sum (v128.const f32x4 0 0 0 0))
    (local.set $i (i32.const 0))
    
    ;; SIMD reduction loop
    (loop $simd_reduce_loop
      (br_if $simd_done (i32.ge_s (local.get $i) (local.get $len)))
      (local.set $sum
        (f32x4.add (local.get $sum)
          (v128.load (i32.add (local.get $ptr) (i32.mul (local.get $i) (i32.const 4))))
        )
      )
      (local.set $i (i32.add (local.get $i) (i32.const 4)))
      (br $simd_reduce_loop)
    )
    
    ;; Extract and sum the 4 components
    (local.set $scalar_sum
      (f32.add
        (f32.add
          (f32x4.extract_lane 0 (local.get $sum))
          (f32x4.extract_lane 1 (local.get $sum))
        )
        (f32.add
          (f32x4.extract_lane 2 (local.get $sum))
          (f32x4.extract_lane 3 (local.get $sum))
        )
      )
    )
    
    (return (local.get $scalar_sum))
    $simd_done
  )
    
  (func $main (export "main") (param $p0 f32) (param $p1 f32) (result f32)
    (local $sine.1 f32)
    (local $constant.1 f32)
    (local $multiply.1 f32)
    (local $cosine.1 f32)
    (local $ROOT %add.1 f32)
    (local $ROOT %cosine_add_fusion f32)
    (f32.store (i32.const 2144) (local.get $p0))
    (f32.store (i32.const 2048) (local.get $p1))
    (local.set $sine.1 (f32.const 0.0)) ;; TODO: Implement sine(%param_0.1),
    (local.set $constant.1 (f32.const 0.0)) ;; TODO: Implement constant(-1)
    (local.set $multiply.1 (f32.const 0.0)) ;; TODO: Implement multiply(%sine.1,
    (local.set $cosine.1 (f32.const 0.0)) ;; TODO: Implement cosine(%param_0.1),
    (local.set $ROOT %add.1 (f32.const 0.0)) ;; TODO: Implement add(%multiply.1,
    (local.set $ROOT %cosine_add_fusion (f32.const 0.0)) ;; TODO: Implement fusion(%Arg_0.1),
    (local.get $ROOT %cosine_add_fusion)
  )
)