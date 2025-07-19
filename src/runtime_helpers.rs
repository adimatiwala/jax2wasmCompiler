pub fn matmul_helper() -> String {
    r#"
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
    "#.to_string()
}

pub fn reduce_sum_helper() -> String {
    r#"
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
    "#.to_string()
}

pub fn elementwise_helpers() -> String {
    r#"
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
    "#.to_string()
}

pub fn math_helpers() -> String {
    r#"
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
    "#.to_string()
}
