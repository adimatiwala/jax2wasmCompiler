use crate::hlo_parser::{parse_hlo, HloInstruction};
use crate::memory::MemoryManager;
use crate::runtime_helpers;
use crate::simd;

pub fn generate_wat(hlo_text: &str, pass_name: &str) -> String {
    let module = parse_hlo(hlo_text);
    let mut memory_manager = MemoryManager::new();
    
    // Pre-allocate memory for all tensors
    for instruction in &module.instructions {
        memory_manager.allocate_tensor(
            &instruction.name,
            &instruction.dtype,
            &instruction.shape,
            instruction.op == "parameter",
        );
    }

    let mut wat = vec![
        "(module".to_string(),
        format!("  ;; Generated from {} HLO", pass_name),
        "  (memory (export \"memory\") 2)".to_string(), // Increased memory
    ];

    // Add runtime helper functions
    wat.push(runtime_helpers::matmul_helper());
    wat.push(runtime_helpers::reduce_sum_helper());
    wat.push(runtime_helpers::elementwise_helpers());
    wat.push(runtime_helpers::math_helpers());
    
    // Add SIMD functions if needed
    wat.push(simd::simd_add_helper());
    wat.push(simd::simd_mul_helper());

    // Generate main computation function
    let params = memory_manager.get_parameters();
    let param_list: String = params
        .iter()
        .enumerate()
        .map(|(i, _)| format!(" (param $p{} f32)", i))
        .collect();

    wat.push(format!(
        "  (func $main (export \"main\"){} (result f32)",
        param_list
    ));

    // Declare local variables for intermediate results
    for instruction in &module.instructions {
        if instruction.op != "parameter" {
            wat.push(format!("    (local ${} f32)", instruction.name));
        }
    }

    // Initialize parameters in memory
    for (i, (_name, info)) in params.iter().enumerate() {
        wat.push(format!(
            "    (f32.store (i32.const {}) (local.get $p{}))",
            info.offset, i
        ));
    }

    // Generate computation for each instruction
    for instruction in &module.instructions {
        if instruction.op == "parameter" {
            continue; // Already handled above
        }
        
        let computation = translate_instruction(instruction, &memory_manager);
        wat.push(format!("    {}", computation));
    }

    // Return the final result
    if let Some(last_instruction) = module.instructions.last() {
        if last_instruction.op != "parameter" {
            wat.push(format!("    (local.get ${})", last_instruction.name));
        } else {
            wat.push("    (f32.const 0.0) ;; No computation found".to_string());
        }
    }

    wat.push("  )".to_string());
    wat.push(")".to_string());

    wat.join("\n")
}

fn translate_instruction(instruction: &HloInstruction, memory: &MemoryManager) -> String {
    match instruction.op.as_str() {
        "add" => translate_binary_op(instruction, memory, "f32.add"),
        "subtract" => translate_binary_op(instruction, memory, "f32.sub"),
        "multiply" => translate_binary_op(instruction, memory, "f32.mul"),
        "divide" => translate_binary_op(instruction, memory, "f32.div"),
        "maximum" => translate_binary_op(instruction, memory, "f32.max"),
        "minimum" => translate_binary_op(instruction, memory, "f32.min"),
        
        "exp" => translate_unary_op(instruction, memory, "exp"),
        "log" => translate_unary_op(instruction, memory, "log"),
        "tanh" => translate_unary_op(instruction, memory, "tanh"),
        "negate" => translate_unary_op(instruction, memory, "f32.neg"),
        
        "reduce" | "reduce-sum" => translate_reduce(instruction, memory),
        "dot" => translate_dot(instruction, memory),
        "broadcast" => translate_broadcast(instruction, memory),
        "reshape" => translate_reshape(instruction, memory),
        
        _ => format!(
            "(local.set ${} (f32.const 0.0)) ;; TODO: Implement {}",
            instruction.name, instruction.op
        ),
    }
}

fn translate_binary_op(instruction: &HloInstruction, memory: &MemoryManager, op: &str) -> String {
    if instruction.operands.len() != 2 {
        return format!(";; Error: {} requires 2 operands", op);
    }

    let lhs = &instruction.operands[0];
    let rhs = &instruction.operands[1];

    format!(
        "(local.set ${} ({} (local.get ${}) (local.get ${})))",
        instruction.name, op, lhs, rhs
    )
}

fn translate_unary_op(instruction: &HloInstruction, _memory: &MemoryManager, op: &str) -> String {
    if instruction.operands.len() != 1 {
        return format!(";; Error: {} requires 1 operand", op);
    }

    let operand = &instruction.operands[0];

    if op.starts_with("f32.") {
        format!(
            "(local.set ${} ({} (local.get ${})))",
            instruction.name, op, operand
        )
    } else {
        format!(
            "(local.set ${} (call ${} (local.get ${})))",
            instruction.name, op, operand
        )
    }
}

fn translate_reduce(instruction: &HloInstruction, memory: &MemoryManager) -> String {
    let operand = &instruction.operands[0];
    let operand_info = memory.get_tensor_info(operand).unwrap();
    let length = if operand_info.shape.is_empty() { 1 } else { operand_info.shape.iter().product() };

    format!(
        "(local.set ${} (call $reduce_sum (i32.const {}) (i32.const {})))",
        instruction.name, operand_info.offset, length
    )
}

fn translate_dot(instruction: &HloInstruction, memory: &MemoryManager) -> String {
    let lhs = &instruction.operands[0];
    let rhs = &instruction.operands[1];
    let lhs_offset = memory.get_offset(lhs).unwrap();
    let rhs_offset = memory.get_offset(rhs).unwrap();
    let out_offset = memory.get_offset(&instruction.name).unwrap();

    format!(
        "(call $matmul (i32.const {}) (i32.const {}) (i32.const {}))",
        lhs_offset, rhs_offset, out_offset
    )
}

fn translate_broadcast(instruction: &HloInstruction, _memory: &MemoryManager) -> String {
    // For scalar broadcast, just copy the value
    let operand = &instruction.operands[0];
    format!("(local.set ${} (local.get ${}))", instruction.name, operand)
}

fn translate_reshape(instruction: &HloInstruction, _memory: &MemoryManager) -> String {
    // For reshape, just reference the same memory location
    let operand = &instruction.operands[0];
    format!("(local.set ${} (local.get ${}))", instruction.name, operand)
}
