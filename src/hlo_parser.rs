use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct HloInstruction {
    pub name: String,
    pub op: String,
    pub operands: Vec<String>,
    pub shape: Vec<usize>,
    pub dtype: String,
    #[allow(dead_code)]
    pub attributes: HashMap<String, String>,
}

#[derive(Debug)]
pub struct HloModule {
    #[allow(dead_code)]
    pub name: String,
    pub instructions: Vec<HloInstruction>,
    #[allow(dead_code)]
    pub entry_function: String,
    #[allow(dead_code)]
    pub parameters: Vec<HloInstruction>,
}

pub fn parse_hlo(hlo: &str) -> HloModule {
    let mut instructions = Vec::new();
    let mut parameters = Vec::new();
    let mut module_name = "unknown".to_string();
    let mut entry_function = "main".to_string();

    for line in hlo.lines() {
        let line = line.trim();
        
        // Parse module name
        if line.starts_with("HloModule") {
            module_name = line.split_whitespace()
                .nth(1)
                .unwrap_or("unknown")
                .to_string();
            continue;
        }

        // Parse entry function
        if line.starts_with("ENTRY") {
            entry_function = line.split_whitespace()
                .nth(1)
                .unwrap_or("main")
                .to_string();
            continue;
        }

        if line.is_empty() || line.starts_with("%") && line.contains("ROOT") {
            continue;
        }

        if let Some((lhs, rhs)) = line.split_once('=') {
            let name = lhs.trim().trim_start_matches('%').to_string();
            
            // Handle parameters specially
            if rhs.contains("parameter(") {
                let param = parse_parameter_instruction(&name, rhs);
                parameters.push(param.clone());
                instructions.push(param);
                continue;
            }

            // Parse regular instruction
            if let Some(instruction) = parse_instruction(&name, rhs) {
                instructions.push(instruction);
            }
        }
    }

    HloModule {
        name: module_name,
        instructions,
        entry_function,
        parameters,
    }
}

fn parse_parameter_instruction(name: &str, rhs: &str) -> HloInstruction {
    let (dtype, shape) = extract_dtype_and_shape(rhs);
    
    HloInstruction {
        name: name.to_string(),
        op: "parameter".to_string(),
        operands: vec![],
        shape,
        dtype,
        attributes: HashMap::new(),
    }
}

fn parse_instruction(name: &str, rhs: &str) -> Option<HloInstruction> {
    let parts: Vec<&str> = rhs.trim().split_whitespace().collect();
    if parts.len() < 2 { return None; }

    let (dtype, shape) = extract_dtype_and_shape(parts[0]);
    let op = parts[1].to_string();

    // Extract operands from parentheses
    let operand_start = rhs.find('(')?;
    let operand_end = rhs.find(')')?;
    let operand_str = &rhs[operand_start + 1..operand_end];
    
    let operands: Vec<String> = operand_str
        .split(',')
        .map(|s| {
            s.trim()
                .split_whitespace()
                .last()
                .unwrap_or("")
                .trim_start_matches('%')
                .to_string()
        })
        .filter(|s| !s.is_empty())
        .collect();

    // Parse attributes (dimensions, etc.)
    let mut attributes = HashMap::new();
    if let Some(dims_start) = rhs.find("dimensions={") {
        if let Some(dims_end) = rhs[dims_start..].find('}') {
            let dims_str = &rhs[dims_start + 12..dims_start + dims_end];
            attributes.insert("dimensions".to_string(), dims_str.to_string());
        }
    }

    Some(HloInstruction {
        name: name.to_string(),
        op,
        operands,
        shape,
        dtype,
        attributes,
    })
}

fn extract_dtype_and_shape(type_spec: &str) -> (String, Vec<usize>) {
    let dtype = type_spec
        .chars()
        .take_while(|c| c.is_alphabetic() || c.is_numeric())
        .collect::<String>();

    let shape = type_spec
        .split_once('[')
        .and_then(|(_, shape_str)| shape_str.strip_suffix(']'))
        .unwrap_or("")
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();

    (dtype, shape)
}
