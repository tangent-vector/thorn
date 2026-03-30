// ============================================================================
// calc — a simple command-line calculator
// ============================================================================
//
// Purpose:
//   A command-line calculator that supports symbolic evaluation and
//   simplification of arithmetic expressions involving constants,
//   variables, and standard mathematical functions.
//
// Responsibilities:
//   - Read lines from stdin and evaluate each as an expression
//   - Expressions are simplified/evaluated and the result is printed
//   - Lines of the form `<variable> = <expression>` define variables;
//     subsequent expressions referring to defined variables substitute
//     the stored definition during simplification/evaluation
//   - Lines beginning with `#` are comments and are ignored
//   - Lines beginning with `quit` or `exit` terminate the program
//   - Support standard arithmetic operators: +, -, *, /
//   - Support standard functions: sin, cos, sqrt (at minimum)
//
// Requirements/Constraints:
//   - C++20 standard
//   - No third-party dependencies (standard library only)
//   - Clean separation of concerns across modules
//

#include "parser.h"
#include "evaluator.h"
#include "expression.h"

#include <cctype>
#include <exception>
#include <iostream>
#include <string>

int main()
{
    evaluator::Environment env;
    std::string line;
    
    while (std::getline(std::cin, line)) {
        // Trim leading whitespace
        std::size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) {
            continue;  // Empty line
        }
        line = line.substr(start);
        
        // Comment check
        if (!line.empty() && line[0] == '#') {
            continue;
        }
        
        // Quit/exit check
        if (line.starts_with("quit") || line.starts_with("exit")) {
            break;
        }
        
        // Check for variable assignment: var = expr
        std::size_t equals_pos = std::string::npos;
        int paren_depth = 0;
        for (std::size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '(') {
                paren_depth++;
            } else if (line[i] == ')') {
                paren_depth--;
            } else if (line[i] == '=' && paren_depth == 0) {
                equals_pos = i;
                break;
            }
        }
        
        if (equals_pos != std::string::npos && equals_pos > 0) {
            std::string var_part = line.substr(0, equals_pos);
            std::string expr_part = line.substr(equals_pos + 1);
            
            // Trim var_part
            std::size_t var_end = var_part.find_last_not_of(" \t");
            if (var_end != std::string::npos) {
                var_part = var_part.substr(0, var_end + 1);
            }
            
            // Check if valid identifier
            bool valid_var = !var_part.empty() && 
                (std::isalpha(static_cast<unsigned char>(var_part[0])) || var_part[0] == '_');
            for (std::size_t i = 1; valid_var && i < var_part.size(); ++i) {
                valid_var = std::isalnum(static_cast<unsigned char>(var_part[i])) || var_part[i] == '_';
            }
            
            if (valid_var) {
                try {
                    parser::ParseResult result = parser::parse(std::string_view(expr_part));
                    if (result.success()) {
                        evaluator::EvalResult eval_result = evaluator::evaluate(**result.expr, env);
                        std::cout << var_part << " = " << expression::to_string(*eval_result.expr) << std::endl;
                        env[var_part] = std::move(eval_result.expr);
                    } else {
                        std::cerr << "Error: " << result.error << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error: " << e.what() << std::endl;
                }
                continue;
            }
        }
        
        // Regular expression evaluation
        try {
            parser::ParseResult result = parser::parse(std::string_view(line));
            if (result.success()) {
                evaluator::EvalResult eval_result = evaluator::evaluate(**result.expr, env);
                if (eval_result.value.has_value()) {
                    std::cout << eval_result.value.value() << std::endl;
                } else {
                    std::cout << expression::to_string(*eval_result.expr) << std::endl;
                }
            } else {
                std::cerr << "Error: " << result.error << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    
    return 0;
}
