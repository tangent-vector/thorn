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
// Dependencies:
//   - environment: for managing variable bindings
//   - expression: for expression representation and evaluation
//   - parser: for parsing input into expressions
//

#include "environment.h"
#include "expression.h"
#include "parser.h"

#include <iostream>
#include <string>
#include <unordered_map>

/// Trims leading whitespace from a string_view
static std::string_view trimLeft(std::string_view sv) {
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) {
        sv.remove_prefix(1);
    }
    return sv;
}

/// Checks if a line starts with a specific prefix (case-insensitive for commands)
static bool startsWithCommand(std::string_view line, std::string_view cmd) {
    if (line.size() < cmd.size()) {
        return false;
    }
    for (std::size_t i = 0; i < cmd.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(line[i])) != 
            std::tolower(static_cast<unsigned char>(cmd[i]))) {
            return false;
        }
    }
    // After the command, must be end of line or whitespace
    if (line.size() == cmd.size()) {
        return true;
    }
    return std::isspace(static_cast<unsigned char>(line[cmd.size()]));
}

int main()
{
    // Use a map of variable names to expressions for substitution
    std::unordered_map<std::string, expression::ExpressionPtr> variables;
    
    std::string line;
    
    while (std::getline(std::cin, line)) {
        // Trim leading whitespace for command detection
        std::string_view trimmed = trimLeft(line);
        
        // Skip empty lines
        if (trimmed.empty()) {
            continue;
        }
        
        // Skip comments
        if (trimmed.front() == '#') {
            continue;
        }
        
        // Handle quit/exit commands
        if (startsWithCommand(trimmed, "quit") || startsWithCommand(trimmed, "exit")) {
            break;
        }
        
        // Parse the input as a statement (could be expression or assignment)
        auto result = parser::parseStatement(line);
        
        if (!result.success) {
            std::cout << "Error: " << result.error->message << std::endl;
            continue;
        }
        
        if (result.isAssignment) {
            // Handle assignment: <variable> = <expression>
            const auto& assignment = result.getAssignment();
            
            // Substitute known variables in the expression
            auto substituted = expression::substitute(assignment.value, variables);
            
            // Simplify the expression
            auto simplified = expression::simplify(substituted);
            
            // Store in variables map
            variables[assignment.variableName] = simplified;
            
            // Print the result
            std::cout << assignment.variableName << " = " 
                      << expression::toPrettyString(simplified) << std::endl;
        } else {
            // Handle expression evaluation
            auto expr = result.getExpression();
            
            // Substitute known variables
            auto substituted = expression::substitute(expr, variables);
            
            // Simplify the expression
            auto simplified = expression::simplify(substituted);
            
            // Try to evaluate to a numeric result
            auto evalResult = expression::evaluate(simplified);
            
            if (evalResult.success) {
                std::cout << evalResult.value << std::endl;
            } else {
                // If evaluation failed (e.g., has unbound variables), print simplified form
                std::cout << expression::toPrettyString(simplified) << std::endl;
            }
        }
    }
    
    return 0;
}
