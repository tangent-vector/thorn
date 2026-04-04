#include "repl.h"

#include <iostream>

int main() {
    repl::run(std::cin, std::cout);
    return 0;
}
