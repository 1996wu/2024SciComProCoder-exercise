#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace integral {
template <typename T>
auto load(const size_t sorb, const std::string fname) {
    std::vector<T> ovlp_int;
    std::ifstream istrm(fname);
    if (!istrm) {
        std::cout << "Failed to open " << fname << std::endl;
        exit(42);
    }
    std::string line;
    T eri;
    while (std::getline(istrm, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream is(line);
        if (is >> eri) { 
            ovlp_int.push_back(eri);
        }
    }
    return ovlp_int;
}
}  // namespace integral