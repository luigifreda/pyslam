#include <tbb/global_control.h>
#include <tbb/parallel_for_each.h>
#include <vector>

int main() {
    std::vector<int> v{1, 2, 3};
    tbb::global_control limit(tbb::global_control::max_allowed_parallelism, 2);
    tbb::parallel_for_each(v.begin(), v.end(), [](int &x) { x *= 2; });
    return v[0] == 2 ? 0 : 1;
}
