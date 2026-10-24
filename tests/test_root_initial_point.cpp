#include <gpuboscia/tree.h>
#include <limits>
#include <iostream>

struct BoxCheck {
    bool relaxation_feasible_cpu(const std::vector<double>&,
        const std::vector<double>&,const std::vector<double>&,double) const { return true; }
};
struct CustomSet {};
struct SimplexCheck {
    bool relaxation_feasible_cpu(const std::vector<double>& x,
        const std::vector<double>&,const std::vector<double>&,double tol) const {
        double sum=0;
        for(double v:x) {if(v < -tol) return false;sum+=v;}
        return sum<=1+tol;
    }
};
template<class F> void rejects(F f) {
    try {f();} catch(const std::invalid_argument&) {return;}
    throw std::runtime_error("invalid initial point was accepted");
}
int main() {
    Problem p{2,{0,0},{1,1},{0,1}};
    using gpuboscia::validate_root_initial_point;
    validate_root_initial_point(p,BoxCheck{},{});
    // Fractional values on integer variables are valid continuous starts.
    validate_root_initial_point(p,BoxCheck{},{.2,.7});
    rejects([&]{validate_root_initial_point(p,BoxCheck{},{.5});});
    rejects([&]{validate_root_initial_point(p,BoxCheck{},{-.1,.2});});
    rejects([&]{validate_root_initial_point(p,BoxCheck{},{std::numeric_limits<double>::quiet_NaN(),0});});
    rejects([&]{validate_root_initial_point(p,BoxCheck{},{0,std::numeric_limits<double>::infinity()});});
    rejects([&]{validate_root_initial_point(p,SimplexCheck{},{.8,.8});});
    validate_root_initial_point(p,SimplexCheck{},{.2,.7});
    rejects([&]{validate_root_initial_point(p,CustomSet{},{.2,.7});});
    bool called=false;
    auto check=[&](const auto& x,const auto&,const auto&,double) {called=true;return x[0]+x[1]<=1;};
    validate_root_initial_point(p,CustomSet{},{.2,.7},check);
    if(!called) throw std::runtime_error("custom validator was skipped");
    rejects([&]{validate_root_initial_point(p,CustomSet{},{.8,.8},check);});
    rejects([&]{validate_root_initial_point(p,BoxCheck{},{.2,.7},{},-1);});
    std::cout<<"Root initial-point validation passed\n";
}
