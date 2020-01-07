#include <iostream>
#include "xtensor/xnorm.hpp"
#include "hmm.hpp"
template <typename T>
constexpr auto type_name()
{
    std::string_view name, prefix, suffix;
#ifdef __clang__
    name = __PRETTY_FUNCTION__;
    prefix = "auto type_name() [T = ";
    suffix = "]";
#elif defined(__GNUC__)
    name = __PRETTY_FUNCTION__;
    prefix = "constexpr auto type_name() [with T = ";
    suffix = "]";
#elif defined(_MSC_VER)
    name = __FUNCSIG__;
    prefix = "auto __cdecl type_name<";
    suffix = ">(void)";
#endif
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    return name;
}
int main() {
    xt::print_options::set_precision(8);

    auto hmmptr = std::make_unique<HMM::DiscreteHiddenMarkovModel>(2, 6);
    std::cout << "HMM emission probs" << std::endl;
    std::cout << hmmptr->showEmissionProbs() << std::endl;
    std::cout << "HMM transition probs" << std::endl;
    std::cout << hmmptr->showTransitionProbs() << std::endl;

//    std::vector<int> bsaData{
//        3,1,5,1,1,6,2,4,6,4,4,6,6,4,4,2,4,5,3,1,1,3,2,1,6,3,1,1,6,4,
//        1,5,2,1,3,3,6,2,5,1,4,4,5,4,3,6,3,1,6,5,6,6,2,6,5,6,6,6,6,6,
//        6,5,1,1,6,6,4,5,3,1,3,2,5,4,1,2,4,5,6,3,6,6,6,4,6,3,1,6,3,6,
//        6,6,3,1,6,2,3,2,6,4,5,5,2,3,6,2,6,6,6,6,6,6,2,5,1,5,1,6,3,1,
//        2,2,2,5,5,5,4,4,1,6,6,6,5,6,6,5,6,3,5,6,4,3,2,4,3,6,4,1,3,1,
//        5,1,3,4,6,5,1,4,6,3,5,3,4,1,1,1,2,6,4,1,4,6,2,6,2,5,3,3,5,6,
//        3,6,6,1,6,3,6,6,6,4,6,6,2,3,2,5,3,4,4,1,3,6,6,1,6,6,1,1,6,3,
//        2,5,2,5,6,2,4,6,2,2,5,5,2,6,5,2,5,2,2,6,6,4,3,5,3,5,3,3,3,6,
//        2,3,3,1,2,1,6,2,5,3,6,4,4,1,4,4,3,2,3,3,5,1,6,3,2,4,3,6,3,3,
//        6,6,5,5,6,2,4,6,6,6,6,2,6,3,2,6,6,6,6,1,2,3,5,5,2,4,5,2,4,2
//    };

    std::vector<int> data{
            1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 2, 2, 1, 3, 2,
            3, 1, 3, 1, 2, 3, 2, 3, 1, 3, 1, 3, 3, 1, 3, 3, 3, 1, 1, 2, 1, 2, 3, 3, 3, 3, 1,
            3, 3, 3, 2, 3, 1, 2, 1, 1, 3, 2, 3, 2, 2, 2, 1, 3, 1, 1, 2, 2, 3, 1, 2, 3, 1, 2,
            1, 3, 2, 1, 1, 3, 1, 2, 1, 3, 2, 3, 2, 2, 3, 2, 3, 3, 3, 2, 3, 2, 3, 2, 2, 2, 3,
            3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 2, 1, 2, 1, 2, 1, 2, 3,
            1, 3, 3, 2, 1, 1, 2, 2, 3, 3, 1, 3, 1, 1, 1, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2,
            3, 2, 3, 2, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 2, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3,
            3, 1, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 1, 3, 1, 2, 3, 1, 2, 1, 2, 2, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 2, 1, 1, 2, 3, 2, 1, 3, 3, 2, 3, 3, 3, 2, 1, 2, 3, 3, 3, 2, 1, 2,
            1, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 1, 3, 1, 2, 2, 3, 1, 1, 3, 3, 3, 2, 2, 1, 1,
            2, 3, 2, 3, 2, 1, 3, 1, 3, 3, 1, 1, 1, 2, 1, 2, 2, 2, 3, 3, 1, 2, 3, 3, 3, 1, 2,
            2, 3, 3, 1, 2, 3, 3, 3, 3, 3, 3, 1, 2, 3, 3, 1, 3, 1, 3, 3, 3, 2, 3, 3, 3, 2, 2,
            2, 2, 3, 1, 1, 1, 3, 3, 2, 2, 3, 2, 1, 3, 2, 2, 2, 1, 2, 3, 2, 3, 2, 3, 3, 3, 1,
            3, 1, 1, 3, 3, 3, 3, 3, 3, 2, 1, 2, 2, 2, 3, 2, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3,
            3, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1, 3, 2, 1, 3, 3, 1, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3,
            3, 2, 3, 1, 3, 2, 3, 3, 3, 2, 3, 3, 3, 1, 1, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3,
            3, 1, 3, 3, 2, 3, 3, 3, 3, 2, 3, 1, 3, 2, 3, 3, 1, 2, 1, 2, 3, 2, 1, 3, 3, 3, 2,
            1, 2, 1, 3, 2, 3, 3, 3, 1, 3, 2, 3, 3, 1, 2, 3, 1, 1, 2, 1, 2, 2, 2, 3, 2, 1, 2,
            3, 2, 3, 3, 1, 1, 1, 3, 2, 2, 3, 3, 2, 3
    };

    for (auto &val : data) val--;

//    std::vector<int> data2{3,1,5,1,1,1,2,4,6,6,6,6,6,6,6,6,6,6,6,1,1,3,2,1,6,3,1,1,6,4};

//    xt::xarray<double> emission_probs = {
//            {1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6},
//            {1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 2}
//    };

    hmmptr = std::make_unique<HMM::DiscreteHiddenMarkovModel>(2, 3);
//    xt::xarray<double> emission_probs = {
//            {0.16, 0.26, 0.58},
//            {0.25, 0.28, 0.47}
//    };

    xt::xarray<double> emission_probs = {
            {1.0/9, 3.0/9, 5.0/9},
            {1.0/6, 1.0/3, 1.0/2}
    };
    hmmptr->setEmissionProbs(emission_probs);

    std::cout << "Updated HMM emission probs" << std::endl;
    std::cout << hmmptr->showEmissionProbs() << std::endl;

    xt::xarray<double> transition_probs = {
            {0.5, 0.5},
            {0.5, 0.5}
    };

//    xt::xarray<double> transition_probs = {
//            {0.54, 0.46},
//            {0.49, 0.51}
//    };

    hmmptr->setTransitionProbs(transition_probs);

    std::cout << "Updated HMM transition probs" << std::endl;
    std::cout << hmmptr->showTransitionProbs() << std::endl;

    // First 100 values from data
    std::vector<int> small(data.begin(), data.begin() + 100);


    hmmptr->setData(data);
    auto v = hmmptr->viterbi();

//    std::cout << "viterbi\n" << xt::adapt(v) << std::endl;

    auto fwd = hmmptr->forward();
//    std::cout << "fwd\n" << fwd << std::endl;

    auto bkwd = hmmptr->backward();
//    std::cout << "bkwd\n" << bkwd << std::endl;

    auto posterior = hmmptr->posterior();
//    std::cout << "posterior\n" << posterior << std::endl;

    auto baum_welch = hmmptr->baumWelch(500);
    return 0;
}
