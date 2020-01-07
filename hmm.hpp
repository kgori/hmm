//
// Created by Kevin Gori on 12/10/2019.
//

#ifndef HMM_HMM_HPP
#define HMM_HMM_HPP

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <sstream>
#include "xsimd/xsimd.hpp"
#include "xsimd/stl/algorithms.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor-blas/xlinalg.hpp"

namespace HMM {
using xmatrix = xt::xtensor<double, 2>;
struct OptimResult {
    xmatrix transition;
    xmatrix emission;
};

class DiscreteHiddenMarkovModel {
public:
    DiscreteHiddenMarkovModel() = delete;
    DiscreteHiddenMarkovModel(unsigned nstates, unsigned _noutputs);

    ~DiscreteHiddenMarkovModel();

    void setData(std::vector<int>& data);

    void setEmissionProbs(const xmatrix &e);

    std::string showEmissionProbs(int precision = 5);

    void setTransitionProbs(const xmatrix &t);

    std::string showTransitionProbs(int precision = 5);

    void setInitialProbs(const xt::xarray<double> &init);

    std::vector<int> viterbi();

    xmatrix forward();

    xmatrix backward();

    xmatrix posterior();

    double logprob(const xmatrix& fwd);

    OptimResult baumWelch(unsigned int niter);

    xmatrix& getEmissionMatrix() { return _emission_matrix; }
    const xmatrix& getEmissionMatrix() const { return _emission_matrix; }

    xmatrix& getTransitionMatrix() { return _transition_matrix; }
    const xmatrix& getTransitionMatrix() const { return _transition_matrix; }

private:
    xmatrix _emission_matrix;
    xmatrix _transition_matrix;
    xt::xarray<double> _initial_distribution;
    unsigned _nstates;
    unsigned _noutputs;
    xt::xarray<int> _data;
};

template <typename T>
double logsumexp(T xtvec) {
    auto m = xt::amax(xtvec)[{0}];
    return xt::log(xt::sum(xt::exp(xtvec - m)))[{0}] + m;
}

// Rescale a vector of log-probabilities to sum to 1, return in normal space (not log).
template<typename T>
xt::xarray<double> marginal_prob_from_logs(T logs) {
    double m = xt::amax(logs)[{0}];
    auto e = xt::exp(logs - m);
    return e / xt::sum(e);
}

}

#endif //HMM_HMM_HPP
