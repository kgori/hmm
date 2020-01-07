//
// Created by Kevin Gori on 13/12/2019.
//

#include "hmm.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xnorm.hpp"
#include <cmath>

namespace HMM {
double SCALE_THRESHOLD = 1e-20;
DiscreteHiddenMarkovModel::~DiscreteHiddenMarkovModel() {
    std::cout << "Destroying a DiscreteHiddenMarkovModel" << std::endl;
}

DiscreteHiddenMarkovModel::DiscreteHiddenMarkovModel(unsigned nstates, unsigned noutputs) : _nstates(nstates),
                                                                                            _noutputs(noutputs) {
    _emission_matrix = xt::ones<double>({nstates, noutputs});
    xt::xarray<double> rsums = xt::sum(_emission_matrix, {1});
    rsums.reshape({nstates, 1});
    _emission_matrix /= rsums;
    _transition_matrix = xt::ones<double>({nstates, nstates});
    rsums = xt::sum(_transition_matrix, {1});
    rsums.reshape({nstates, 1});
    _transition_matrix /= rsums;

    _initial_distribution = xt::ones<double>({nstates}) / nstates;
}

std::string DiscreteHiddenMarkovModel::showEmissionProbs(int precision) {
    std::stringstream sstr;
    sstr << _emission_matrix;
    return sstr.str();
}

std::string DiscreteHiddenMarkovModel::showTransitionProbs(int precision) {
    std::stringstream sstr;
    sstr << _transition_matrix;
    return sstr.str();
}

void DiscreteHiddenMarkovModel::setData(std::vector<int> &data) {
    _data = xt::adapt(data);
}

void DiscreteHiddenMarkovModel::setEmissionProbs(const xmatrix &e) {
    if (e.shape()[0] != _nstates || e.shape()[1] != _noutputs) {
        throw std::invalid_argument("Emission probability matrix doesn't have the required shape");
    }
    _emission_matrix = e;
}

void DiscreteHiddenMarkovModel::setTransitionProbs(const xmatrix &t) {
    if (t.shape()[0] != _nstates || t.shape()[1] != _nstates) {
        throw std::invalid_argument("Transition probability matrix doesn't have the required shape");
    }
    _transition_matrix = t;
}

void DiscreteHiddenMarkovModel::setInitialProbs(const xt::xarray<double> & init) {
    if (init.shape()[0] != _nstates) {
        throw std::invalid_argument("Initial probabilities vector doesn't have the required shape");
    }
    _initial_distribution = init;
}

std::vector<int> DiscreteHiddenMarkovModel::viterbi() {
    if (_data.size() == 0) return std::vector<int>();

    unsigned K = _nstates;
    unsigned T = _data.size();
    auto A = xt::log(_transition_matrix);
    auto B = xt::log(_emission_matrix);
    auto y = _data; //xt::adapt(_data) - 1;

    xt::xtensor<double, 2> T1 = xt::zeros<double>({K, T});
    xt::xtensor<int, 2> T2 = xt::zeros<int>({K, T});

    for (int i = 0; i < K; ++i) {
        T1(i, 0) = (1.0 / K) + B(i, y[0]);
        T2(i, 0) = 0;
    }

    for (int j = 1; j < T; ++j) {
        for (int i = 0; i < K; ++i) {
            T1(i, j) = xt::amax(xt::view(T1, xt::all(), j - 1) + xt::view(A, xt::all(), i) + B(i, y[j]))(0);
            T2(i, j) = xt::argmax(xt::view(T1, xt::all(), j - 1) + xt::view(A, xt::all(), i), 0)(0);
        }
    }

    xt::xtensor<int, 1> z = xt::zeros<int>({T});

    z(T - 1) = xt::argmax(xt::view(T1, xt::all(), T - 1))(0);

    for (int j = T - 1; j > 0; --j) {
        z(j - 1) = T2(z[j], j);
    }

    std::vector<int> x(z.begin(), z.end());
    return x;
}

xmatrix DiscreteHiddenMarkovModel::forward() {
    unsigned K = _nstates;
    unsigned T = _data.size();
    auto A = getTransitionMatrix();
    auto B = getEmissionMatrix();
    auto V = _data;

    xt::xtensor<double, 2> alpha = xt::zeros<double>({T, K});

    // multiply by initial distribution
    xt::view(alpha, 0, xt::all()) = _initial_distribution * xt::view(B, xt::all(), V[0]);

    xt::xarray<double> scaler = xt::zeros<double>({T});
    for (int t = 1; t < T; ++t) {
        xt::view(alpha, t, xt::all()) =
                xt::linalg::dot(xt::view(alpha, t - 1, xt::all()), A) * xt::view(B, xt::all(), V[t]);

        // Scaling to avoid underflow
        double m = xt::amax(xt::view(alpha, t, xt::all()))[{0}];
        if (m < SCALE_THRESHOLD) {
            xt::view(alpha, t, xt::all()) /= m;
            scaler[t] = log(m);
        }
        scaler[t] += scaler[t - 1];
    }
    scaler.reshape({-1, 1});
    return xt::log(alpha) + scaler;
}

xmatrix DiscreteHiddenMarkovModel::backward() {
    unsigned K = _nstates;
    unsigned T = _data.size();
    auto A = getTransitionMatrix();
    auto B = getEmissionMatrix();
    auto V = _data;

    xt::xtensor<double, 2> beta = xt::zeros<double>({T, K});
    xt::view(beta, T-1, xt::all()) = 1;

    xt::xarray<double> scaler = xt::zeros<double>({T});
    for (int t = T-2; t >= 0; --t) {
        auto tmp = xt::view(beta, t+1, xt::all()) * xt::view(B, xt::all(), V[t+1]);
        xt::view(beta, t, xt::all()) =
                xt::linalg::dot(A, tmp);

        // Scaling to avoid underflow
        double m = xt::amax(xt::view(beta, t, xt::all()))[{0}];
        if (m < SCALE_THRESHOLD) {
            xt::view(beta, t, xt::all()) /= m;
            scaler[t] = log(m);
        }
        scaler[t] += scaler[t+1];
    }
    scaler.reshape({-1, 1});
    return xt::log(beta) + scaler;
}

xmatrix DiscreteHiddenMarkovModel::posterior() {
    auto fwd = forward();
    auto bkwd = backward();
    auto view = xt::view(fwd, _data.size()-1, xt::all());
    auto logp = logsumexp(view);
    return xt::exp(fwd + bkwd - logp);
}

double DiscreteHiddenMarkovModel::logprob(const xmatrix& fwd) {
    auto view = xt::view(fwd, _data.size()-1, xt::all());
    return logsumexp(view);
}

OptimResult DiscreteHiddenMarkovModel::baumWelch(unsigned int niter) {
    double TOLERANCE = 1e-3;
    unsigned K = _nstates;
    unsigned T = _data.size();
    auto V = _data;
    unsigned M = _nstates;

    xt::xtensor<double, 3> xi = xt::zeros<double>({T-1, M, M});
    xt::xtensor<double, 2> gamma = xt::zeros<double>({T, M});
    double prev_lp = -100000;
    for (unsigned int i = 0; i < niter; ++i) {
        auto A = getTransitionMatrix();
        auto B = getEmissionMatrix();
        xmatrix alpha = forward();
        xmatrix beta = backward();
        // std::cout << "i=" << i << "; forward=\n" << alpha << "\nbackward=\n" << beta << std::endl;
        // double lp = logprob(alpha);
        // std::cout << "lp and prev_lp=" << lp << ", " << prev_lp << std::endl;
//        if (lp < prev_lp) {
//            std::cerr << "prob got worse after " << i << " iterations" << std::endl;
//            break;
//        }
//        if (lp - prev_lp < TOLERANCE) {
//            std::cerr << "within tolerance after " << i << " iterations" << std::endl;
//            break;
//        }
        // prev_lp = lp;

        for (unsigned int t = 0; t < T-1; ++t) {
            // Compute xi[t]
            auto alpha_view = xt::view(alpha, t, xt::all(), xt::newaxis());
            auto beta_view = xt::view(beta, t + 1, xt::all());
            auto B_view = xt::view(B, xt::all(), V[t + 1]);

            // std::cout << "[" << t << "]: beta_view=\n" << xt::exp(beta_view) << std::endl;
            // std::cout << "A=\n" << A << std::endl;
            // std::cout << "B_view=\n" << B_view << std::endl;
            // std::cout << "alpha_view=\n" << xt::exp(alpha_view) << std::endl;

            auto numerator = xt::exp(beta_view - xt::amax(beta_view)) * A * B_view * xt::exp(alpha_view - xt::amax(alpha_view));
            //auto numerator = xt::exp(beta_view) * A * B_view * xt::exp(alpha_view);
            // std::cout << "[" << i << "]numerator[" << t << "]=\n" << numerator << std::endl;
            xt::view(xi, t, xt::all(), xt::all()) = numerator / xt::sum(numerator);
            // std::cout << "[" << i << "]xi[" << t << "]=\n" << xt::view(xi, t, xt::all(), xt::all()) << std::endl;
        }
        // All but the last values of gamma
        xt::view(gamma, xt::range(0, T-1), xt::all()) = xt::sum(xi, {2});

        // Final value of gamma
        auto alpha_view = xt::view(alpha, T-1, xt::all());
        auto beta_view = xt::view(beta, T-1, xt::all());
        auto gamma_view = xt::view(gamma, T-1, xt::all());
        gamma_view = marginal_prob_from_logs(alpha_view + beta_view);
        // std::cout << "gamma=\n" << gamma << std::endl;

        // Update parameters
//        setInitialProbs(xt::view(gamma, 0, xt::all()));

        xt::xarray<double> gammasum = xt::sum(xt::view(gamma, xt::range(0, T-1), xt::all()), {0});

        // std::cout << "numerator=\n" << xt::sum(xi, {0}) << std::endl;
        // std::cout << "denominator=\n" << gammasum << std::endl;
        // std::cout << "probs=\n" << xt::sum(xi, {0}) / xt::reshape_view(gammasum, {(int)M, 1}) << std::endl;
        xmatrix newA = xt::sum(xi, {0}) / xt::reshape_view(gammasum, {(int)M, 1});
        // auto a_norm = xt::norm_l2(A - newA);
        setTransitionProbs(xt::sum(xi, {0}) / xt::reshape_view(gammasum, {(int)M, 1}));


        gammasum += xt::view(gamma, T-1, xt::all());

        auto newB = xt::zeros_like(B);
        for (int l = 0; l < _noutputs; ++l) {
            for (int t = 0; t < T; ++t) {
                if (V[t] == l) {
                    xt::view(newB, xt::all(), l) += xt::view(gamma, t, xt::all());
                }
            }
            xt::view(newB, xt::all(), l) /= gammasum;
        }
        // auto b_norm = xt::norm_l2(B - newB);
        setEmissionProbs(newB);

        // std::cout << "Parameter change=\n" << a_norm + b_norm << std::endl;

        // std::cout << "_initial_distribution=\n" << _initial_distribution << std::endl;

        // std::cout << "xtsum=\n" << xt::sum(xi, {0}) << std::endl;
        // std::cout << "gammasum=\n" << xt::sum(xt::view(gamma, xt::range(0, T-2), xt::all()), {0}) << std::endl;
        // std::cout << "[" << i << "]\na*=\n" << _transition_matrix << std::endl;
        // std::cout << "b*=\n" << _emission_matrix << std::endl;
    }
     std::cout << "a*=\n" << _transition_matrix << std::endl;
     std::cout << "b*=\n" << _emission_matrix << std::endl;
    return OptimResult{getTransitionMatrix(), getEmissionMatrix()};
}

}

