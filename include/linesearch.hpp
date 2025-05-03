#ifndef C3AFCF23_3C7B_452A_ABD8_D594217381C5
#define C3AFCF23_3C7B_452A_ABD8_D594217381C5

#include <config.hpp>
#include <variant>

namespace minimizer
{
    struct AbstractLineSearch
    {
        virtual ~AbstractLineSearch() {};
        virtual double distance(const Func &func, const Grad &grad, const Vec &x, const Vec &d) const = 0;
    };

    class BacktrackingLineSearch : public AbstractLineSearch
    {
    public:
        explicit BacktrackingLineSearch(double alpha = 0.5, double beta = 0.5)
            : alpha(alpha), beta(beta) {}

        double distance(const Func &func, const Grad &grad, const Vec &x, const Vec &d) const override
        {
            double t = 1.0;
            double f0 = func(x);
            Vec g = grad(x);
            double g_dot_d = g.dot(d);

            // Backtracking until Armijo (sufficient decrease) holds:
            while (func(x + t * d) > f0 + alpha * t * g_dot_d)
            {
                t *= beta;
            }
            return t;
        };

    private:
        double alpha; // Parameter for sufficient decrease condition
        double beta;  // Reduction factor for the step length
    };

    class GoldenSectionLineSearch : public AbstractLineSearch
    {
    public:
        explicit GoldenSectionLineSearch(double tol = 1e-6, int max_iters = 100, double a0 = 0.0, double b0 = 1.0)
            : tol_(tol), max_iters_(max_iters), a0_(a0), b0_(b0) {};

        double distance(const Func &func, const Grad &grad, const Vec &x, const Vec &d) const override
        {
            double a = a0_;
            double b = b0_;
            const double gr = (std::sqrt(5.0) - 1.0) / 2.0; // golden ratio factor

            // interior points
            double c = b - gr * (b - a);
            double e = a + gr * (b - a);
            double f_c = func(x + c * d);
            double f_e = func(x + e * d);

            int iter = 0;
            while ((b - a) > tol_ && iter < max_iters_)
            {
                if (f_c > f_e)
                {
                    // minimum in [c, b]
                    a = c;
                    c = e;
                    f_c = f_e;
                    e = a + gr * (b - a);
                    f_e = func(x + e * d);
                }
                else
                {
                    // minimum in [a, e]
                    b = e;
                    e = c;
                    f_e = f_c;
                    c = b - gr * (b - a);
                    f_c = func(x + c * d);
                }
                ++iter;
            }
            return 0.5 * (a + b);
        };

    private:
        double a0_, b0_, tol_;
        int max_iters_;
    };

    // Alternatively, we can use a variant and the visitor patter (std::visit) to handle different signatures
    // using LineSearchMethod = std::variant<BacktrackingLineSearch, GoldenSectionLineSearch>;
}

#endif /* C3AFCF23_3C7B_452A_ABD8_D594217381C5 */
