#ifndef B6CF27FD_C9E3_4DF6_AD5B_333BDEE827E8
#define B6CF27FD_C9E3_4DF6_AD5B_333BDEE827E8

#include <config.hpp>
#include <vector>
#include <optional>
#include <descent.hpp>
#include <linesearch.hpp>
#include <variant>
#include <stdexcept>
#include <memory>
#include <iostream>

namespace minimizer
{
    /**
     * @brief Abstract base class for gradient-based minimization algorithms.
     *
     * The AbstractMinimizer class defines the interface and common functionality
     * for minimization routines that operate by iteratively improving a candidate
     * solution to optimize a given function. It supports various norm types to
     * evaluate convergence based on the norm of vectors.
     *
     * @section Norms Enumeration of Norm Types
     * The class defines an enumeration, Norm, which specifies the norm used:
     * - L1  : L1 norm (sum of absolute values).
     * - L2  : L2 norm (Euclidean norm).
     * - FRO : Squared Frobenius norm.
     * - INF : Infinity norm (maximum absolute value).
     *
     * @section ResultStruct Minimization Result Structure
     * The nested struct Result encapsulates the output of the minimization process:
     * - x     : The result vector (solution).
     * - evals : Number of function evaluations performed.
     * - opt_f : The optimum value of the objective function.
     * - opt_g : The optimum value of the gradient norm.
     *
     * @section Constructors and Destructors
     * - The constructor initializes the minimizer with maximum iterations, tolerance levels,
     *   and the selected norm type.
     * - A virtual destructor ensures proper cleanup in derived classes.
     *
     * @section Setters Configuration Methods
     * The following setter methods allow configuration of the minimization process:
     * - set_line_search() transfers ownership of a specified line search method.
     * - set_descent() transfers ownership of a specified descent method.
     * - set_norm() sets which norm to use during computation.
     *
     * @section Solve Method
     * The pure virtual function solve() must be implemented by derived classes.
     * It is the core routine performing the minimization given a function and its gradient.
     *
     * @note
     * The protected member function norm(const Vec &x) computes the norm of a vector
     * according to the currently selected metric and throws an exception if an unknown norm
     * type is encountered.
     */
    class AbstractMinimizer
    {
    public:
        enum Norm
        {
            L1,
            L2,
            FRO,
            INF
        };

        struct Result
        {
            Vec x;
            size_t evals;
            double opt_f;
            double opt_g;
        };

        AbstractMinimizer(size_t n_iter, double g_tol, double f_tol, Norm norm) : n_iter_(n_iter),
                                                                                  grad_tol_(g_tol), func_tol_(f_tol), norm_(norm) {};
        virtual ~AbstractMinimizer() {};

        // Setters
        void set_line_search(std::unique_ptr<AbstractLineSearch> &&line_search_method) { line_search_method_ = std::move(line_search_method); } // we want to transfer ownership of the method to the minimizer
        void set_descent(std::unique_ptr<AbstractDescentMethod> &&descent_method) { descent_method_ = std::move(descent_method); }              // calls release after moving ownership
        void set_norm(Norm n) { norm_ = n; }

        virtual Result solve(const Func &f, const Grad &g, std::optional<Vec> initial_guess = std::nullopt) const = 0;

    protected:
        double norm(const Vec &x) const
        {
            switch (norm_)
            {
            case L1:
                return x.lpNorm<1>();
            case L2:
                return x.norm();
            case FRO:
                return x.squaredNorm();
            case INF:
                return x.lpNorm<Eigen::Infinity>();
            default:
                throw std::invalid_argument("Unknown norm type");
            }
        };

        size_t n_iter_ = 100;
        double func_tol_ = 1e-5;
        double grad_tol_ = 1e-5;

        std::unique_ptr<AbstractLineSearch> line_search_method_ = nullptr;
        std::unique_ptr<AbstractDescentMethod> descent_method_ = nullptr;

        Norm norm_ = Norm::L2;
    };

    /**
     * @class BFGSMinimizer
     * @brief Implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm for function minimization.
     *
     * BFGSMinimizer is a concrete implementation of the AbstractMinimizer interface that uses the
     * quasi-Newton BFGS update strategy to approximate the inverse Hessian matrix during iteration.
     * It integrates a line search method (Golden Section) and a descent direction method (Newton Descent)
     * to iteratively update the solution.
     *
     * The minimizer updates its approximation to the Hessian matrix using the BFGS formula:
     *   H_{k+1} = (I - ρ s y^T) H_k (I - ρ y s^T) + ρ s s^T,
     * where ρ = 1/(y^T s), s = x_{k+1} - x_k, and y = grad_{k+1} - grad_k.
     *
     * Convergence is assessed based on:
     * - The norm of the gradient falling below a specified tolerance (g_tol).
     * - The absolute change in the function value being less than a specified tolerance (f_tol).
     *
     * @param n_iter  Maximum number of iterations allowed (default is 100).
     * @param g_tol   Tolerance for the gradient norm to consider optimality (default is 1e-5).
     * @param f_tol   Tolerance for the change in function value for convergence (default is 1e-5).
     * @param norm    Norm function used to measure the gradient magnitude (default is Norm::L2).
     *
     * The class relies on dynamic polymorphism for the line search and descent methods, which are set
     * during construction.
     *
     * Example usage:
     * @code
     * BFGSMinimizer minimizer;
     * auto result = minimizer.solve(func, grad, initial_guess);
     * // result contains the final solution vector, number of evaluations, final function value, and final gradient norm.
     * @endcode
     */
    class BFGSMinimizer : public AbstractMinimizer
    {
    public:
        BFGSMinimizer(size_t n_iter = 100, double g_tol = 1e-5, double f_tol = 1e-5, Norm norm = Norm::L2) : AbstractMinimizer(n_iter, g_tol, f_tol, norm)
        {
            set_line_search(std::make_unique<GoldenSectionLineSearch>()); // actually, it would be better to use StrongWolfe conditions, but it goes beyond the scope of this example
            set_descent(std::make_unique<NewtonDescent>());
        };

        Result solve(const Func &func, const Grad &grad, std::optional<Vec> initial_guess = std::nullopt) const override
        {
            // 1) Initialization
            Vec x = initial_guess.value_or(Vec::Zero());
            size_t evals = 0;

            // initial f and g
            double func_val = func(x);
            Vec grad_vals = grad(x);

            Matrix hessian_ = Matrix::Identity();
            Matrix I = Matrix::Identity(x.size(), x.size());

            double opt_g = norm(grad_vals);

            for (size_t k = 0; k < n_iter_; ++k)
            {
                ++evals;

                Vec direction = descent_method_->direction(grad, x, hessian_);
                double alpha = line_search_method_->distance(func, grad, x, direction);

                Vec x_next = x + alpha * direction;
                double f_next = func(x_next);
                Vec g_next = grad(x_next);

                Vec s = x_next - x;
                Vec y = g_next - grad_vals;
                double rho = 1.0 / y.dot(s);
                hessian_ = (I - rho * s * y.transpose()) *
                               hessian_ *
                               (I - rho * y * s.transpose()) +
                           rho * (s * s.transpose());

                double f_prev = func_val; // save old value
                x = x_next;
                func_val = f_next;
                grad_vals = g_next;

                double grad_norm = norm(grad_vals);
                opt_g = grad_norm;

                if (grad_norm < grad_tol_)
                    break;
                if (std::abs(func_val - f_prev) < func_tol_)
                    break;
            }
            return {x, evals, func_val, opt_g};
        }
    };

    /**
     * @class GradientDescentMinimizer
     * @brief Implements gradient descent optimization with backtracking line search.
     *
     * This class performs function minimization using a gradient descent strategy combined
     * with backtracking line search to determine the step size. It extends AbstractMinimizer
     * and utilizes provided descent and line search strategy objects.
     *
     * @param n_iter Maximum number of iterations to run the algorithm (default 100).
     * @param grad_tol Tolerance for the gradient norm below which the algorithm stops (default 1e-6).
     * @param func_tol Tolerance for function value improvement (default 0.0).
     * @param norm Norm to measure the gradient (default Norm::L2).
     * @param alpha Initial step size factor for backtracking line search (default 1e-4).
     * @param beta Reduction factor for backtracking line search (default 0.5).
     *
     * @note The solve() method computes:
     *       - The current gradient norm and checks for convergence.
     *       - A descent direction based on the provided gradient.
     *       - A step size using the backtracking line search method.
     *       - Updates the current iterate.
     *
     * @see AbstractMinimizer
     *
     * @throws std::logic_error Thrown in solve() if the descent or line search strategies are not set.
     */
    class GradientDescentMinimizer : public AbstractMinimizer
    {
    public:
        GradientDescentMinimizer(std::size_t n_iter = 100,
                                 double grad_tol = 1e-6,
                                 double func_tol = 0.0,
                                 Norm norm = Norm::L2,
                                 double alpha = 1e-4,
                                 double beta = 0.5)
            : AbstractMinimizer(n_iter, grad_tol, func_tol, norm)
        {
            set_descent(std::make_unique<GradientDescent>());
            set_line_search(std::make_unique<BacktrackingLineSearch>(alpha, beta));
        }

        Result solve(const Func &f, const Grad &g, std::optional<Vec> x0 = std::nullopt) const override
        {
            if (!descent_method_ || !line_search_method_)
                throw std::logic_error("GradientDescentMinimizer: strategy pointers not set");

            Vec x = x0.value_or(Vec::Zero());
            Vec grad_x = g(x);
            double grad_norm = norm(grad_x);

            std::size_t evals = 1; // we already evaluated g
            double f_val = f(x);

            for (std::size_t k = 0; k < n_iter_; ++k)
            {
                ++evals;
                if (grad_norm <= grad_tol_)
                    break;

                Vec d = descent_method_->direction(g, x);
                double t = line_search_method_->distance(f, g, x, d);

                x += t * d;
                f_val = f(x);
                grad_x = g(x);
                grad_norm = norm(grad_x);
            }
            return {x, evals, f_val, grad_norm};
        }
    };
}
#endif /* B6CF27FD_C9E3_4DF6_AD5B_333BDEE827E8 */
