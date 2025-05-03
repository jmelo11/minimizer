#include <minimizer.hpp>
#include <iostream>

using namespace minimizer;

int main()
{
    // Example
    Func f = [](Vec x)
    {
        // Rosenbrock function in 2D: f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
        double t1 = 1.0 - x[0];
        double t2 = x[1] - x[0] * x[0];
        return t1 * t1 + 100 * t2 * t2;
    };
    Grad g = [](Vec x)
    {
        // Gradient of the Rosenbrock function
        double df_dx0 = -2 * (1.0 - x[0]) - 400 * x[0] * (x[1] - x[0] * x[0]);
        double df_dx1 = 200 * (x[1] - x[0] * x[0]);
        Vec grad(2);
        grad << df_dx0, df_dx1;
        return grad;
    };

    // BFGS
    BFGSMinimizer bfgs;
    auto result = bfgs.solve(f, g);
    std::cout << "Optimal x: " << result.x.transpose() << "\n";
    std::cout << "Function value: " << result.opt_f << "\n";
    std::cout << "Gradient norm: " << result.opt_g << "\n";
    std::cout << "Number of evaluations: " << result.evals << "\n";
    return 0;
}