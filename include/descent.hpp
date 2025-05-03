#ifndef E67B00F9_133E_4920_8DAA_8FAF0AE412C7
#define E67B00F9_133E_4920_8DAA_8FAF0AE412C7

#include <config.hpp>
#include <variant>

namespace minimizer
{

    struct AbstractDescentMethod
    {
        virtual ~AbstractDescentMethod() {};
        virtual Vec direction(const Grad &g, const Vec &x, const std::optional<Matrix> &inv_hess = std::nullopt) const = 0;
    };

    class NewtonDescent : public AbstractDescentMethod
    {
    public:
        explicit NewtonDescent() {};

        Vec direction(const Grad &g, const Vec &x, const std::optional<Matrix> &inv_hess) const override
        {
            if (!inv_hess)
            {
                throw std::logic_error("Netwon method requires the inverse hessian.");
            };
            return -(inv_hess.value() * g(x));
        };
    };

    class GradientDescent : public AbstractDescentMethod
    {
    public:
        explicit GradientDescent() {};

        Vec direction(const Grad &g, const Vec &x, const std::optional<Matrix> &inv_H = std::nullopt) const override
        {
            return -g(x);
        };
    };

    // Same as before, we could use a variant
    // using DescentMethod = std::variant<GradientDescent, NewtonDescent>;
}

#endif /* E67B00F9_133E_4920_8DAA_8FAF0AE412C7 */
