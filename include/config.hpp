#ifndef AB94CEBC_F21A_4379_B4BD_F36A242890B2
#define AB94CEBC_F21A_4379_B4BD_F36A242890B2

#include <Eigen/Dense>
#include <functional>

namespace minimizer
{
    using Vec = Eigen::Vector2d;
    using Matrix = Eigen::Matrix2d;
    using Func = std::function<double(Vec)>;
    using Grad = std::function<Vec(Vec)>;

}

#endif /* AB94CEBC_F21A_4379_B4BD_F36A242890B2 */
