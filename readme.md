# Minimizer

**Author**: Jose Melo

## Description

Example project of a simple minimizer inspired in the Python Library [[`optimistix`](https://docs.kidger.site/optimistix/)]. The idea behind the implementation is that optimizers in general can be divided in three main components:

- ***Direction***: The direction component is responsible for choosing the direction of the search. For example can be a gradient descent method or a newton inspired method (using some form of Hessian matrix).
- ***Search***: The search component is responsible for exploring the search space. It can be a line search method, trust region method, or any other method that explores the search space.
- ***Update (minimizer)***: The update component is responsible for updating the state of the optimizer. For example, in quasi-newton methods, the update step is responsible of updating the Hessian approximation.

## Code Stucture

`AbstractMinimizer`, `AbstractLineSearch` and `AbstractDescentMethod`: This classes provide interfaces for the minimizer, line search and descent method respectively. The idea is that implementations of `AbstractMinimizer` define the minimization algorithm (update step) and use predefined line search and descent methods.

## Further improvements

This is a basic implementation of a minimizer for educational purposes but it could be improved in many ways. Some ideas for further improvements are:

- Add automatic differentiation to the library, so that the user does not have to provide the gradient of the function.
- Implement more advance methods and add constraints to the minimizer.
