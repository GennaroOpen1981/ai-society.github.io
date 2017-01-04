---
layout: post
title: Hello, Gradient Descent.
---

Hi there, this is the first article of a series called ***"Hello, \<algorithm\>."*** in which we will give some insights about how different Machine Learning and Artificial Intelligente algorithms work. Today we are going to talk about Gradient Descent, a simple but yet powerful optimization algorithm for finding the local minimum of a function. It is worth to mention that if the function is convex, the value encountered by Gradient Descent will be the global optimum.

## The insight

The idea behind Gradient Descent is inspired by Calculus. Basically, it says that if we have a differentiable function, the fastest way to decrease is by taking steps proportional to negative the gradient of that function. This is because the gradient of a function points to the steepest direction on the surface upwards. With this in mind we can repeatedly perform this procedure, and continue taking steps in the right direction until we converge into our local minimum. In the following image you can look an example about how Gradient Descent performs at each iteration starting from x0 down to x4.

# INSERT GRADIENT DESCENT EXAMPLE HERE

As we all remember, we can calculate the gradient of a function using partial derivatives. That way we will know based on the point we are, how much each variable or weight should change in order to reach the optimum. But we need to be very careful at each step, because the steepest direction may be too steepest and we could end up "passing over" our local optimum. This is a serious "bug" that could slow down our algorithm or make it loop forever as in the examples below.

# INSERT BUGGY EXAMPLES HERE

The approach taken to solve this issue is to perform steps scaled down by a learning rate. The learning rate is a number in the range (0, 1) (exclusive at both sides) that helps out algorithm to perform better by avoiding convergence problems. It's worth to mention that this parameter is an important decision in the Gradient Descent world. This is because if the learning rate is too small the algorithm will take too much time too converge, and if the rate is too big the "bugs" that we talked about earlier will persist.

## Some formulas

We hope that up to this point you have some intuition about the Gradient Descent algorithm. Now it's time to formalize a little bit of what we've talked about earlier. Suppose we have a two-variable function like this one:

###### INSERT IMAGE OF FORMULA y = theta1^2 + theta2^2

If we look closely, the local and global optimum of this function occurs at point(0, 0). The next step is to calculate the gradient of that function. This can be done by using partial derivatives with respect to each variable theta:

###### INSERT IMAGE OF GRADIENT \<2*theta1, 2*theta2\>

The brackets around the expression are used to indicate that this is a vector. Now we can simulate some of the algorithm's steps. Let us choose the starting point at (1, 3), and a learning rate called alpha = 0.1:

###### INSERT IMAGE OF STEPS

As you can see, each step approximates us more and more to the optimum (0, 0). If we look at a plot of this example it's going to look like this:

# INSERT IMAGE OF PLOT

# Hacking time

Okay, so we have gone through insight and formulas. In the last part of this article, we will train a digit recognition script using Logistic Regression and the Gradient Descent algorithm.
