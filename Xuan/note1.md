# **Topic1 - Gradient Descent**
## 1. What is gradient ? 
The direction of normal vector of the contour line of Loss function.

## 2. Adaptive Learning Rate
At the beginning, we are far from the destination, so we use larger learning rate. After several epochs, we are close to the destination, so we reduce the learning rate. E.g. $\eta^t=\dfrac{\eta}{\sqrt{t+1}}$

Learning rate can't be one-size-fits-all, different parameters could have different learning rates. 

## 3. Adagrad

$$ w^{t+1} \leftarrow w^t - \dfrac{\eta^t}{\sigma^t}g^t, $$
where
- $\eta^t=\dfrac{\eta}{\sqrt{t+1}}$
- $g^t=\dfrac{\partial L(\theta^t)}{\partial w}$
- $\sigma^t = \sqrt{\dfrac{1}{t+1} \displaystyle \sum_{i=0}^t (g^i)^2}$

Hence,

$$ w^{t+1} \leftarrow w^t - \dfrac{\eta}{\sqrt{\sum_{i=0}^t (g^i)^2}}g^t $$

## 4. Stochastic Gradient Descent
Loss for only one example (pick an example for $x^n$) :

$$ L^n = \left(y^n - (b + \sum w_ix_i^n)\right)^2 $$

Then,

$$ \theta^i = \theta^{i-1} - \eta \bigtriangledown L^n(\theta^{i-1}) $$

## 5. Each time we update the parameters, we <font color="darkred"> *don't* </font> obtain $\theta$ that makes $L(\theta)$ smaller.

Given one paramerter and condition on other parameters are constant, the direction that makes that parameter to minimize the loss. Hence, if we summarize all the directions, if might not be the minimum of the loss.

## 6. 
