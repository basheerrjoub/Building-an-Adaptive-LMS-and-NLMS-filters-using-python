# Building-an-Adaptive-LMS-and-NLMS-filters-using-python
build and compare between the two adaptive digital filters.
# Abstract
This article examines two adaptive filters algorithms, LMS and the normalized version NLMS, introducing the computations and implementation of these two algorithms that are mainly used for unknown system identification. Moreover, comparing the results obtained from the adaptation of these adaptive filters, deriving the coefficients of the filters and using these filters for noise cancellation purposes, and finally compare the results obtained from the two algorithms.
Keywords—NLMS, LMS, Adaptive Filter, discrete time, noise, convolution, spectrum.
# INTRODUCTION
Adaptive filters are one type of digital signal processing systems, that make use of the feedback system to generate an approximation for the unknown system, in ordinary cases these types of filters are used to specify the coefficients of an unknown system, also they can do the same job even with interfering noise to the input signal.
These types of filters are usually used in many applications such as noise reduction, system identification and equalization, there are many types of adaptive filters algorithms these types include the Least Mean Squares (LMS) algorithm and the Normalized Least Mean Squares (NLMS), these two algorithms are examined and derived in this paper.
# PROBLEM SPECIFICATION
Sometimes there are systems, are hard to obtain their transfer function in order to specify its behavior in the phase and amplitude response, or some systems that interfering with noise. So, using adaptive filters can be the solution for this problem in the real-life applications, as the adaptive filter tries to learn from the signal, as it feeds the system an input and used this input in its internal behavior in order to learn the response of the unknown system.
# DATA
Into mimicking the behavioral of the signals in the real-life applications, as known any signal can be formed as a Fourier series as a summation of sinusoidal, so in this paper using a simple sinusoidal as an input to the adaptive filter.

First of all, generating the input signal:
x[n]= cos(0.03πn),N=2000 samples. 

# EVALUATION CRITERIA
To evaluate the results obtained in this paper, the best evaluation techniques can include the error convergence, because the error convergence to zero specifies that the adaptive filter has learned well the signal behavioral, moreover other evaluation metric can be used is to compare and visualize the behavior of the signal in time domain and also in frequency domain to compare the phase and amplitude response of the original system with the adaptive filter


# METHODOLOGY OF LMS AND AN APPROACH TO SYSTEM IDENTIFICATION.
LMS is one of the most popular and widely used algorithms regarding adaptive filters, it is a gradient error minimization algorithm that is based on a single error value to adjust the coefficients until reaching a good point of accuracy on minimizing the error hence providing a good approximation for the original unknown system.
The algorithm for the LMS adaptive filter goes as in Algorithm 1,  first of all initializing the parameters for the LMS function, then repeating for the number of samples of the signal into minimizing gradually the amount of error, then using this error to learn the coefficients and adjusting the coefficients w into the new corresponding values, and so on, until deriving the approximating equation of the unknown systems response, in the case of DSP the approximating signal is a list of impulses for the specified number of samples.

# METHODOLOGY OF NLMS AND AN APPROACH TO SYSTEM IDENTICATION
Regarding LMS, the step size or what mean as the learning rate is always fixed for each iteration, this will make an overhead of understanding the behavior of the unknown system before the real implementation of the adaptive filter, which is hard to maintain. [2]
The solution for the fixed step size in LMS came as the Normalized LMS (NLMS), which avoids this problem by computing the step size in each iteration as in the following equation
Mu=1/(〖x[n]〗^T x[n])

    And then the weights are updated as in the following equation:
	w(n+1)=w(n)+u(n)e(n)x(n) 	(2)
    Each iteration requires N more multiplications than the standard LMS algorithm. However, the increase of complexity isn’t something to worry about when getting more stable response and better echo attenuation.

# RESULTS OF LMS ALGORITHM
First of all, after implementing the LMS algorithm which is included in appendix, the several figures are used to observe the behavioral of this Algorithm, as in the following figures the algorithm is ran with 2000 iterations and mu = 0.01 with a clear channel, meaning that there is no interfering noise (ideal case).

# COMPARING LMS AND NMLS

For the comparison of these two algorithms, first the initial learning rate should be fixed for the two algorithms so it will be fixed to 0.01, and no interfering noise to the signal will be assumed, so that no random occurring events to result in wrong assumptions.
   

# DEVELOPMENT

Into developing a more robust adaptive filter, reducing the learning rate can make the algorithm stronger and more accurate, but there is more overhead occurring because of increasing the time complexity. Moreover, using other adaptive filter algorithms can sometimes be better suited for some systems with specific events like specific type of noise can result in better accuracy like RLS, Kalman filter and APA.
   
# Conclusion

In practice, the choice between LMS and NLMS will depend on the specific requirements of the application. If stability is a concern, the NLMS algorithm may be the better choice, while if faster convergence is desired, the LMS algorithm may be more appropriate.
   NLMS is more used in real-life application due to its robust and accuracy, but they both have their applications to use, and depends on the compromise between the accuracy and complexity.
