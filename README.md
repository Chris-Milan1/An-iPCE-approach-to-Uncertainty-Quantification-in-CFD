# Reduced-Order Surrogate Modelling for Engineering Systems

## Project Overview
This project focuses on the development of a **Reduced-Order Surrogate Modelling (ROM)** framework designed to approximate computationally expensive engineering simulations. By using advanced statistical methods, the framework provides a low-cost alternative to high-fidelity simulations (such as CFD), enabling faster design iterations and robust uncertainty quantification without sacrificing significant accuracy.

## Key Features
* **Surrogate Framework:** Developed in Python to emulate complex system responses and improve the scalability of engineering workflows.
* **Uncertainty Quantification (UQ):** Implementation of methods to analyze how variations in model inputs propagate through a system.
* **Efficiency Analysis:** A deep dive into the trade-offs between model fidelity and computational cost to optimize simulation performance.
* **Validation:** Rigorous testing of surrogate predictions against high-fidelity numerical data to ensure predictive robustness.

## Methodologies Evaluated
The project investigates and compares two primary methods for uncertainty quantification:
1.   **Polynomial Chaos Expansion (PCE):** Utilized to create highly efficient emulators of system responses and quantify parameter influences.
2.   **Monte Carlo Methods:** Used as a baseline to evaluate the computational efficiency and accuracy of the PCE approach.

## Key Skills & Tech Stack
* **Languages:** Python
* **Core Concepts:** Reduced-Order PCE Modelling , Surrogate Modelling, Uncertainty Quantification (UQ)
* **Analysis:** Statistical Analysis, Computational Fluid Dynamics (CFD) validation, Parameter Sensitivity Analysis

## Conclusion
The results of this study demonstrate that **Polynomial Chaos Expansion (PCE)** significantly reduces computational overhead compared to traditional Monte Carlo methods while maintaining the accuracy required for high-stakes engineering applications. This makes it a viable tool for real-time simulation and optimization in complex engineering systems.
