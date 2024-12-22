# Solar Heating System with Air Thermics - Physical Informed Neural Networks (PINN)

This project aims to predict the performance of a solar heating system using an air thermic principle. We built a model to predict the temperature and air speed at different points of the system under various sunlight and external temperature conditions. This model was implemented using a Physics-Informed Neural Network (PINN).

## Overview

The system is designed to operate as a solar heating solution, where solar radiation passes through a double-glazed window, is absorbed by an aluminum plate, and transfers heat to the cold air, which then rises through a duct and returns to the house at a higher temperature. Our goal is to model and predict the system's behavior to optimize its efficiency under different conditions.

### Key Steps:
1. **Simulation and Data Generation:**
   We used COMSOL to simulate an ideal system and generate data based on solving partial differential equations (PDEs) for heat transfer, fluid dynamics, and thermodynamics.

2. **Neural Network Model:**
   We initially used a Physically Informed Neural Network (PINN) based on the generated data. However, the initial model required about 200 training data points, which was impractical with our real-world measurements. Therefore, we adapted the approach to a simplified version that requires fewer data points for training.

3. **Inverse Problem Solution:**
   By incorporating a simplified analytical model and treating the unknown coefficients as parameters, we formulated an inverse problem using another PINN to estimate the system's coefficients (h1 and h2) with significantly fewer data points.

### Files:
1. **`direct_pinn.py`**: Contains the PINN for predicting the temperature and air speed under various conditions.
2. **`inverse_pinn.py`**: Solves the inverse problem by estimating the system's unknown coefficients (h1, h2) based on real-world measurements.
3. **`report.pdf`**: The original presentation and report for the solar heating project.

## Model Description

### 1. **PINN (Direct Model)**:
   - **Inputs**: Coordinates of a point in the system, solar power, and external temperature.
   - **Outputs**: Predicted temperature and air speed at that point.
   - **Training**: The network is trained using a combination of real-world data and PDEs based on physical laws like mass conservation, Navier-Stokes equations, and energy conservation.

### 2. **Inverse Problem (PINN)**:
   - **Objective**: To estimate unknown parameters (h1 and h2) by minimizing the error between the simplified analytical model and the real-world data.
   - **Training**: The PINN learns these coefficients with far fewer training points (around 10 measurements).

## Experimental Setup

- **Solar Power**: Measured using a sensor.
- **Air Temperature and Speed**: Measured using a hot-wire anemometer at 11 points along the system.
- **Simulation Setup**: COMSOL simulation to generate a synthetic dataset for comparison with real-world measurements.

## Results

- **Direct Model**: Using 200 measurements, the model predicted the temperature and speed with a good error margin.
- **Inverse Model**: The inverse problem approach significantly reduced the number of training data points to around 10, while achieving a good fit with real-world data.

### Performance:
- **Accuracy**: The PINN achieved a mean error of 0.4 K for temperature predictions, compared to 2.6 K for COMSOL.
- **Energy Gain**: The system achieved an energy gain of 572 kWh annually for a 0.5m² system, with a potential to heat a 10m² room.

## Conclusion

This project demonstrates that a simplified solar heating system can be modeled efficiently using a Physics-Informed Neural Network, providing a viable solution for energy savings in urban environments. The inverse modeling approach significantly reduces the number of required measurements, making it suitable for real-world applications.

## Requirements

To run this project, the following Python packages are required:

- `torch`
- `numpy`
- `matplotlib`
- `pyDOE`
