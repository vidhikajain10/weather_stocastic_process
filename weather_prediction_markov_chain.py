import random
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Data Generation
def generate_synthetic_data(days):
    weather_conditions = ["Sunny", "Rainy"]
    weather_data = [random.choice(weather_conditions) for _ in range(days)]
    weather_data_binary = [0 if weather == "Sunny" else 1 for weather in weather_data]
    return weather_data_binary

# Step 2: Construction of the TPM
def compute_tpm(weather_data):
    tpm = np.zeros((2, 2))
    for i in range(len(weather_data) - 1):
        current_state = weather_data[i]
        next_state = weather_data[i + 1]
        tpm[current_state][next_state] += 1
    tpm[0] = tpm[0] / tpm[0].sum()
    tpm[1] = tpm[1] / tpm[1].sum()
    return tpm

# Step 3: Prediction of Future Weather Probabilities
def predict_weather_probability(tpm, initial_state, days):
    weather_prob = np.linalg.matrix_power(tpm, days)
    return weather_prob[initial_state]

# Step 4: Conditional Probability Computation
def conditional_probability(tpm, current_state, future_days):
    future_prob_matrix = np.linalg.matrix_power(tpm, future_days)
    return future_prob_matrix[current_state]

# Visualization
def visualize_weather_probabilities(tpm, initial_state, days):
    probabilities = []
    for day in range(1, days + 1):
        prob = predict_weather_probability(tpm, initial_state, day)
        probabilities.append(prob)
    plt.plot(range(1, days + 1), [prob[0] for prob in probabilities], label="Sunny")
    plt.plot(range(1, days + 1), [prob[1] for prob in probabilities], label="Rainy")
    plt.xlabel("Days")
    plt.ylabel("Probability")
    plt.title("Weather Probabilities Over Time")
    plt.legend()
    plt.show()

# Main Script
if __name__ == "__main__":
    # Generate data
    weather_data_binary = generate_synthetic_data(10)
    print("Weather Data:", weather_data_binary)

    # Compute TPM
    tpm = compute_tpm(weather_data_binary)
    print("Transition Probability Matrix (TPM):\n", tpm)

    # Predict weather probabilities
    initial_state = 0  # Sunny
    days_ahead = 1
    probability = predict_weather_probability(tpm, initial_state, days_ahead)
    print(f"Probability of weather condition on day {days_ahead}:", probability)

    # Compute conditional probabilities
    current_state = 0  # Sunny
    future_days = 2
    conditional_prob = conditional_probability(tpm, current_state, future_days)
    print(f"Conditional probability on day {future_days} given current state {current_state}:", conditional_prob)

    # Visualize probabilities over time
    visualize_weather_probabilities(tpm, initial_state, 30)
