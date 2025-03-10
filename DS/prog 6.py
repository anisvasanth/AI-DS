import numpy as np
import matplotlib.pyplot as plt

def plot_normal_distribution(mean, std_dev):

  x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
  y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

  plt.plot(x, y)
  plt.xlabel('x')
  plt.ylabel('Probability Density')
  plt.title('Normal Distribution')
  plt.show()

# Example usage:
mean = 5
std_dev = 1
plot_normal_distribution(mean, std_dev)


--------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def simulate_exam_scores(mean_score, std_dev_score, num_students):
    """Generates simulated exam scores based on a normal distribution."""
    return np.random.normal(mean_score, std_dev_score, num_students)

def plot_exam_distribution(scores, mean_score, std_dev_score):
    """Plots a histogram of the exam scores and overlays a normal curve."""
    plt.hist(scores, bins=20, density=True, label="Exam Score Distribution")
    x = np.linspace(min(scores), max(scores), 100)
    plt.plot(x, stats.norm.pdf(x, mean_score, std_dev_score), label="Normal Curve")
    plt.xlabel("Score")
    plt.ylabel("Probability Density")
    plt.title("Exam Score Distribution")
    plt.legend()
    plt.show()

def main():
    """Main function to run the simulation and visualization."""
    mean_score = 70
    std_dev_score = 10
    num_students = 1000

    scores = simulate_exam_scores(mean_score, std_dev_score, num_students)
    plot_exam_distribution(scores, mean_score, std_dev_score)

if __name__ == "__main__":
    main()

--------------------------------------------------------------------------------------------------------