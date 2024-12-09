import numpy as np
import matplotlib.pyplot as plt

class Robot:
    def __init__(self, x, y, theta, max_speed, max_omega, dt):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = 0  # Linear velocity
        self.omega = 0  # Angular velocity
        self.max_speed = max_speed
        self.max_omega = max_omega
        self.dt = dt  # Time step

def simulate_motion(robot, v, omega, dt):
    """Simulates the robot's motion for one step."""
    x = robot.x + v * np.cos(robot.theta) * dt
    y = robot.y + v * np.sin(robot.theta) * dt
    theta = robot.theta + omega * dt
    return x, y, theta

def dynamic_window(robot):
    """Defines the dynamic window based on the robot's kinematic constraints."""
    dw = {
        "v_min": max(-robot.max_speed, robot.v - 0.5),
        "v_max": min(robot.max_speed, robot.v + 0.5),
        "omega_min": max(-robot.max_omega, robot.omega - 0.5),
        "omega_max": min(robot.max_omega, robot.omega + 0.5)
    }
    return dw

def evaluate_trajectory(robot, v, omega, goal, obstacles):
    """Evaluates a trajectory using the cost function."""
    x, y, theta = simulate_motion(robot, v, omega, robot.dt)
    
    # Heading cost: Distance to the goal
    heading_cost = np.hypot(goal[0] - x, goal[1] - y)
    
    # Clearance cost: Distance to nearest obstacle
    clearance_cost = float("inf")
    for obs in obstacles:
        dist = np.hypot(obs[0] - x, obs[1] - y)
        clearance_cost = min(clearance_cost, dist)
    
    # Velocity cost: Preference for higher velocities
    velocity_cost = -v
    
    # Weighted sum of costs
    cost = heading_cost + (1 / max(clearance_cost, 0.1)) + velocity_cost
    return cost

def dwa(robot, goal, obstacles):
    """Implements the Dynamic Window Approach."""
    best_v, best_omega = 0, 0
    best_cost = float("inf")
    
    dw = dynamic_window(robot)
    for v in np.arange(dw["v_min"], dw["v_max"], 0.1):
        for omega in np.arange(dw["omega_min"], dw["omega_max"], 0.1):
            cost = evaluate_trajectory(robot, v, omega, goal, obstacles)
            if cost < best_cost:
                best_cost = cost
                best_v, best_omega = v, omega
                
    return best_v, best_omega

def main():
    # Initialize robot and simulation parameters
    robot = Robot(x=0, y=0, theta=0, max_speed=1.0, max_omega=1.0, dt=0.1)
    goal = [10, 10]  # Goal position
    obstacles = [[5, 5], [6, 6], [7, 8]]  # List of obstacles
    
    # Simulate path planning
    path_x, path_y = [robot.x], [robot.y]
    for _ in range(100):
        v, omega = dwa(robot, goal, obstacles)
        robot.x, robot.y, robot.theta = simulate_motion(robot, v, omega, robot.dt)
        path_x.append(robot.x)
        path_y.append(robot.y)
        
        # Check if goal is reached
        if np.hypot(goal[0] - robot.x, goal[1] - robot.y) < 0.5:
            print("Goal reached!")
            break

    # Plot the results
    plt.figure()
    plt.plot(path_x, path_y, '-o', label="Path")
    plt.plot(goal[0], goal[1], 'rx', label="Goal")
    for obs in obstacles:
        plt.plot(obs[0], obs[1], 'ko', label="Obstacle" if obs == obstacles[0] else "")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
