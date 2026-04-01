# 🏎️ Racetrack Reinforcement Learning (Monte Carlo Control)

This project implements **on-policy Monte Carlo control with ε-soft policies** to solve the classic **racetrack problem** (Sutton & Barto, Exercise 5.12).

The goal is to learn a policy that drives a car from a start line to a finish line **as fast as possible**, while respecting velocity dynamics and avoiding crashes.

---

# 📌 Problem Overview

The environment models a car moving on a 2D racetrack with:

- **State**: `(x, y, v_x, v_y)`
- **Actions**: acceleration in both axes  
  `a ∈ {-1, 0, +1} × {-1, 0, +1}`
- **Dynamics**:
  - Velocity updates with acceleration
  - Position updates with velocity
- **Stochasticity**:
  - With probability **0.1**, acceleration is ignored (slip)
- **Rewards**:
  - `-1` per step (encourages shortest path)
- **Termination**:
  - Episode ends when the finish line is crossed
- **Crash handling**:
  - Going off-track resets the car to a start position

---

# 🧠 Algorithm

We use **on-policy Monte Carlo control (ε-soft)**.

### Key components:
- **Policy**: ε-soft (ensures exploration)
- **Value function**: `Q(s,a)`
- **Returns**: full episode returns `G_t`
- **Update rule**: `Q(s,a) ← Q(s,a) + α (G_t - Q(s,a))`

---

# 🔁 Learning Loop

1. Generate an episode using current policy  
2. Compute returns for each timestep  
3. Update Q-values (every-visit MC)  
4. Improve policy (ε-greedy)  
5. Repeat  

---

# ⚠️ Practical Modifications

Compared to textbook MC:

- ✅ **Incremental update (α)** instead of averaging → handles non-stationarity  
- ✅ **Every-visit MC** instead of first-visit → faster learning  
- ✅ **Optimistic initialization (Q = 1)** → encourages exploration  

---

# 📊 Visualizations

## 1. Value Function

- Computed as: `V(s) = max_a Q(s,a)`
- Averaged over velocity dimensions
- Shows how “good” each position is

**Interpretation:**
- Bright → closer to finish / faster paths  
- Dark → risky or inefficient areas  

---

## 2. Policy (Action Arrows)

- Arrows represent **average acceleration**
- Averaged over velocities

⚠️ **Important:**
- Arrows show **acceleration**, not movement  
- Actual motion depends on velocity  

---

# 🧩 Key Insights

### 🚗 Learned behavior:
- Accelerate early to build speed  
- Follow diagonal path along track  
- Avoid boundaries  
- Adjust near finish due to noise  

---

### ⚠️ Non-stationarity:
- Policy changes over time  
- Early episodes become irrelevant  
- Fixed step-size `α` helps “forget” bad experience  

---

### 📉 Plateau behavior:
- Learning stabilizes but does not fully converge  
- Due to:
- constant ε (ongoing exploration)
- stochastic dynamics  

---

# 🛠️ Project Structure
.
├── racetrack/
│   ├── *.csv              # track layouts
├── mc.py                  # Monte Carlo control algorithm
├── racetrack.py           # Racetrack environment and visualization functions
├── Magsigay_RL_Project    # Main notebook
├── media/                 # saved plots
└── README.md

---

Install independencies

```bash
pip install gym numpy matplotlib scikit-image
```

# 📈 Output

The code produces:

- Value function heatmaps  
- Policy visualizations (arrows)  
- Learning curve (`returns_log`)  

All outputs are saved in:

```bash
media\
```

---

# 📚 References

- Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 5  

---

# 🎯 Summary

> This project demonstrates how Monte Carlo control can learn a momentum-aware, risk-sensitive driving policy in a stochastic racetrack environment.
