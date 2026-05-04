import os
from flask import Flask, jsonify
import numpy as np
import random

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, static_folder=ROOT_DIR, static_url_path='/')

@app.route('/')
def index():
    return app.send_static_file('index.html')

class CliffWalkingEnv:
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = (3, 0)
        self.goal = (3, 11)

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c = self.state
        if action == 0:   # Up
            r = max(0, r - 1)
        elif action == 1: # Right
            c = min(self.cols - 1, c + 1)
        elif action == 2: # Down
            r = min(self.rows - 1, r + 1)
        elif action == 3: # Left
            c = max(0, c - 1)
        
        self.state = (r, c)
        
        # Check cliff
        if r == 3 and 1 <= c <= 10:
            return self.start, -100, False
        
        if self.state == self.goal:
            return self.state, -1, True
            
        return self.state, -1, False

def get_action(state, q_table, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)
    q_values = q_table[state[0], state[1]]
    max_q = np.max(q_values)
    actions = np.where(q_values == max_q)[0]
    return random.choice(actions)

def run_experiment(algo, runs=20, episodes=500, alpha=0.5, gamma=0.9, epsilon=0.1):
    all_rewards = np.zeros((runs, episodes))
    final_q_table = None
    
    for r in range(runs):
        env = CliffWalkingEnv()
        q_table = np.zeros((env.rows, env.cols, 4))
        
        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            if algo == 'sarsa':
                action = get_action(state, q_table, epsilon)
                
            while not done:
                if algo == 'q_learning':
                    action = get_action(state, q_table, epsilon)
                    next_state, reward, done = env.step(action)
                    best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
                    td_target = reward + gamma * q_table[next_state[0], next_state[1], best_next_action] * (not done)
                else: # sarsa
                    next_state, reward, done = env.step(action)
                    next_action = get_action(next_state, q_table, epsilon)
                    td_target = reward + gamma * q_table[next_state[0], next_state[1], next_action] * (not done)
                
                td_error = td_target - q_table[state[0], state[1], action]
                q_table[state[0], state[1], action] += alpha * td_error
                
                state = next_state
                if algo == 'sarsa':
                    action = next_action
                    
                total_reward += reward
                if total_reward < -2000: # safety break
                    break
                
            all_rewards[r, ep] = total_reward
            
        if r == runs - 1:
            final_q_table = q_table
            
    avg_rewards = np.mean(all_rewards, axis=0)
    
    policy = np.zeros((4, 12), dtype=int)
    for i in range(4):
        for j in range(12):
            policy[i, j] = int(np.argmax(final_q_table[i, j]))
            
    return avg_rewards.tolist(), policy.tolist()

@app.route('/api/train')
def train():
    q_rewards, q_policy = run_experiment('q_learning', runs=20, episodes=500, alpha=0.5, epsilon=0.1)
    s_rewards, s_policy = run_experiment('sarsa', runs=20, episodes=500, alpha=0.5, epsilon=0.1)
    
    return jsonify({
        'q_learning': {
            'rewards': q_rewards,
            'policy': q_policy
        },
        'sarsa': {
            'rewards': s_rewards,
            'policy': s_policy
        }
    })
