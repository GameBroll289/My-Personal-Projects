import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

# Create CarRacing-v3 environment
env = gym.make("CarRacing-v3", render_mode="human")
obs, info = env.reset()

# Hyperparameters
IMG_SIZE = (96, 96)  # Resize input images
GAMMA = 0.99         # Discount factor
LR = 0.0003          # Learning rate
EPSILON = 0.2        # Clipping range

# Preprocess function
def preprocess(obs):
    obs = cv2.resize(obs, IMG_SIZE)  # Resize
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    obs = obs / 255.0  # Normalize
    return np.expand_dims(obs, axis=-1)  # Add channel dimension

# Define PPO Model (Policy & Value networks)
def build_ppo_model():
    inputs = keras.Input(shape=(96, 96, 1))
    x = keras.layers.Conv2D(32, (8, 8), strides=4, activation="relu")(inputs)
    x = keras.layers.Conv2D(64, (4, 4), strides=2, activation="relu")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    
    action_out = keras.layers.Dense(3, activation="tanh")(x)  # Continuous action space (-1 to 1)
    value_out = keras.layers.Dense(1, activation=None)(x)  # Value function output
    
    model = keras.Model(inputs, [action_out, value_out])
    return model

ppo_model = build_ppo_model()
ppo_optimizer = keras.optimizers.Adam(learning_rate=LR)

# PPO Training Step (Online update per step)
def train_ppo(state, action, reward, next_state, done):
    state = np.expand_dims(state, axis=0)
    next_state = np.expand_dims(next_state, axis=0)
    
    with tf.GradientTape() as tape:
        new_action, value = ppo_model(state)
        _, next_value = ppo_model(next_state)
        
        advantage = reward + GAMMA * next_value * (1 - int(done)) - value
        action_loss = -tf.reduce_mean(tf.minimum(
            advantage * new_action,  # Policy update
            tf.clip_by_value(new_action, -EPSILON, EPSILON) * advantage
        ))
        value_loss = keras.losses.MSE(tf.stop_gradient(reward + GAMMA * next_value * (1 - int(done))), value)  # Critic update
        loss = action_loss + 0.5 * value_loss
    
    gradients = tape.gradient(loss, ppo_model.trainable_variables)
    ppo_optimizer.apply_gradients(zip(gradients, ppo_model.trainable_variables))

# Training Loop
EPISODES = 1000
for episode in range(EPISODES):
    state, info = env.reset()
    state = preprocess(state)
    done = False
    total_reward = 0
    fuel = info.get("fuel", 100)  # Ensure fuel is tracked correctly
    
    while True:
        env.render()
        action, _ = ppo_model(np.expand_dims(state, axis=0))
        action = action.numpy()[0]
        
        next_state, reward, done, truncated, info = env.step(action)
        next_state = preprocess(next_state)
        fuel = info.get("fuel", fuel)  # Update fuel value safely
        
        train_ppo(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        if done or truncated or fuel <= 0:
            break
    
    print(f"Episode {episode + 1}: Total Reward: {total_reward}, Remaining Fuel: {fuel}")
    
env.close()
