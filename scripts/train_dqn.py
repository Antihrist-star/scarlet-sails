"""
DQN TRAINING SCRIPT
Trains DQN agent on trading environment

ИСПРАВЛЕНО:
- Gradient clipping для предотвращения NaN
- Улучшенная reward normalization
- Warm-up period перед обучением
- Better logging и мониторинг

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 17, 2025
"""

import numpy as np
import pandas as pd
import sys
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.trading_environment import TradingEnvironment
from rl.dqn import DQNAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_training_data(n_bars=2000, seed=42):
    """Generate training data with multiple regimes"""
    np.random.seed(seed)
    
    logger.info(f"Generating {n_bars} bars of training data...")
    
    # Multiple regimes for robustness
    regime_lengths = [500, 700, 500, 300]
    regime_trends = [0.0002, -0.0001, 0.0003, -0.0002]  # Bull, Sideways, Bull, Bear
    regime_vols = [0.015, 0.020, 0.025, 0.040]
    
    close_prices = [50000]
    
    for regime_len, trend, vol in zip(regime_lengths, regime_trends, regime_vols):
        for _ in range(regime_len):
            ret = np.random.normal(trend, vol)
            new_price = close_prices[-1] * (1 + ret)
            close_prices.append(new_price)
    
    close_prices = np.array(close_prices[:n_bars])
    close_prices = np.maximum(close_prices, 1000)
    
    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.0005, n_bars)),
        'high': close_prices * (1 + np.abs(np.random.normal(0.002, 0.003, n_bars))),
        'low': close_prices * (1 - np.abs(np.random.normal(0.002, 0.003, n_bars))),
        'close': close_prices,
        'volume': np.random.lognormal(5, 0.5, n_bars)
    })
    
    logger.info(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    logger.info(f"  Date range: {len(df)} bars")
    
    return df


def train_dqn(num_episodes=100, save_freq=10, warmup_episodes=10):
    """
    Train DQN agent
    
    ИСПРАВЛЕНО: Добавлен warmup period и gradient monitoring
    
    Parameters:
    -----------
    num_episodes : int
        Number of training episodes
    save_freq : int
        Save model every N episodes
    warmup_episodes : int
        Number of episodes to fill replay buffer before training
    """
    print("\n" + "="*80)
    print("DQN TRAINING - SCARLET SAILS")
    print("="*80 + "\n")
    
    # Generate data
    print("STEP 1: DATA GENERATION")
    print("-"*80)
    df = generate_training_data(n_bars=2000)
    print()
    
    # Create environment
    print("STEP 2: ENVIRONMENT INITIALIZATION")
    print("-"*80)
    env = TradingEnvironment(df)
    logger.info(f"Environment created:")
    logger.info(f"  State dimension: {env.state_dim}")
    logger.info(f"  Action dimension: {env.action_dim}")
    logger.info(f"  Max steps per episode: {env.max_steps}")
    print()
    
    # Create agent
    print("STEP 3: DQN AGENT INITIALIZATION")
    print("-"*80)
    config = {
        'gamma': 0.95,
        'learning_rate': 0.0001,  # Lower LR для стабильности
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,      # Увеличен min epsilon
        'epsilon_decay': 0.995,
        'target_update_freq': 20, # Чаще обновляем target network
        'batch_size': 64,         # Увеличен batch size
        'buffer_capacity': 10000,
        'hidden_dims': [128, 128]  # Больше neurons
    }
    agent = DQNAgent(env.state_dim, env.action_dim, config)
    
    logger.info(f"DQN Agent configured:")
    logger.info(f"  Gamma (discount): {agent.gamma}")
    logger.info(f"  Learning rate: {agent.learning_rate}")
    logger.info(f"  Epsilon: {agent.epsilon_start} → {agent.epsilon_end}")
    logger.info(f"  Batch size: {config['batch_size']}")
    logger.info(f"  Hidden layers: {config['hidden_dims']}")
    print()
    
    # Training loop
    print("STEP 4: TRAINING")
    print("="*80)
    print(f"Total episodes: {num_episodes}")
    print(f"Warmup episodes: {warmup_episodes}")
    print("="*80 + "\n")
    
    best_reward = -float('inf')
    best_pnl = -float('inf')
    episode_rewards = []
    episode_losses = []
    episode_pnls = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        loss_count = 0
        steps = 0
        
        # Episode loop
        while True:
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train (only after warmup)
            if episode >= warmup_episodes:
                loss = agent.train_step()
                
                # Check for NaN
                if np.isnan(loss) or np.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at episode {episode}, step {steps}! Skipping...")
                    loss = 0.0
                else:
                    episode_loss += loss
                    loss_count += 1
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        # Episode end
        agent.episode_end()
        
        # Get episode stats
        stats = env.get_episode_stats()
        
        episode_rewards.append(episode_reward)
        episode_pnls.append(stats['total_pnl'])
        
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
        episode_losses.append(avg_loss)
        
        # Logging
        if episode % 5 == 0 or episode == num_episodes - 1:
            recent_rewards = episode_rewards[-10:]
            recent_pnls = episode_pnls[-10:]
            recent_losses = episode_losses[-10:]
            
            avg_reward = np.mean(recent_rewards)
            avg_pnl = np.mean(recent_pnls)
            avg_loss = np.mean(recent_losses)
            
            status = "WARMUP" if episode < warmup_episodes else "TRAINING"
            
            print(f"\n[{status}] Episode {episode+1}/{num_episodes}")
            print(f"  Reward: {episode_reward:+.4f} (avg-10: {avg_reward:+.4f})")
            
            if episode >= warmup_episodes:
                print(f"  Loss: {episode_losses[-1]:.6f} (avg-10: {avg_loss:.6f})")
            
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Final equity: ${stats['final_equity']:,.2f}")
            print(f"  Total PnL: {stats['total_pnl']:+.2%} (avg-10: {avg_pnl:+.2%})")
            print(f"  Max DD: {stats['max_drawdown']:.2%}")
            print(f"  Actions: Enter={stats['actions']['enter']}, Exit={stats['actions']['exit']}, Hold={stats['actions']['no_action']}")
            print(f"  Buffer size: {len(agent.replay_buffer)}")
        
        # Save best model (by reward)
        if episode_reward > best_reward and episode >= warmup_episodes:
            best_reward = episode_reward
            agent.save('models/dqn_best_reward.pth')
            logger.info(f"  ✅ New best reward model saved! Reward: {best_reward:.4f}")
        
        # Save best model (by PnL)
        if stats['total_pnl'] > best_pnl and episode >= warmup_episodes:
            best_pnl = stats['total_pnl']
            agent.save('models/dqn_best_pnl.pth')
            logger.info(f"  ✅ New best PnL model saved! PnL: {best_pnl:.2%}")
        
        # Periodic save
        if (episode + 1) % save_freq == 0 and episode >= warmup_episodes:
            agent.save(f'models/dqn_episode_{episode+1}.pth')
            logger.info(f"  💾 Checkpoint saved: episode {episode+1}")
    
    # Final save
    agent.save('models/dqn_final.pth')
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nPerformance Summary:")
    print(f"  Best reward: {best_reward:.4f}")
    print(f"  Best PnL: {best_pnl:+.2%}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Replay buffer size: {len(agent.replay_buffer)}")
    
    print(f"\nModels saved to models/:")
    print(f"  • dqn_best_reward.pth (best reward: {best_reward:.4f})")
    print(f"  • dqn_best_pnl.pth (best PnL: {best_pnl:+.2%})")
    print(f"  • dqn_final.pth (final model)")
    print()
    
    # Save training history
    history_df = pd.DataFrame({
        'episode': range(1, num_episodes + 1),
        'reward': episode_rewards,
        'pnl': episode_pnls,
        'loss': episode_losses
    })
    history_df.to_csv('models/training_history.csv', index=False)
    logger.info("Training history saved to models/training_history.csv")
    
    return agent, episode_rewards, episode_losses


def test_trained_agent(agent, df, num_episodes=10):
    """
    Test trained agent on data
    
    Parameters:
    -----------
    agent : DQNAgent
        Trained agent
    df : DataFrame
        Test data
    num_episodes : int
        Number of test episodes
    """
    print("\n" + "="*80)
    print("TESTING TRAINED AGENT")
    print("="*80 + "\n")
    
    env = TradingEnvironment(df)
    
    # Set epsilon to 0 (pure exploitation)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    test_rewards = []
    test_pnls = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        stats = env.get_episode_stats()
        test_rewards.append(episode_reward)
        test_pnls.append(stats['total_pnl'])
        
        print(f"Test Episode {episode+1}/{num_episodes}:")
        print(f"  Reward: {episode_reward:.4f}")
        print(f"  PnL: {stats['total_pnl']:+.2%}")
        print(f"  Final equity: ${stats['final_equity']:,.2f}")
        print(f"  Actions: {stats['actions']}")
        print()
    
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    avg_reward = np.mean(test_rewards)
    avg_pnl = np.mean(test_pnls)
    
    print("="*80)
    print("TEST RESULTS:")
    print(f"  Average reward: {avg_reward:.4f}")
    print(f"  Average PnL: {avg_pnl:+.2%}")
    print("="*80 + "\n")
    
    return test_rewards, test_pnls


if __name__ == "__main__":
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train
    print("\n🚀 Starting DQN Training Pipeline...\n")
    
    agent, rewards, losses = train_dqn(
        num_episodes=100,
        save_freq=20,
        warmup_episodes=10
    )
    
    # Test
    print("\n🧪 Testing trained agent...\n")
    test_df = generate_training_data(n_bars=1000, seed=123)
    test_rewards, test_pnls = test_trained_agent(agent, test_df, num_episodes=5)
    
    print("\n" + "="*80)
    print("✅ ALL DONE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Check models/ directory for saved models")
    print("  2. Review training_history.csv for performance")
    print("  3. Integrate best model into Hybrid Strategy")
    print("  4. Re-run Dispersion Analysis with RL component")
    print()
