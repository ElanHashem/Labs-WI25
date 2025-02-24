from tqdm import tqdm
from collections import defaultdict
from .blackjack import BlackjackEnv, Hand, Card

ACTIONS = ['hit', 'stick']

def policy_evaluation(env, V, policy, episodes=500000, gamma=1.0):
    """
    Monte Carlo policy evaluation:
    - Generate episodes using the current policy
    - Update state value function as an average return
    """
    # Track number of visits to each state and track sum of returns for each state
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    
    for _ in tqdm(range(episodes), desc="Policy evaluation"):
        episode = []
        state = env.reset()
        done = False
        
        # Generate one episode
        while not done:
            action = policy.get(state, 'hit')  # Ensure policy has a defined action
            next_state, reward, done = env.step(action)  # Ensure correct unpacking
            episode.append((state, reward))
            state = next_state
        
        # First-visit Monte Carlo: Update returns for the first occurrence of each state
        visited_states = set()
        G = 0  # Return (cumulative reward)
        for state, reward in reversed(episode):
            G = reward + gamma * G  # Apply proper return discounting
            if state not in visited_states:
                visited_states.add(state)  # Ensure first-visit MC
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]   # Average return update
                
                # Debugging output
                print(f"Debug: State={state}, Computed V={V[state]}, Expected V={1.5 if state == (10, 10, False) else 1.0}, Returns Sum={returns_sum[state]}, Returns Count={returns_count[state]}")
    
    return V
