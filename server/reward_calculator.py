# reward_calculator.py

class RewardCalculator:
    @staticmethod
    def calculate_reward(game_state, previous_state_vector, bot_id, bot_states):
        try:
            reward = 0
            
            # Safely get values with defaults
            total_treasures = max(1, game_state.get('total_treasures', 1))
            
            # Treasure collection reward (increased)
            current_treasures = game_state.get('bot_treasures', 0)
            previous_treasures = previous_state_vector[4] * total_treasures
            if current_treasures > previous_treasures:
                reward += 25  # Increased from 10
            
            # Visibility rewards (increased for aggressive play)
            visible_treasures = game_state.get('visible_treasures', [])
            visible_bots = game_state.get('visible_bots', [])
            
            if len(visible_treasures) > 0:
                reward += 2  # Reward for seeing treasures
                # Extra reward for moving toward treasures
                if bot_states[bot_id]['last_action'] in ['move_up', 'move_down', 'move_left', 'move_right']:
                    reward += 1
            
            if len(visible_bots) > 0:
                reward += 2 * len(visible_bots)  # Reward for seeing other bots
                
                # Extra reward for aggressive actions when bots are visible
                if bot_states[bot_id]['last_action'] == 'drop_trap':
                    reward += 5
                elif bot_states[bot_id]['last_action'] == 'explode_traps':
                    reward += 7
            
            # Penalty for dying (increased)
            if not game_state.get('bot_alive', True):
                reward -= 50  # Increased from 20
            
            # Movement penalty (reduced) to encourage exploration
            if bot_states[bot_id]['last_action'] in ['move_up', 'move_down', 'move_left', 'move_right']:
                reward -= 0.1
            
            # Waiting penalty (increased to discourage inactivity)
            if bot_states[bot_id]['last_action'] == 'wait':
                reward -= 2
            
            # Track treasure collection for inactivity penalty
            if 'steps_without_treasure' not in bot_states[bot_id]:
                bot_states[bot_id]['steps_without_treasure'] = 0
            
            if current_treasures <= previous_treasures:
                bot_states[bot_id]['steps_without_treasure'] += 1
            else:
                bot_states[bot_id]['steps_without_treasure'] = 0
            
            # Increasing penalty for not finding treasures
            if bot_states[bot_id]['steps_without_treasure'] > 10:
                reward -= (bot_states[bot_id]['steps_without_treasure'] - 10) * 0.5
            
            # Ranking reward (competitive pressure)
            bot_scores = game_state.get('bot_scores', [])
            my_score = next((score for score in bot_scores if score['id'] == bot_id), None)
            if my_score and bot_scores:
                my_rank = sorted(bot_scores, key=lambda x: x['treasures'], reverse=True).index(my_score)
                reward += (len(bot_scores) - my_rank - 1) * 2  # Increased multiplier
            
            # Strategic positioning reward
            if len(visible_treasures) > 0 and len(visible_bots) > 0:
                # Reward for being between a treasure and another bot
                reward += 3
            
            # Trap usage efficiency
            if game_state.get('bot_remaining_traps', 0) < previous_state_vector[5] * 10:
                # Used a trap - check if it was strategic
                if len(visible_bots) > 0:
                    reward += 5  # Good trap placement
                else:
                    reward -= 2  # Wasted trap
            
            return reward
        
        except Exception as e:
            print(f"Error in calculate_reward: {e}")
            return 0