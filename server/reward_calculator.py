# reward_calculator.py
from agent_config import REWARD_CONFIG

class RewardCalculator:
    @staticmethod
    def calculate_reward(game_state, previous_state_vector, bot_id, bot_states):
        try:
            reward = 0
            
            # Safely get values with defaults
            total_treasures = max(1, game_state.get('total_treasures', 1))
            
            # 1. TREASURE COLLECTION REWARD (Primary objective)
            current_treasures = game_state.get('bot_treasures', 0)
            if previous_state_vector is not None and len(previous_state_vector) > 4:
                previous_treasures = previous_state_vector[4] * total_treasures
            else:
                previous_treasures = 0
            
            treasures_collected = current_treasures - previous_treasures
            if treasures_collected > 0:
                reward += treasures_collected * REWARD_CONFIG['treasure_collection']
            
            # 2. SPOTTING ENEMIES REWARD (Information gathering)
            visible_bots = game_state.get('visible_bots', [])
            if len(visible_bots) > 0:
                reward += len(visible_bots) * REWARD_CONFIG['spot_enemy']
                
                # Extra reward for spotting enemies with treasures
                for bot in visible_bots:
                    if bot.get('treasures', 0) > 0:
                        reward += REWARD_CONFIG['spot_enemy_with_treasure']
            
            # 3. ENEMIES DESTROYED REWARD (Combat success)
            # Track enemy eliminations by checking alive status of visible bots
            if 'previous_visible_bots' in bot_states[bot_id]:
                previous_visible = bot_states[bot_id]['previous_visible_bots']
                current_visible = {bot['id']: bot for bot in visible_bots}
                
                # Check if any previously visible bot is now gone (potentially eliminated)
                for prev_bot_id, prev_bot in previous_visible.items():
                    if prev_bot_id not in current_visible:
                        # Bot disappeared from view - check if it's actually dead
                        all_bots = game_state.get('bot_scores', [])
                        dead_bot = next((b for b in all_bots if b['id'] == prev_bot_id and not b['alive']), None)
                        if dead_bot:
                            reward += REWARD_CONFIG['enemy_elimination']
                            # Extra reward for eliminating enemies with treasures
                            if dead_bot.get('treasures', 0) > 0:
                                reward += dead_bot['treasures'] * REWARD_CONFIG['eliminate_enemy_with_treasure_multiplier']
            
            # Update tracking for next iteration
            bot_states[bot_id]['previous_visible_bots'] = {bot['id']: bot for bot in visible_bots}
            
            # 4. PENALTIES
            
            # Penalty for wasted explosion (no kills)
            if bot_states[bot_id]['last_action'] == 'explode_traps':
                # Check if this explosion resulted in any kills
                explosion_killed_someone = False
                
                # Simple heuristic: if we exploded and there are fewer alive bots than before
                if 'previous_alive_count' in bot_states[bot_id]:
                    current_alive = sum(1 for b in game_state.get('bot_scores', []) if b['alive'])
                    if current_alive < bot_states[bot_id]['previous_alive_count']:
                        explosion_killed_someone = True
                
                if not explosion_killed_someone and len(visible_bots) == 0:
                    reward += REWARD_CONFIG['wasted_explosion_penalty']
            
            # Update alive count for next iteration
            bot_states[bot_id]['previous_alive_count'] = sum(1 for b in game_state.get('bot_scores', []) if b['alive'])
            
            # Penalty for being spotted (inferred from game state)
            # Check if we appear in other bots' visible lists
            my_position = game_state.get('bot_position', {})
            for other_bot in visible_bots:
                # If another bot is looking roughly in our direction, we might be spotted
                other_x = other_bot.get('x', 0) + my_position.get('x', 0)
                other_y = other_bot.get('y', 0) + my_position.get('y', 0)
                
                # Simple check: if we're close to another bot, assume we might be spotted
                distance = abs(other_x - my_position.get('x', 0)) + abs(other_y - my_position.get('y', 0))
                if distance < 5:  # Within 5 tiles
                    reward += REWARD_CONFIG['being_spotted_penalty']
            
            # Penalty for dying
            if not game_state.get('bot_alive', True):
                reward += REWARD_CONFIG['death_penalty']
            
            # Small penalties/rewards for basic actions
            if bot_states[bot_id]['last_action'] in ['move_up', 'move_down', 'move_left', 'move_right']:
                reward += REWARD_CONFIG['movement_cost']
            elif bot_states[bot_id]['last_action'] == 'wait':
                reward += REWARD_CONFIG['wait_penalty']
            elif bot_states[bot_id]['last_action'] == 'drop_trap':
                if len(visible_bots) > 0:
                    reward += REWARD_CONFIG['strategic_trap_reward']
                else:
                    reward += REWARD_CONFIG['blind_trap_penalty']
            
            # Survival bonus (small reward for staying alive)
            if game_state.get('bot_alive', True):
                reward += REWARD_CONFIG['survival_bonus']
            
            # Competition reward based on ranking
            bot_scores = game_state.get('bot_scores', [])
            my_score = next((score for score in bot_scores if score['id'] == bot_id), None)
            if my_score and bot_scores:
                my_rank = sorted(bot_scores, key=lambda x: x.get('treasures', 0), reverse=True).index(my_score)
                if my_rank == 0:
                    reward += REWARD_CONFIG['leader_bonus']
                elif my_rank == len(bot_scores) - 1:
                    reward += REWARD_CONFIG['last_place_penalty']
            
            return reward
        
        except Exception as e:
            print(f"Error in calculate_reward: {e}")
            import traceback
            traceback.print_exc()
            return 0