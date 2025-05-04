# game_state_processor.py
import numpy as np

class GameStateProcessor:
    @staticmethod
    def get_state_vector(game_state, state_size=68):
        try:
            features = []
            
            # Bot position (normalized) - 2 features
            features.extend([
                game_state['bot_position']['x'] / 40,
                game_state['bot_position']['y'] / 40
            ])
            
            # Bot direction (sin and cos) - 2 features
            features.extend([
                np.sin(game_state['bot_direction']),
                np.cos(game_state['bot_direction'])
            ])
            
            # Bot stats - 3 features
            total_treasures = max(1, game_state['total_treasures'])
            features.extend([
                game_state['bot_treasures'] / total_treasures,
                game_state['bot_remaining_traps'] / 10,  # max traps is 10
                float(game_state['bot_alive'])
            ])
            
            # Game progress - 2 features
            features.extend([
                game_state['treasures_collected'] / total_treasures,
                game_state.get('time_remaining', 60000) / 60000
            ])
            
            # Visible treasures (up to 5) - 10 features
            visible_treasures = game_state.get('visible_treasures', [])[:5]
            for i in range(5):
                if i < len(visible_treasures):
                    features.extend([
                        visible_treasures[i]['x'] / 10,
                        visible_treasures[i]['y'] / 10
                    ])
                else:
                    features.extend([0, 0])
            
            # Visible bots (up to 3) - 12 features
            visible_bots = game_state.get('visible_bots', [])[:3]
            for i in range(3):
                if i < len(visible_bots):
                    features.extend([
                        visible_bots[i]['x'] / 10,
                        visible_bots[i]['y'] / 10,
                        visible_bots[i]['treasures'] / total_treasures,
                        visible_bots[i].get('traps_remaining', 0) / 10
                    ])
                else:
                    features.extend([0, 0, 0, 0])
            
            # Visible traps (up to 3) - 9 features
            visible_traps = game_state.get('visible_traps', [])[:3]
            for i in range(3):
                if i < len(visible_traps):
                    features.extend([
                        visible_traps[i]['x'] / 10,
                        visible_traps[i]['y'] / 10,
                        1 if visible_traps[i].get('owner') == 'self' else 0
                    ])
                else:
                    features.extend([0, 0, 0])
            
            # Obstacle grid (5x5) - 25 features
            obstacle_grid = np.zeros(25)
            for obstacle in game_state.get('visible_obstacles', []):
                x, y = obstacle['x'] + 2, obstacle['y'] + 2
                if 0 <= x < 5 and 0 <= y < 5:
                    obstacle_grid[y * 5 + x] = 1
            features.extend(obstacle_grid.tolist())
            
            # Other bots' scores - 3 features
            bot_scores = game_state.get('bot_scores', [])
            bot_scores_sorted = sorted(bot_scores, key=lambda x: x['treasures'], reverse=True)
            score_features = []
            for score in bot_scores_sorted:
                if score['id'] != game_state['bot_id'] and len(score_features) < 3:
                    score_features.append(score['treasures'] / total_treasures)
            while len(score_features) < 3:
                score_features.append(0)
            features.extend(score_features)
            
            # Can we explode traps? - 1 feature
            features.append(float(game_state.get('explosions_available', False)))
            
            # Total features: 2 + 2 + 3 + 2 + 10 + 12 + 9 + 25 + 3 + 1 = 69
            # We need to remove 1 feature to get to 68
            
            # Print debugging info if we have wrong size
            total_features = len(features)
            if total_features != state_size:
                print(f"Warning: Expected {state_size} features, but got {total_features}")
                print(f"Feature breakdown:")
                print(f"  Position: 2")
                print(f"  Direction: 2")
                print(f"  Bot stats: 3")
                print(f"  Game progress: 2")
                print(f"  Visible treasures: 10")
                print(f"  Visible bots: 12")
                print(f"  Visible traps: 9")
                print(f"  Obstacle grid: 25")
                print(f"  Bot scores: 3")
                print(f"  Explosion available: 1")
                print(f"  Total: {total_features}")
            
            # Ensure we return exactly state_size features
            if len(features) > state_size:
                features = features[:state_size]
            elif len(features) < state_size:
                features.extend([0] * (state_size - len(features)))
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error in get_state_vector: {e}")
            return np.zeros(state_size, dtype=np.float32)