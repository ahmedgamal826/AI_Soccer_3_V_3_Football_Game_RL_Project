import asyncio
import platform
import pygame
import numpy as np
from gym import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Initialize Pygame and mixer
pygame.init()
pygame.mixer.init()

# Game settings
WIDTH, HEIGHT = 800, 600
FPS = 30
PLAYER_SIZE = 40
BALL_SIZE = 30
TEAM_COLORS = [(255, 0, 0), (0, 0, 255)]  # Red, Blue
FONT_SIZE = 10

# Load sound (assuming Sound.mp3 is in your directory)
try:
    WHISTLE_SOUND = pygame.mixer.Sound("Sound.mp3")
except FileNotFoundError:
    print("Sound.mp3 not found. Please ensure it's in the working directory.")
    WHISTLE_SOUND = None

class AdvancedSoccerEnv(Env):
    
    def __init__(self):
        super(AdvancedSoccerEnv, self).__init__()

        self.action_space = spaces.MultiDiscrete([5, 5, 5, 5])
        self.observation_space = spaces.Box(low=-1, high=1, shape=(24,), dtype=np.float32)
        
        self.ball_image = pygame.image.load("football.png")
        self.ball_image = pygame.transform.scale(self.ball_image, (BALL_SIZE, BALL_SIZE))

        # Physics settings
        self.red_max_speed = 5
        self.blue_max_speed = 5
        self.friction = 0.82
        self.kick_power = 20
        self.goal_size = 120
        self.scores = {'red': 0, 'blue': 0}
        self.last_kicker = None
        self.goal_scorer_text = None
        self.goal_scorer_timer = 0
        self.min_distance = 150
        self.last_ball_pos = None

        # Timer settings
        self.total_time = 5400
        self.time_scale = 5400 / 180
        self.current_time = 0
        self.halftime_shown = False
        self.fulltime_shown = False
        self.dialog_timer = 0
        self.dialog_text = None

        # Pygame setup
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.score_font = pygame.font.SysFont('Arial', 30)
        self.timer_font = pygame.font.SysFont('Arial', 30)
        self.dialog_font = pygame.font.SysFont('Arial', 40)
        pygame.display.set_caption("AI Football Soccer 2v2 - Pass & Goal")
        self.clock = pygame.time.Clock()

        self.running = True
        self.reset()

    def reset(self):
        self.players = {
            'Elnemr': pygame.Rect(WIDTH//2 - 200, HEIGHT//2 - 150, PLAYER_SIZE, PLAYER_SIZE),
            'Beherry': pygame.Rect(WIDTH//2 - 200, HEIGHT//2 + 150, PLAYER_SIZE, PLAYER_SIZE),
            'Abanoub': pygame.Rect(WIDTH//2 + 200, HEIGHT//2 - 150, PLAYER_SIZE, PLAYER_SIZE),
            'ElShokary': pygame.Rect(WIDTH//2 + 200, HEIGHT//2 + 150, PLAYER_SIZE, PLAYER_SIZE)
        }
        self.ball = pygame.Rect(WIDTH//2, HEIGHT//2, BALL_SIZE, BALL_SIZE)
        self.velocities = {
            'Elnemr': [0, 0], 'Beherry': [0, 0], 'Abanoub': [0, 0], 'ElShokary': [0, 0], 'ball': [0, 0]
        }
        self.last_kicker = None
        self.goal_scorer_text = None
        self.goal_scorer_timer = 0
        self.last_ball_pos = (self.ball.x, self.ball.y)
        if self.current_time == 0:
            self.current_time = 0
            self.halftime_shown = False
            self.fulltime_shown = False
            self.dialog_timer = 0
            self.dialog_text = None
        return self._get_obs()

    def _get_obs(self):
        ball_pos = [
            self.ball.x/WIDTH - 0.5, self.ball.y/HEIGHT - 0.5,
            self.velocities['ball'][0]/self.kick_power, self.velocities['ball'][1]/self.kick_power
        ]
        player_obs = []
        for team in ['Elnemr', 'Beherry', 'Abanoub', 'ElShokary']:
            player_obs.extend([
                (self.players[team].x - self.ball.x)/WIDTH,
                (self.players[team].y - self.ball.y)/HEIGHT,
                self.velocities[team][0]/(self.red_max_speed if 'Elnemr' in team or 'Beherry' in team else self.blue_max_speed),
                self.velocities[team][1]/(self.red_max_speed if 'Elnemr' in team or 'Beherry' in team else self.blue_max_speed),
                (WIDTH - self.players[team].x if 'Elnemr' in team or 'Beherry' in team else self.players[team].x)/WIDTH
            ])
        return np.array(ball_pos + player_obs, dtype=np.float32)

    def step(self, actions):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        if self.dialog_text and (pygame.time.get_ticks() - self.dialog_timer < 5000):
            return self._get_obs(), 0, False, {}  

        self.last_ball_pos = (self.ball.x, self.ball.y)
        self._apply_action(actions[0], 'Elnemr')
        self._apply_action(actions[1], 'Beherry')
        self._apply_action(actions[2], 'Abanoub')
        self._apply_action(actions[3], 'ElShokary')
        self._apply_repulsion()
        self._update_physics()

        # Update timer with scaled time
        self.current_time += (1.0 / FPS) * self.time_scale  # Increment by scaled frame time

        reward = self._calculate_rewards()
        done = self._check_goal() or not self.running or self.current_time >= self.total_time
        return self._get_obs(), reward, done, {}

    def _apply_action(self, action, team):
        distance_to_ball = np.hypot(self.ball.x - self.players[team].x, self.ball.y - self.players[team].y)
        base_speed = 5
        speed_modifier = min(1.5, max(0.5, 200 / (distance_to_ball + 1e-5)))
        max_speed = base_speed * speed_modifier

        dx_ball = self.ball.x - self.players[team].x
        dy_ball = self.ball.y - self.players[team].y
        distance_ball = np.hypot(dx_ball, dy_ball)

        if distance_ball > PLAYER_SIZE + BALL_SIZE:
            self.velocities[team][0] += speed_modifier * np.sign(dx_ball)
            self.velocities[team][1] += speed_modifier * np.sign(dy_ball)
        else:
            if action == 0: self.velocities[team][1] -= speed_modifier
            elif action == 1: self.velocities[team][1] += speed_modifier
            elif action == 2: self.velocities[team][0] -= speed_modifier
            elif action == 3: self.velocities[team][0] += speed_modifier
            elif action == 4: self._smart_kick(team)

        self.velocities[team][0] = np.clip(self.velocities[team][0] * self.friction, -max_speed, max_speed)
        self.velocities[team][1] = np.clip(self.velocities[team][1] * self.friction, -max_speed, max_speed)


    def _apply_repulsion(self):
        for team1, team2 in [('Elnemr', 'Beherry'), ('Abanoub', 'ElShokary')]:
            dx = self.players[team1].x - self.players[team2].x
            dy = self.players[team1].y - self.players[team2].y
            distance = np.hypot(dx, dy)
            if distance < self.min_distance and distance > 0:
                force = 2.5 * (self.min_distance - distance) / self.min_distance
                max_speed = self.red_max_speed if 'Elnemr' in team1 or 'Beherry' in team1 else self.blue_max_speed
                self.velocities[team1][0] += np.clip(force * dx / distance, -max_speed, max_speed)
                self.velocities[team1][1] += np.clip(force * dy / distance, -max_speed, max_speed)
                self.velocities[team2][0] -= np.clip(force * dx / distance, -max_speed, max_speed)
                self.velocities[team2][1] -= np.clip(force * dy / distance, -max_speed, max_speed)
        
    def _smart_kick(self, team):
        self.last_kicker = team
        teammate = 'Beherry' if team == 'Elnemr' else 'Elnemr' if team == 'Beherry' else 'ElShokary' if team == 'Abanoub' else 'Abanoub'
        distance_to_teammate = np.hypot(self.players[teammate].x - self.ball.x, self.players[teammate].y - self.ball.y)
        distance_to_goal = abs(self.ball.x - (WIDTH if 'Elnemr' in team or 'Beherry' in team else 0))

        teammate_to_goal = abs(self.players[teammate].x - (WIDTH if 'Elnemr' in team or 'Beherry' in team else 0))
        if distance_to_goal < 150 and distance_to_goal < teammate_to_goal:
            target_x = WIDTH if 'Elnemr' in team or 'Beherry' in team else 0
            target_y = HEIGHT // 2 + np.random.uniform(-30, 30)
            dx = target_x - self.ball.x
            dy = target_y - self.ball.y
        elif distance_to_teammate < 250 and teammate_to_goal < distance_to_goal:
            dx = self.players[teammate].x - self.ball.x
            dy = self.players[teammate].y - self.ball.y
        else:
            target_x = WIDTH if 'Elnemr' in team or 'Beherry' in team else 0
            target_y = HEIGHT // 2 + np.random.uniform(-30, 30)
            dx = target_x - self.ball.x
            dy = target_y - self.ball.y

        distance = np.hypot(dx, dy)
        if distance != 0:
            power = self.kick_power * (1.5 + (distance / WIDTH))
            self.velocities['ball'][0] = (dx / distance) * power
            self.velocities['ball'][1] = (dy / distance) * power

    def _update_physics(self):
        for entity in self.players:
            self.players[entity].x += self.velocities[entity][0]
            self.players[entity].y += self.velocities[entity][1]
            self._enforce_boundaries(entity)
        self.ball.x += self.velocities['ball'][0]
        self.ball.y += self.velocities['ball'][1]
        self.velocities['ball'][0] *= self.friction
        self.velocities['ball'][1] *= self.friction
        self._enforce_boundaries('ball')

    def _enforce_boundaries(self, entity):
        if entity in self.players:
            self.players[entity].x = np.clip(self.players[entity].x, 0, WIDTH - PLAYER_SIZE)
            self.players[entity].y = np.clip(self.players[entity].y, 0, HEIGHT - PLAYER_SIZE)
        else:
            self.ball.x = np.clip(self.ball.x, 0, WIDTH - BALL_SIZE)
            self.ball.y = np.clip(self.ball.y, 0, HEIGHT - BALL_SIZE)

    def _calculate_rewards(self):
        rewards = 0
        goal_scored = False  
        scoring_team = None     

        for team in ['red', 'blue']:
            target_x = 0 if team == 'blue' else WIDTH
            players = ['Elnemr', 'Beherry'] if team == 'red' else ['Abanoub', 'ElShokary']
            for player in players:
                ball_distance = np.hypot(self.players[player].x - self.ball.x, self.players[player].y - self.ball.y)
                rewards += 15.0 / (ball_distance + 1e-5)
                goal_progress = abs(self.ball.x - target_x) / WIDTH
                rewards += (1 - goal_progress) * 20.0

                if self._is_goal_scored(team) and not goal_scored:  
                    rewards += 1000.0
                    self.scores[team] += 1  
                    goal_scored = True  
                    scoring_team = team
                    if self.last_kicker:
                        self.goal_scorer_text = self.score_font.render(f"Goal by {self.last_kicker}!", True, (255, 255, 255))
                        self.goal_scorer_timer = pygame.time.get_ticks()

                if self.last_kicker == player and self.last_ball_pos:
                    teammate = 'Beherry' if player == 'Elnemr' else 'Elnemr' if player == 'Beherry' else 'ElShokary' if player == 'Abanoub' else 'Abanoub'
                    old_distance_to_teammate = np.hypot(self.players[teammate].x - self.last_ball_pos[0], self.players[teammate].y - self.last_ball_pos[1])
                    new_distance_to_teammate = np.hypot(self.players[teammate].x - self.ball.x, self.players[teammate].y - self.ball.y)
                    if new_distance_to_teammate < old_distance_to_teammate and new_distance_to_teammate < 200:
                        rewards += 100.0

        self.last_ball_pos = (self.ball.x, self.ball.y)
        return rewards

    def _is_goal_scored(self, team):
        if team == 'red' and self.ball.x >= WIDTH - BALL_SIZE and HEIGHT//2 - self.goal_size//2 < self.ball.y < HEIGHT//2 + self.goal_size//2:
            return True
        if team == 'blue' and self.ball.x <= BALL_SIZE and HEIGHT//2 - self.goal_size//2 < self.ball.y < HEIGHT//2 + self.goal_size//2:
            return True
        return False

    def _check_goal(self):
        return self._is_goal_scored('red') or self._is_goal_scored('blue')

    def _draw_player(self, x, y, team):
        color = TEAM_COLORS[0] if 'Elnemr' in team or 'Beherry' in team else TEAM_COLORS[1]
        pygame.draw.circle(self.screen, color, (x, y - 15), 10)  # Head
        pygame.draw.rect(self.screen, color, (x - 7.5, y - 5, 15, 25))  # Body
        pygame.draw.line(self.screen, color, (x - 5, y + 20), (x - 5, y + 35), 3)  # Left leg
        pygame.draw.line(self.screen, color, (x + 5, y + 20), (x + 5, y + 35), 3)  # Right leg
        pygame.draw.line(self.screen, color, (x - 7.5, y - 5), (x - 15, y + 5), 3)  # Left arm
        pygame.draw.line(self.screen, color, (x + 7.5, y - 5), (x + 15, y + 5), 3)  # Right arm
        player_font = pygame.font.SysFont('Arial', 20)
        label = player_font.render(team, True, (255, 255, 255))
        self.screen.blit(label, (x - label.get_width()//2, y - 35))


    def render(self, mode='human'):
        self.screen.fill((30, 150, 30))
        # Draw field
        pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, WIDTH, HEIGHT), 5)
        pygame.draw.line(self.screen, (255, 255, 255), (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)
        pygame.draw.circle(self.screen, (255, 255, 255), (WIDTH // 2, HEIGHT // 2), 50, 2)

        # Draw goals
        self._draw_goal(0, HEIGHT // 2 - self.goal_size // 2, 20, self.goal_size)
        self._draw_goal(WIDTH - 20, HEIGHT // 2 - self.goal_size // 2, 20, self.goal_size)

        # Draw players
        for team, rect in self.players.items():
            self._draw_player(rect.x + PLAYER_SIZE // 2, rect.y + PLAYER_SIZE // 2, team)
        
        # Draw ball
        self.screen.blit(self.ball_image, (self.ball.x, self.ball.y))

        # Draw score
        score_text = self.score_font.render(f"Red: {self.scores['red']} - {self.scores['blue']} :Blue", True, (255, 255, 255))
        self.screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))

        # Draw timer next to score on the right
        minutes = int(self.current_time // 60)
        seconds = int(self.current_time % 60)
        timer_text = self.timer_font.render(f"{minutes:02d}:{seconds:02d}", True, (255, 255, 255))
        self.screen.blit(timer_text, (WIDTH - timer_text.get_width() - 10, 20))

        # Handle halftime (45 minutes = 2700 seconds in-game)
        if self.current_time >= 2700 and not self.halftime_shown:
            if WHISTLE_SOUND:
                WHISTLE_SOUND.play()  
                pygame.time.wait(int(WHISTLE_SOUND.get_length() * 1000))  
            self.halftime_shown = True
            self.dialog_timer = pygame.time.get_ticks()
            self.dialog_text = self.dialog_font.render(f"Half Time: Red {self.scores['red']} - {self.scores['blue']} Blue", True, (255, 255, 255))
            
            self.players = {
                'Elnemr': pygame.Rect(WIDTH//2 - 200, HEIGHT//2 - 150, PLAYER_SIZE, PLAYER_SIZE),
                'Beherry': pygame.Rect(WIDTH//2 - 200, HEIGHT//2 + 150, PLAYER_SIZE, PLAYER_SIZE),
                'Abanoub': pygame.Rect(WIDTH//2 + 200, HEIGHT//2 - 150, PLAYER_SIZE, PLAYER_SIZE),
                'ElShokary': pygame.Rect(WIDTH//2 + 200, HEIGHT//2 + 150, PLAYER_SIZE, PLAYER_SIZE)
            }
            self.ball = pygame.Rect(WIDTH//2, HEIGHT//2, BALL_SIZE, BALL_SIZE)
            self.velocities = {
                'Elnemr': [0, 0], 'Beherry': [0, 0], 'Abanoub': [0, 0], 'ElShokary': [0, 0], 'ball': [0, 0]
            }
            self.last_kicker = None
            self.last_ball_pos = (self.ball.x, self.ball.y)

        # Handle full time (90 minutes = 5400 seconds in-game)
        if self.current_time >= 5400 and not self.fulltime_shown:
            if WHISTLE_SOUND:
                WHISTLE_SOUND.play()
                pygame.time.wait(int(WHISTLE_SOUND.get_length() * 1000))
            self.fulltime_shown = True
            self.dialog_timer = pygame.time.get_ticks()
            self.dialog_text = self.dialog_font.render(f"Full Time: Red {self.scores['red']} - {self.scores['blue']} Blue", True, (255, 255, 255))

        # Display dialog for 5 seconds (halftime or full time)
        if self.dialog_text and (pygame.time.get_ticks() - self.dialog_timer < 5000):
            dialog_rect = self.dialog_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            pygame.draw.rect(self.screen, (0, 0, 0, 180), (dialog_rect.x - 20, dialog_rect.y - 20, dialog_rect.width + 40, dialog_rect.height + 40))
            self.screen.blit(self.dialog_text, dialog_rect)
        elif self.fulltime_shown and (pygame.time.get_ticks() - self.dialog_timer >= 5000):
            self.running = False  

        # Draw goal scorer text
        if self.goal_scorer_text and (pygame.time.get_ticks() - self.goal_scorer_timer < 3000):
            self.screen.blit(self.goal_scorer_text, (WIDTH // 2 - self.goal_scorer_text.get_width() // 2, HEIGHT // 2))

        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_goal(self, x, y, width, height):
        pygame.draw.rect(self.screen, (255, 255, 255), (x, y, width, height), 3)
        for i in range(1, 6):
            pygame.draw.line(self.screen, (255, 255, 255), (x + i * (width // 6), y), (x + i * (width // 6), y + height))
        for j in range(1, 6):
            pygame.draw.line(self.screen, (255, 255, 255), (x, y + j * (height // 6)), (x + width, y + j * (height // 6)))

def load_or_train_model():
    env = DummyVecEnv([lambda: AdvancedSoccerEnv()])
    model_path = "soccer_2v2_pass_and_goal_ai"
    try:
        model = PPO.load(model_path, env=env)
        print("تم تحميل الموديل المدرب بنجاح!")
    except FileNotFoundError:
        print("لم يتم العثور على الموديل، جاري التدريب...")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=4096, batch_size=512, n_epochs=20)
        model.learn(total_timesteps=200_000)
        model.save(model_path)
        print("تم تدريب الموديل وحفظه بنجاح!")
    return model

async def main():
    model = load_or_train_model()
    env = AdvancedSoccerEnv()
    while env.running:
        obs = env.reset()
        done = False
        while not done and env.running:
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)
            env.render()
            await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())