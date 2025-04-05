import pygame
import numpy as np
from gym import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Tuple
import time
import random

pygame.mixer.init()
pygame.mixer.music.load("Sound.mp3")   

WIDTH, HEIGHT = 800, 600
FPS = 60
PLAYER_SIZE = 40
BALL_SIZE = 30
TEAM_COLORS = [(255, 0, 0), (0, 0, 255)]
FONT_SIZE = 30


class AdvancedSoccerEnv(Env):
    def __init__(self):
        super(AdvancedSoccerEnv, self).__init__()
        
        self.action_space = spaces.MultiDiscrete([5, 5])   
        self.observation_space = spaces.Box(
            low=-1, high=1, 
            shape=(14,),  
            dtype=np.float32
        )

        self.ball_image = pygame.image.load("football.png") 
        self.ball_image = pygame.transform.scale(self.ball_image, (BALL_SIZE, BALL_SIZE))
        
        self.max_speed = 10
        self.friction = 0.82
        self.kick_power = 25
        self.goal_size = 120
        self.scores = {'red': 0, 'blue': 0}

        self.clock = pygame.time.Clock()
        self.match_time = 0
        self.start_time = pygame.time.get_ticks()
        self.half = 1
        self.show_half_time_notification = False  
        self.notification_start_time = None    
        self.running = True

        
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.SysFont('Arial', FONT_SIZE)
        pygame.display.set_caption("AI Football Soccer")
        self.clock = pygame.time.Clock()


        self.match_time = 0 
        self.start_time = pygame.time.get_ticks()    
        self.half = 1  
        self.running = True   
        self.slow_player = random.choice(['red', 'blue'])   
        self.reset()

    def reset(self):
        self.players = {
            'red': pygame.Rect(3*WIDTH//4 - PLAYER_SIZE, HEIGHT//2, PLAYER_SIZE, PLAYER_SIZE),  # اللاعب الأحمر في الجهة اليمنى
            'blue': pygame.Rect(WIDTH//4, HEIGHT//2, PLAYER_SIZE, PLAYER_SIZE)  # اللاعب الأزرق في الجهة اليسرى
        }
        self.ball = pygame.Rect(WIDTH//2, HEIGHT//2, BALL_SIZE, BALL_SIZE)
        self.velocities = {
            'red': [0, 0], 'blue': [0, 0], 'ball': [0, 0]
        }

        self.min_distance_between_players = 200    

        return self._get_obs()



    def _get_obs(self) -> np.ndarray:
        """ملاحظات متقدمة مع سرعة الكرة"""
        ball_pos = [
            self.ball.x/WIDTH - 0.5,
            self.ball.y/HEIGHT - 0.5,
            self.velocities['ball'][0]/self.kick_power,
            self.velocities['ball'][1]/self.kick_power
        ]
        
        red_obs = [
            (self.players['red'].x - self.ball.x)/WIDTH,
            (self.players['red'].y - self.ball.y)/HEIGHT,
            self.velocities['red'][0]/self.max_speed,
            self.velocities['red'][1]/self.max_speed,
            (WIDTH - self.players['red'].x)/WIDTH
        ]
        
        blue_obs = [
            (self.players['blue'].x - self.ball.x)/WIDTH,
            (self.players['blue'].y - self.ball.y)/HEIGHT,
            self.velocities['blue'][0]/self.max_speed,
            self.velocities['blue'][1]/self.max_speed,
            self.players['blue'].x/WIDTH
        ]
        
        return np.array(ball_pos + red_obs + blue_obs, dtype=np.float32)
   
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        self._apply_action(actions[0], 'red')
        self._apply_action(actions[1], 'blue')
        self._update_physics()
        
        reward = self._calculate_rewards()
        done = self._check_goal()
        
        if done:
            if self._is_goal_scored('red') or self._is_goal_scored('blue'):
                print(f"Goal scored by team: {'red' if self._is_goal_scored('red') else 'blue'}")
                pygame.mixer.music.play()

                pass
        
        return self._get_obs(), reward, done, {}

    def _apply_action(self, action: int, team: str):
        """تحسين حركة اللاعبين بحيث اللاعب الأقرب للكرة يصبح أسرع"""
        
        base_speed = 1   
        max_speed = self.max_speed
        friction = 0.82  

        distance_red = np.hypot(self.ball.x - self.players['red'].x, self.ball.y - self.players['red'].y)
        distance_blue = np.hypot(self.ball.x - self.players['blue'].x, self.ball.y - self.players['blue'].y)

        if distance_red < distance_blue:
            red_speed_boost = max(3, 5 - (distance_red / 50))  
            blue_speed_boost = max(1, 3 - (distance_blue / 50))  
        else:
            blue_speed_boost = max(3, 5 - (distance_blue / 50))  
            red_speed_boost = max(1, 3 - (distance_red / 50))  

        speed_boost = red_speed_boost if team == 'red' else blue_speed_boost

        player_rect = self.players[team]
        dx_ball = self.ball.x - player_rect.x
        dy_ball = self.ball.y - player_rect.y
        distance_ball = np.hypot(dx_ball, dy_ball)

        if distance_ball > PLAYER_SIZE + BALL_SIZE:
            self.velocities[team][0] += speed_boost * np.sign(dx_ball)
            self.velocities[team][1] += speed_boost * np.sign(dy_ball)
        else:
            if action == 0: self.velocities[team][1] -= speed_boost  
            elif action == 1: self.velocities[team][1] += speed_boost  
            elif action == 2: self.velocities[team][0] -= speed_boost  
            elif action == 3: self.velocities[team][0] += speed_boost  
            elif action == 4: self._smart_kick(team)  

        self.velocities[team][0] = np.clip(self.velocities[team][0] * friction, -max_speed, max_speed)
        self.velocities[team][1] = np.clip(self.velocities[team][1] * friction, -max_speed, max_speed)

    def _smart_kick(self, team: str):
        """تسديد أو تمرير ذكي"""
        target_x = 0 if team == 'red' else WIDTH
        target_y = HEIGHT // 2 + np.random.uniform(-30, 30)   

        teammates = [p for t, p in self.players.items() if t != team]
        closest_teammate = min(teammates, key=lambda p: np.hypot(p.x - self.ball.x, p.y - self.ball.y))

        if np.hypot(closest_teammate.x - self.ball.x, closest_teammate.y - self.ball.y) < 150:
            dx = closest_teammate.x - self.ball.x
            dy = closest_teammate.y - self.ball.y
            distance = np.hypot(dx, dy)
            if distance != 0:  
                power = self.kick_power * (2.5 + (distance / WIDTH))
                self.velocities['ball'][0] = (dx / distance) * power
                self.velocities['ball'][1] = (dy / distance) * power
            else:
                self.velocities['ball'][0] = 0
                self.velocities['ball'][1] = 0
        else:
            dx = target_x - self.ball.x
            dy = target_y - self.ball.y
            distance = np.hypot(dx, dy)
            if distance != 0:   
                power = self.kick_power * (2.5 + (distance / WIDTH))
                self.velocities['ball'][0] = (dx / distance) * power
                self.velocities['ball'][1] = (dy / distance) * power
            else:
                self.velocities['ball'][0] = 0
                self.velocities['ball'][1] = 0
    

    def _update_physics(self):
        """محاكاة الفيزياء مع احتكاك واقعي"""
        for entity in ['red', 'blue', 'ball']:
            if entity in self.players:
                self.players[entity].x += self.velocities[entity][0]
                self.players[entity].y += self.velocities[entity][1]
            else:
                self.ball.x += self.velocities[entity][0]
                self.ball.y += self.velocities[entity][1]
            
            self.velocities[entity][0] *= self.friction
            self.velocities[entity][1] *= self.friction
            
            self._enforce_boundaries(entity)

    def _enforce_boundaries(self, entity: str):
        if entity in self.players:
            self.players[entity].x = np.clip(
                self.players[entity].x, 
                0, 
                WIDTH - PLAYER_SIZE
            )
            self.players[entity].y = np.clip(
                self.players[entity].y, 
                0, 
                HEIGHT - PLAYER_SIZE
            )
        else:
            self.ball.x = np.clip(
                self.ball.x, 
                0, 
                WIDTH - BALL_SIZE
            )
            self.ball.y = np.clip(
                self.ball.y, 
                0, 
                HEIGHT - BALL_SIZE
            )

    def _calculate_rewards(self):
        """نظام مكافآت متطور مع عقوبات صارمة"""
        rewards = {'red': 0.0, 'blue': 0.0}
        
        for team in ['red', 'blue']:
            target_x = 0 if team == 'blue' else WIDTH
            ball_distance = np.hypot(
                self.players[team].x - self.ball.x,
                self.players[team].y - self.ball.y
            )
            
            rewards[team] += 25.0 / (ball_distance + 1e-5)
            
            goal_progress = (abs(self.ball.x - target_x)/WIDTH)
            rewards[team] += (1 - goal_progress) * 15.0
            
            if self._is_goal_scored(team):
                rewards[team] += 1000.
                pygame.mixer.music.play()
                self.scores[team] += 1

                
            if np.linalg.norm(self.velocities[team]) < 1.5:
                rewards[team] -= 3.0
                
        return sum(rewards.values())

    def _is_goal_scored(self, team: str) -> bool:
        """التأكد من دخول الكرة في المرمى بدقة"""
        if team == 'red':
            if self.ball.x >= WIDTH - BALL_SIZE and \
            (HEIGHT // 2 - self.goal_size // 2 < self.ball.y < HEIGHT // 2 + self.goal_size // 2):
                if self.scores['red'] == 0:  
                    self.scores['red'] += 1   
                return True
        else:
            if self.ball.x <= BALL_SIZE and \
            (HEIGHT // 2 - self.goal_size // 2 < self.ball.y < HEIGHT // 2 + self.goal_size // 2):
                if self.scores['blue'] == 0:  
                    self.scores['blue'] += 1  
                return True
        return False



    def _check_goal(self) -> bool:
        return self._is_goal_scored('red') or self._is_goal_scored('blue')
    

    def _draw_player(self, x, y, team):
        """رسم لاعب كشخص بشري"""
        head_radius = 10
        body_width = 15
        body_height = 25
        leg_length = 15
        arm_length = 10
        color = TEAM_COLORS[0] if team == 'red' else TEAM_COLORS[1]

        # رسم الرأس
        pygame.draw.circle(self.screen, color, (x, y - body_height // 2), head_radius)

        # رسم الجسم
        pygame.draw.rect(self.screen, color, (x - body_width // 2, y, body_width, body_height))

        # رسم الأرجل
        pygame.draw.line(self.screen, color, (x - 5, y + body_height), (x - 5, y + body_height + leg_length), 3)
        pygame.draw.line(self.screen, color, (x + 5, y + body_height), (x + 5, y + body_height + leg_length), 3)

        # رسم الذراعين
        pygame.draw.line(self.screen, color, (x - body_width // 2, y + 5), (x - body_width // 2 - arm_length, y + 10), 2)
        pygame.draw.line(self.screen, color, (x + body_width // 2, y + 5), (x + body_width // 2 + arm_length, y + 10), 2)


    def render(self, mode='human'):
        """عرض النتائج والملعب"""
        pygame.event.pump()
        self.screen.fill((30, 150, 30))

         # رسم الملعب
        pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, WIDTH, HEIGHT), 5)
        pygame.draw.line(self.screen, (255, 255, 255), (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)  # خط المنتصف
        pygame.draw.circle(self.screen, (255, 255, 255), (WIDTH // 2, HEIGHT // 2), 50, 2)  # دائرة المنتصف

       
        self._draw_goal(0, HEIGHT // 2 - self.goal_size // 2, 20, self.goal_size)  # مرمى اليسار
        self._draw_goal(WIDTH - 20, HEIGHT // 2 - self.goal_size // 2, 20, self.goal_size)  # مرمى اليمين

        # رسم اللاعبين
        for team, rect in self.players.items():
            self._draw_player(rect.x + PLAYER_SIZE // 2, rect.y + PLAYER_SIZE // 2, team)

        
        # رسم الكرة
        self.screen.blit(self.ball_image, (self.ball.x, self.ball.y))


        # # عرض النتيجة
        self._update_match_time()   
        self._draw_scoreboard()    

        pygame.mixer.music.play()

        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_goal(self, x, y, width, height):
        """رسم المرمى بشبكة"""
        pygame.draw.rect(self.screen, (255, 255, 255), (x, y, width, height), 3)  # رسم الإطار

        # رسم الخطوط العمودية
        for i in range(1, 6):
            pygame.draw.line(self.screen, (255, 255, 255), (x + i * (width // 6), y), (x + i * (width // 6), y + height))

        # رسم الخطوط الأفقية
        for j in range(1, 6):
            pygame.draw.line(self.screen, (255, 255, 255), (x, y + j * (height // 6)), (x + width, y + j * (height // 6)))
    
    def _update_match_time(self):
        """تحديث المؤقت بحيث يحاكي 90 دقيقة خلال 3 دقائق"""
        elapsed_time = (pygame.time.get_ticks() - self.start_time) / 1000   
        
        if self.half == 2:
            elapsed_time += 45 * 60  
        
        self.match_time = int((elapsed_time / 180) * 90)  
        if self.match_time >= 45 and self.half == 1 and not self.show_half_time_notification:
            self.show_half_time_notification = True
            self.notification_start_time = pygame.time.get_ticks()   
            self.half = 2    
            
            half_time_text = self.font.render("End of First Half", True, (255, 255, 255))
            text_rect = half_time_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))   
            self.screen.blit(half_time_text, text_rect)  
            pygame.display.update()  

            pygame.time.delay(5000)  
            
            self.show_half_time_notification = False  

        if self.match_time >= 90:
            print(f"المباراة انتهت! النتيجة: {self.scores['red']} - {self.scores['blue']}")  # طباعة النتيجة
            self.running = False  # إيقاف المباراة
            pygame.quit()  # إغلاق نافذة اللعبة
            quit()  # إنهاء البرنامج بشكل كامل


    def _draw_scoreboard(self):
        """رسم واجهة النتيجة والتوقيت مع تقسيم المستطيل إلى أجزاء بألوان مختلفة"""
        scoreboard_width = 500  # عرض المستطيل
        scoreboard_height = 40  # تقليص ارتفاع المستطيل
        scoreboard_rect = pygame.Rect((WIDTH - scoreboard_width) // 2, 10, scoreboard_width, scoreboard_height)

        pygame.draw.rect(self.screen, (0, 0, 0), scoreboard_rect, border_radius=20)

        part_width = scoreboard_width // 4  # تحديد عرض كل جزء

        left_part_rect = pygame.Rect(scoreboard_rect.x, scoreboard_rect.y, part_width, scoreboard_height)
        middle_part_rect = pygame.Rect(scoreboard_rect.x + part_width, scoreboard_rect.y, part_width, scoreboard_height)
        right_part_rect = pygame.Rect(scoreboard_rect.x + 2 * part_width, scoreboard_rect.y, part_width, scoreboard_height)
        last_part_rect = pygame.Rect(scoreboard_rect.x + 3 * part_width, scoreboard_rect.y, part_width, scoreboard_height)

        pygame.draw.rect(self.screen, (255, 0, 0), left_part_rect)  # الجزء الأيسر باللون الأحمر
        pygame.draw.rect(self.screen, (0, 0, 0), middle_part_rect)  # الجزء الأوسط باللون الأسود
        pygame.draw.rect(self.screen, (0, 0, 255), right_part_rect)  # الجزء الثالث باللون الأزرق
        pygame.draw.rect(self.screen, (255, 255, 255), last_part_rect)  # الجزء الأخير باللون الأبيض

        left_team_text = self.font.render("Red", True, (255, 255, 255))  # الفريق الأحمر
        right_team_text = self.font.render("Blue", True, (255, 255, 255))  # الفريق الأزرق
        self.screen.blit(left_team_text, (left_part_rect.x + 20, left_part_rect.y + (scoreboard_height - left_team_text.get_height()) // 2))  # الفريق الأول
        self.screen.blit(right_team_text, (right_part_rect.x + 20, right_part_rect.y + (scoreboard_height - right_team_text.get_height()) // 2))  # الفريق الثاني

        score_text = self.font.render(f"{self.scores['blue']} - {self.scores['red']}", True, (255, 255, 255))
        self.screen.blit(score_text, (middle_part_rect.x + (middle_part_rect.width - score_text.get_width()) // 2, middle_part_rect.y + (scoreboard_height - score_text.get_height()) // 2))

        minutes = self.match_time
        seconds = int((pygame.time.get_ticks() / 1000) % 60)
        time_text = self.font.render(f"{minutes:02}:{seconds:02}", True, (0, 0, 0))  # التوقيت باللون الأسود داخل الجزء الأبيض
        self.screen.blit(time_text, (last_part_rect.x + (last_part_rect.width - time_text.get_width()) // 2, last_part_rect.y + (scoreboard_height - time_text.get_height()) // 2))

        if self.show_half_time_notification:
            notification_text = self.font.render(f"نهاية الشوط الأول: {self.scores['blue']} - {self.scores['red']}", True, (255, 255, 255))
            self.screen.blit(notification_text, ((WIDTH - notification_text.get_width()) // 2, HEIGHT // 2))

            if pygame.time.get_ticks() - self.notification_start_time >= 5000:  # 5 ثواني
                self.show_half_time_notification = False     
def train_ai():
    """تدريب النموذج مع إعدادات محسنة"""
    env = DummyVecEnv([lambda: AdvancedSoccerEnv()])
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=4096,
        batch_size=512,
        n_epochs=20,
        gamma=0.995,
        gae_lambda=0.92,
        ent_coef=0.05,
        policy_kwargs={
            'net_arch': [dict(pi=[512, 512], vf=[512, 512])]
        }
    )
    
    try:
        model.learn(total_timesteps=100_000)
        model.save("soccer_pro_ai_v3")
    except KeyboardInterrupt:
        model.save("soccer_pro_ai_interrupted")
    
    return model

def load_model():
    """تحميل النموذج المدرب مسبقًا"""
    env = DummyVecEnv([lambda: AdvancedSoccerEnv()])
    try:
        model = PPO.load("soccer_pro_ai_v3", env=env)
        print("تم تحميل النموذج المدرب بنجاح!")
        return model
    except FileNotFoundError:
        print("لم يتم العثور على نموذج، بدء التدريب...")

if __name__ == "__main__":
    model = load_model()
    
    # تشغيل البيئة بشكل مستمر
    env = AdvancedSoccerEnv()
    
    while True:  # لوب لا نهائي للحفاظ على النافذة مفتوحة
        obs = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)
            env.render()
            pygame.time.delay(50)
            
        
        # هنا بعد نهاية الحلقة يتم إعادة ضبط البيئة بدلاً من الخروج منها
        print("إعادة تشغيل البيئة بعد انتهاء الحلقة!")
