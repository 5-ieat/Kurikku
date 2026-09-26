import pygame
import pandas as pd
import numpy as np
import os
import sys

# 画面サイズと色の設定
WIDTH, HEIGHT = 800, 500
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 150, 255)
RED = (255, 99, 71)

# --- 1. ハエのデータの読み込みと軽量化 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weight_file = os.path.join(BASE_DIR, "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
anno_file = os.path.join(BASE_DIR, "body-annotations-male-cns-v1.0-minconf-0.5.feather")

if not os.path.exists(weight_file) or not os.path.exists(anno_file):
    print("データファイルが見つかりません。同じフォルダに置いてください。")
    sys.exit()

print("ハエのデータを読み込んでいます...（数十秒かかります）")
df_w = pd.read_feather(weight_file)
df_a = pd.read_feather(anno_file)

# 列名の自動判別
pre_col = [c for c in df_w.columns if 'pre' in c.lower()][0]   
post_col = [c for c in df_w.columns if 'post' in c.lower()][0] 
weight_col = [c for c in df_w.columns if 'weight' in c.lower() or 'score' in c.lower()][0]
id_col = [c for c in df_a.columns if 'id' in c.lower()][0]    

# データ全体から上から順に500個のニューロンを使用する（CXなし）
all_neurons = df_a[id_col].unique()
unique_neurons = all_neurons[:200]
neuron_map = {nid: i for i, nid in enumerate(unique_neurons)}
num_neurons = len(unique_neurons)

print(f"使用ニューロン数: {num_neurons}個（データ全体から抽出）")

# 選択したニューロン間のシナプス結合を抽出して脳のマトリクスを作成
df_cx = df_w[df_w[pre_col].isin(unique_neurons) & df_w[post_col].isin(unique_neurons)]

base_brain = np.zeros((num_neurons, num_neurons))
for _, row in df_cx.head(20000).iterrows(): 
    pre_id = row[pre_col]
    post_id = row[post_col]
    if pre_id in neuron_map and post_id in neuron_map:
        base_brain[neuron_map[pre_id], neuron_map[post_id]] = row[weight_col]

print("ハエの脳の組み立てが完了しました！")

# --- 2. 遺伝的アルゴリズム（進化の仕組み）の準備 ---
def create_random_brain():
    return {
        'input': np.random.randn(6, num_neurons) * 0.5,
        'output': np.random.randn(num_neurons, 3) * 0.5,
        'score': 100
    }

POPULATION_SIZE = 15
population = [create_random_brain() for _ in range(POPULATION_SIZE)]
current_fly_index = 0
generation = 1

def mutate(parent_brain):
    child = create_random_brain()
    child['input'] = parent_brain['input'].copy()
    child['output'] = parent_brain['output'].copy()
    child['score'] = 100
    child['input'] += (np.random.randn(6, num_neurons) * 0.25)
    child['output'] += (np.random.randn(num_neurons, 3) * 0.25)
    return child

# --- 3. ゲーム本体（Pygame）の立ち上げ ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ハエの脳進化ピンポン")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

def reset_game():
    return {
        'player_y': HEIGHT // 2 - 40,
        'fly_y': np.random.randint(50, HEIGHT - 130), 
        'ball_x': WIDTH // 2,
        'ball_y': HEIGHT // 2,
        'ball_dx': -6, 
        'ball_dy': np.random.uniform(-3, 3),
        'game_steps': 0,
        'rally_count': 1,
        'last_action': 1 
    }

game = reset_game()
last_outputs = np.zeros(3)

running = True
while running:
    screen.fill(BLACK)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 1. 人間の操作（W/Sキー）
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] and game['player_y'] > 0:
        game['player_y'] -= 6
    if keys[pygame.K_s] and game['player_y'] < HEIGHT - 80:
        game['player_y'] += 6

    # 2. 現在テスト中のハエAIの操作
    current_fly = population[current_fly_index]
    game['game_steps'] += 1
    
    fly_center = game['fly_y'] + 40
    
    norm_ball_x = (game['ball_x'] - (WIDTH / 2)) / (WIDTH / 2)
    norm_ball_y = (game['ball_y'] - (HEIGHT / 2)) / (HEIGHT / 2)
    norm_fly_y = (fly_center - (HEIGHT / 2)) / (HEIGHT / 2)
    norm_ball_dx = game['ball_dx'] / 10.0 
    norm_ball_dy = game['ball_dy'] / 5.0
    distance_y = game['ball_y'] - fly_center
    norm_diff_y = distance_y / HEIGHT

    inputs = np.array([
        norm_ball_x, 
        norm_ball_y, 
        norm_fly_y, 
        norm_ball_dx, 
        norm_ball_dy, 
        norm_diff_y
    ])
    
    stimulus = np.dot(inputs, current_fly['input']) + 0.1
    stimulus = np.maximum(0.01 * stimulus, stimulus) 
    
    brain_network = base_brain + np.eye(num_neurons) * 0.5
    brain_activity = np.dot(stimulus, brain_network)
    brain_activity = np.maximum(0.01 * brain_activity, brain_activity)
    
    outputs = np.dot(brain_activity, current_fly['output'])

    smooth_outputs = (outputs * 0.4) + (last_outputs * 0.6)
    last_outputs = smooth_outputs.copy() 
    
    # 壁ハメ防止マスク
    if game['fly_y'] <= 10:
        smooth_outputs[0] -= 999.0  
    elif game['fly_y'] >= HEIGHT - 90:
        smooth_outputs[2] -= 999.0  

    exp_outputs = np.exp(smooth_outputs - np.max(smooth_outputs))
    probabilities = exp_outputs / np.sum(exp_outputs)
    
    exploration_rate = max(0.10, 1.0 - (generation * 0.05))
    if np.random.rand() < exploration_rate:
        action = np.random.choice([0, 1, 2], p=probabilities)
    else:
        action = np.argmax(smooth_outputs)

    game['last_action'] = action

    old_distance = abs(fly_center - game['ball_y'])

    if action == 0 and game['fly_y'] > 0:
        game['fly_y'] -= 6
    elif action == 2 and game['fly_y'] < HEIGHT - 80:
        game['fly_y'] += 6

    new_fly_center = game['fly_y'] + 40
    new_distance = abs(new_fly_center - game['ball_y'])

    # シンプルでクリーンな追尾評価（近づけばプラス、遠ざかったら小マイナス）
    if new_distance < old_distance:
        current_fly['score'] += 20
    else:
        current_fly['score'] = max(10, current_fly['score'] - 5)

    # 3. ボールの移動と衝突判定
    game['ball_x'] += game['ball_dx']
    game['ball_y'] += game['ball_dy']

    if game['ball_y'] <= 0 or game['ball_y'] >= HEIGHT - 15:
        game['ball_dy'] *= -1

    if game['ball_x'] <= 30 and game['player_y'] <= game['ball_y'] <= game['player_y'] + 80:
        game['ball_dx'] *= -1.05
        game['ball_x'] = 31

    # ハエが打ち返した時（大ボーナス）
    if game['ball_x'] >= WIDTH - 45 and game['fly_y'] <= game['ball_y'] <= game['fly_y'] + 80:
        game['ball_dx'] *= -1.05
        game['ball_x'] = WIDTH - 46
        game['rally_count'] += 1 
        
        hit_reward = 1000 * game['rally_count']
        current_fly['score'] += hit_reward
        print(f"🎉【大成功】ハエがボールを打ち返しました！ スコア +{hit_reward}")

    # --- 4. 世代交代のチェック ---
    if game['ball_x'] < 0:
        game = reset_game()

    if game['ball_x'] > WIDTH:
        miss_distance = abs(fly_center - game['ball_y'])
        miss_penalty = int(miss_distance * 2) + 200
        current_fly['score'] = max(0, current_fly['score'] - miss_penalty)
        print(f"❌【ミス】最終ズレ: {int(miss_distance)}px, ペナルティ: -{miss_penalty}")
        
        current_fly_index += 1
        game = reset_game()
        
        if current_fly_index >= POPULATION_SIZE:
            population.sort(key=lambda x: x['score'], reverse=True)
            print(f"--- 世代 {generation} 終了！ トップスコア: {int(population[0]['score'])} ---")
            
            new_population = []
            for i in range(2):
                elite = {
                    'input': population[i]['input'].copy(),
                    'output': population[i]['output'].copy(),
                    'score': 100
                }
                new_population.append(elite)
            
            for _ in range(POPULATION_SIZE - 2):
                parent = population[0] if np.random.rand() > 0.3 else population[1]
                new_population.append(mutate(parent))
                
            population = new_population
            current_fly_index = 0
            generation += 1

    # --- 5. 画面の描画 ---
    pygame.draw.rect(screen, BLUE, (15, game['player_y'], 15, 80)) # 人間
    pygame.draw.rect(screen, RED, (WIDTH - 30, game['fly_y'], 15, 80)) # ハエ
    pygame.draw.circle(screen, WHITE, (int(game['ball_x']), int(game['ball_y'])), 8) # ボール
    
    text_gen = font.render(f"Gen: {generation}   Fly: {current_fly_index+1}/{POPULATION_SIZE}", True, WHITE)
    text_score = font.render(f"Fly Score: {int(current_fly['score'])}   Rally: {game['rally_count']-1}", True, WHITE)
    screen.blit(text_gen, (20, 20))
    screen.blit(text_score, (20, 50))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()