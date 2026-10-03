import pygame
import pickle
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

# データ全体から上から順に200個のニューロンを使用する
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
INPUT_DIM = 6 # ハエには未来を教えず、通常の6つの情報のみ入力

def create_random_brain():
    return {
        'input': np.random.randn(INPUT_DIM, num_neurons) * 0.5,
        'output': np.random.randn(num_neurons, 3) * 0.5,
        'score': 100
    }

POPULATION_SIZE = 15
population = [create_random_brain() for _ in range(POPULATION_SIZE)]
current_fly_index = 0
generation = 1

save_filename = "best_fly.pkl"

# 💡 保持しているスコアとファイルのスコアを比べて、高ければセーブする関数
def check_and_save_best(fly, filename=save_filename):
    best_score = -1
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                saved_fly = pickle.load(f)
                best_score = saved_fly.get('score', 0)
        except Exception:
            pass
    
    # 記録更新時、またはファイルがない場合にセーブ
    if fly['score'] > best_score:
        with open(filename, "wb") as f:
            pickle.dump(fly, f)
        print(f"🌟 新記録！ スコアが更新されたため、{filename} に保存しました！ スコア: {int(fly['score'])}")

# 手動セーブ用関数
def save_best_fly(pop, filename=save_filename):
    pop.sort(key=lambda x: x['score'], reverse=True)
    best_fly = pop[0]
    with open(filename, "wb") as f:
        pickle.dump(best_fly, f)
    print(f"💾 最優秀のハエを {filename} に保存しました！ スコア: {int(best_fly['score'])}")

# 起動時にファイルがあればロードして最初の個体に引き継ぐ
if os.path.exists(save_filename):
    print(f"📂 保存されたハエのデータ ({save_filename}) を見つけました。ロードします。")
    with open(save_filename, "rb") as f:
        population[0] = pickle.load(f)
    print("✨ 前回の最強のハエの引き継ぎが完了しました！")

def mutate(parent_brain):
    child = create_random_brain()
    child['input'] = parent_brain['input'].copy()
    child['output'] = parent_brain['output'].copy()
    child['score'] = 100
    child['input'] += (np.random.randn(INPUT_DIM, num_neurons) * 0.25)
    child['output'] += (np.random.randn(num_neurons, 3) * 0.25)
    return child

# --- 3. ゲーム本体（Pygame）の立ち上げ ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ハエの脳進化ピンポン - 自動ハイスコアセーブ付き")
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
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s: # 'S'キーでセーブ
                save_best_fly(population)
            elif event.key == pygame.K_l: # 'L'キーでロード
                if os.path.exists(save_filename):
                    with open(save_filename, "rb") as f:
                        population[0] = pickle.load(f)
                    print("📂 ハエのデータをロードしました！")

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

    # ハエへの入力
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

    # アクション実行 (0:上移動, 1:静止, 2:下移動)
    if action == 0 and game['fly_y'] > 0:
        game['fly_y'] -= 6
    elif action == 2 and game['fly_y'] < HEIGHT - 80:
        game['fly_y'] += 6

    new_fly_center = game['fly_y'] + 40
    new_distance = abs(new_fly_center - game['ball_y'])

    # --- 3. ボールの移動と衝突判定 ---
    game['ball_x'] += game['ball_dx']
    game['ball_y'] += game['ball_dy']

    if game['ball_y'] <= 0 or game['ball_y'] >= HEIGHT - 15:
        game['ball_dy'] *= -1

    if game['ball_x'] <= 30 and game['player_y'] <= game['ball_y'] <= game['player_y'] + 80:
        game['ball_dx'] *= -1.05
        game['ball_x'] = 31

    # 落下位置予測とボーナス／ペナルティの判定
    if game['ball_dx'] > 0:
        sim_x = game['ball_x']
        sim_y = game['ball_y']
        sim_dy = game['ball_dy']
        sim_dx = game['ball_dx']
        
        while sim_x < WIDTH - 30 and sim_dx > 0:
            sim_x += sim_dx
            sim_y += sim_dy
            if sim_y <= 0 or sim_y >= HEIGHT - 15:
                sim_dy *= -1
        target_y = sim_y

        dist_to_target = abs(fly_center - target_y)

        if dist_to_target <= 40:
            if action == 1:
                current_fly['score'] += 35 
            else:
                current_fly['score'] += 5
        else:
            if action == 1:
                current_fly['score'] -= 15 # ボールが来ているのに動かないときの減点
    
    # 通常の近接報酬
    if new_distance < old_distance:
        current_fly['score'] += 10
    else:
        current_fly['score'] = max(10, current_fly['score'] - 2)

    if WIDTH - 45 <= game['ball_x'] <= WIDTH - 15:
        current_fly['score'] += 15

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
        
        # 💡 ここで「今回獲得したスコア」と「ファイル保存されているスコア」を比較してセーブ！
        check_and_save_best(current_fly)
        
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