import os
import json
import numpy as np
import pandas as pd
import random
import argparse
import time
from scipy.special import softmax
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    raise ValueError("[Initialization Error] DEEPSEEK_API_KEY is missing. Please verify your .env file.")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

BEHAVIORS = ['Talk', 'Read', 'Phone', 'Hand', 'Lean', 'Stand']
NUM_VARS = len(BEHAVIORS)
BEH_IDX = {b: i for i, b in enumerate(BEHAVIORS)}

# [GIỮ NGUYÊN BẢN CHUẨN] Sức mạnh vừa đủ để nằm trong vùng tuyến tính của Softmax
STRENGTH_MAP = {
    'high': 2.0,   
    'medium': 1.2,
    'low': 0.6    
}

# ==========================================
# 1. AUTOMATED TEST DIRECTORY MANAGEMENT
# ==========================================
def get_next_test_folder(base_dir="synth_data"):
    os.makedirs(base_dir, exist_ok=True)
    existing_tests = [d for d in os.listdir(base_dir) if d.startswith("test_")]
    if not existing_tests:
        next_idx = 1
    else:
        indices = [int(d.split("_")[1]) for d in existing_tests if d.split("_")[1].isdigit()]
        next_idx = max(indices) + 1 if indices else 1
        
    new_dir = os.path.join(base_dir, f"test_{next_idx:02d}")
    os.makedirs(new_dir, exist_ok=True) 
    return new_dir

def generate_dynamic_context():
    times = ["early morning", "mid-morning", "post-lunch food coma", "late afternoon"]
    subjects = ["Calculus lecture", "boring History reading", "interactive discussion"]
    events = ["stressful exam coming up", "teacher is strict", "teacher has back turned"]
    return f"Time: {random.choice(times)}. Subject: {random.choice(subjects)}. Event: {random.choice(events)}."

# ==========================================
# 2. LLM GRAPH GENERATION WITH STRICT PROMPT & RETRY LOGIC
# ==========================================
def generate_valid_graph_from_llm(min_edges=7, max_edges=15):
    context = generate_dynamic_context()
    
    prompt = f"""You are an Expert Data Scientist modeling a classroom causal graph (DAG) for STUDENT behaviors.
Context: {context}

AVAILABLE BEHAVIORS AND THEIR STRICT DEFINITIONS:
- 'Talk': A student talking to peers.
- 'Read': A student reading the textbook.
- 'Phone': A student secretly using a smartphone.
- 'Hand': A student RAISING THEIR HAND to ask a question (DO NOT interpret this as holding an object).
- 'Lean': A student slouching/leaning on the desk due to fatigue/boredom.
- 'Stand': A student standing up from their chair.

STRICT CAUSAL RULES (CRITICAL):
1. Output MUST be a STRICT Directed Acyclic Graph (DAG). NO feedback loops.
2. NO TRANSITIVE SHORTCUTS. 
3. CAUSALITY, NOT CORRELATION. 
4. DO NOT create self-loops (A->A).
5. All behaviors apply to the STUDENTS, not the teacher.
6. Output format: JSON object with a single key "edges" containing a list of edges.
Each edge MUST have: "source", "target", "type" ('positive' or 'negative'), "strength" ('medium', 'high'), and "reasoning".
7. THINK IN PEER INFLUENCE & CONTAGION, NOT INDIVIDUAL SEQUENCES.
   - WRONG: "Raising hand causes standing." (Individual action sequence).
   - RIGHT: "Widespread talking creates a noisy environment, causing others to stop reading." (Macro contagion).

Generate EXACTLY 4 to 6 highly logical, unambiguous causal edges. Keep the graph SPARSE.
"""

    attempt = 1
    while True:
        try:
            if attempt > 1: print(f"      [LLM] Retrying generation (Attempt {attempt})...")
            
            # 1. Gọi DeepSeek sinh data
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a strict DAG generator. Return only valid JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}, 
                temperature=0.4, 
                max_tokens=2000 
            )
            raw_content = response.choices[0].message.content.strip()
            start_idx = raw_content.find('{')
            end_idx = raw_content.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                raise ValueError("JSON Format broken.")
                
            parsed_data = json.loads(raw_content[start_idx:end_idx+1])
            raw_edges = parsed_data.get("edges", [])
            
            # 2. GỌI BẢO KÊ TOÁN HỌC NGAY TẠI ĐÂY
            sanitized_edges = sanitize_llm_graph(raw_edges)
            
            # 3. KIỂM TRA MẬT ĐỘ ĐỒ THỊ (Chống bệnh cắt ác quá)
            if len(sanitized_edges) < min_edges:
                print(f"      [POLICY] Graph too sparse after sanitize (Only {len(sanitized_edges)} edges). Rejecting and Retrying...")
                attempt += 1
                continue # Vứt mẹ đi, quay lại vòng lặp bắt LLM đẻ cái khác
                
            # Đạt chuẩn cả về Toán học lẫn Mật độ
            return sanitized_edges, context
            
        except Exception as e:
            print(f"      [API Error] {e}. Resting 3s...")
            time.sleep(3)
            attempt += 1

# ==========================================
# 3. MATHEMATICAL BOUNCER (SANITIZE POLICY - ĐÃ THÊM LỚP 2)
# ==========================================
def has_cycle(graph_dict):
    """Phát hiện chu trình bằng Depth-First Search (DFS)"""
    visited = set()
    rec_stack = set()
    
    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph_dict.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor): return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False

    for node in graph_dict:
        if node not in visited:
            if dfs(node): return True
        visited.clear() 
    return False

def sanitize_llm_graph(raw_edges):
    valid_edges = []
    seen_pairs = set()
    
    # Bước 1: Lọc ảo giác, tự tham chiếu và cạnh hai chiều (Feedback loops)
    for e in raw_edges:
        src, tgt = e.get("source"), e.get("target")
        if src not in BEHAVIORS or tgt not in BEHAVIORS: continue
        if src == tgt: continue 
        
        # Cấm lật lọng (A->B rồi lại B->A)
        if (src, tgt) in seen_pairs or (tgt, src) in seen_pairs:
            continue
            
        seen_pairs.add((src, tgt))
        valid_edges.append(e)
        
    # Bước 2: Chặt Chu Trình (DAG Check)
    final_edges = []
    adj_list = {b: [] for b in BEHAVIORS}
    
    for e in valid_edges:
        src, tgt = e["source"], e["target"]
        adj_list[src].append(tgt)
        
        if has_cycle(adj_list):
            adj_list[src].remove(tgt) 
            print(f"      [POLICY] Detected & Removed Cycle-inducing edge: {src} -> {tgt}")
        else:
            final_edges.append(e)
            
    return final_edges

# ==========================================
# 4. VAR MATRIX PARSER (GIỮ NGUYÊN)
# ==========================================
def build_matrices(edges_json):
    W = np.zeros((NUM_VARS, NUM_VARS))
    A = np.zeros((NUM_VARS, NUM_VARS), dtype=int)
    for edge in edges_json:
        src, tgt = edge.get("source"), e.get("target") if 'e' in locals() else edge.get("target") # Fallback safety
        e_type, strength = edge.get("type"), edge.get("strength", "low")
        
        if src not in BEH_IDX or tgt not in BEH_IDX or src == tgt: continue
            
        i, j = BEH_IDX[src], BEH_IDX[tgt]
        val = STRENGTH_MAP.get(strength, 0.5)
        
        if e_type == "negative": W[i, j] = -val
        elif e_type == "positive": W[i, j] = val
        A[i, j] = 1
    return W, A

# ==========================================
# 5. CORE MATHEMATICAL ENGINE: PERFECT VAR(1) ABM (GIỮ NGUYÊN)
# ==========================================
def run_simulation(W, num_steps=1000, num_students=100):
    T = 1.0 
    sim_vars = NUM_VARS + 1 
    
    W_sim = np.zeros((sim_vars, sim_vars))
    W_sim[:NUM_VARS, :NUM_VARS] = W # Bỏ fill_diagonal ở đây đi! Quán tính macro là sai lầm!
    
    baseline = np.zeros(sim_vars)
    baseline[NUM_VARS] = 1.0 # Lực hút Idle cơ bản
    
    S = np.random.randint(0, sim_vars, size=num_students)
    X_observed = np.zeros((num_steps, NUM_VARS)) 
    
    for t in range(1, num_steps):
        # 1. Lực lây lan từ tập thể (Peer Influence)
        current_pct = np.bincount(S, minlength=sim_vars) / num_students
        peer_influence = np.dot(current_pct, W_sim)
        
        # 2. [LÕI TOÁN HỌC MỚI] Quán tính cá nhân (Individual Inertia)
        # Sinh viên có xu hướng rất mạnh để GIỮ NGUYÊN trạng thái hiện tại.
        S_onehot = np.eye(sim_vars)[S]
        individual_inertia = S_onehot * 3.5 # Trọng số quán tính cực mạnh
        
        # 3. Tổng hợp động lực cho từng cá nhân
        M_individuals = individual_inertia + peer_influence + baseline + np.random.normal(0, 0.3, size=(num_students, sim_vars))
        
        P = softmax(M_individuals / T, axis=1)
        cum_P = np.cumsum(P, axis=1)
        random_draws = np.random.rand(num_students, 1)
        S = np.argmax(random_draws < cum_P, axis=1)
        
        counts = np.bincount(S, minlength=sim_vars)
        X_observed[t] = (counts[:NUM_VARS] / num_students) * 100.0

    # Nhiễu Camera cực thấp để bảo vệ tín hiệu
    eta = np.random.uniform(0.99, 1.0, size=(num_steps, 1)) 
    X_noisy = (X_observed * eta) + np.random.normal(0, 0.1, size=(num_steps, NUM_VARS))
    
    return pd.DataFrame(np.clip(X_noisy, 0, 100.0), columns=BEHAVIORS)

# ==========================================
# 6. TEST SUITE ORCHESTRATION (TÍCH HỢP ĐẦY ĐỦ)
# ==========================================
# ==========================================
# 6. TEST SUITE ORCHESTRATION
# ==========================================
def generate_test_suite(num_tests=5):
    print(f"[SYSTEM] Generating {num_tests} Causal Test Cases with Policy Bouncer...")
    for i in range(num_tests):
        folder = get_next_test_folder()
        try:
            # GỌI HÀM MỚI (Đã bao gồm cả gọi LLM và Sanitize bên trong)
            sanitized_edges, context = generate_valid_graph_from_llm(min_edges=4, max_edges=10)
            
            W, A = build_matrices(sanitized_edges)
            df = run_simulation(W, num_steps=1000, num_students=100)
            
            graph_data = {
                "scenario_context": context,
                "behaviors": BEHAVIORS,
                "edges_logic_from_llm": sanitized_edges, 
                "matrices": {
                    "W_GroundTruth_Weights": W.tolist(),
                    "A_GroundTruth_Adjacency": A.tolist()
                }
            }
            
            with open(os.path.join(folder, "ground_truth_graph.json"), "w", encoding="utf-8") as f:
                json.dump(graph_data, f, indent=4)
                
            df.to_csv(os.path.join(folder, "time_series_noisy.csv"), index=False)
            print(f"   -> [SUCCESS] Test {folder} generated. Retained {len(sanitized_edges)} valid edges.")
        except Exception as e:
            print(f"   -> [FATAL ERROR] Pipeline crashed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=5)
    args = parser.parse_args()
    generate_test_suite(num_tests=args.num)
