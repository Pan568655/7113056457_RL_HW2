from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(PROJECT_ROOT, 'static')
TEMPLATE_FOLDER = os.path.join(PROJECT_ROOT, 'templates')

os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(TEMPLATE_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder=TEMPLATE_FOLDER)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/', methods=['GET', 'POST'])
def index():
    n = 5
    if request.method == 'POST':
        try:
            n = int(request.form.get('n', 5))
            n = max(3, min(n, 10))
        except:
            pass
    return render_template('index.html', n=n)

@app.route('/api/generate', methods=['POST'])
def generate_policy_image():
    data = request.get_json()
    n = int(data['n'])
    start = tuple(data['start'])
    end = tuple(data['end'])
    walls = [tuple(w) for w in data['walls']]

    actions = [(-1,0), (0,1), (1,0), (0,-1)]
    gamma = 0.9
    value = np.zeros((n, n))
    policy = np.zeros((n, n), dtype=int)

    # Value Iteration to derive optimal policy
    for _ in range(200):
        delta = 0
        new_value = value.copy()
        for i in range(n):
            for j in range(n):
                if (i, j) == end or (i, j) in walls:
                    continue
                q_values = []
                for a, (dx, dy) in enumerate(actions):
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in walls:
                        r = -1
                    else:
                        ni, nj = i, j
                        r = -5
                    q = r + gamma * value[ni, nj]
                    q_values.append(q)
                best_q = max(q_values)
                best_a = np.argmax(q_values)
                new_value[i, j] = best_q
                policy[i, j] = best_a
                delta = max(delta, abs(best_q - value[i, j]))
        value = new_value
        if delta < 1e-3:
            break

    # Draw updated Value and Optimal Policy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    for ax in [ax1, ax2]:
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks(np.arange(n+1))
        ax.set_yticks(np.arange(n+1))
        ax.grid(True)

    ax1.set_title("Value Function V(s)")
    for i in range(n):
        for j in range(n):
            if (i, j) in walls:
                ax1.add_patch(plt.Rectangle((j, n-1-i), 1, 1, color='gray'))
            else:
                ax1.text(j + 0.5, n - 1 - i + 0.5, f"{value[i, j]:.2f}", ha='center', va='center')

    ax2.set_title("Optimal Policy")
    for i in range(n):
        for j in range(n):
            if (i, j) in walls:
                ax2.add_patch(plt.Rectangle((j, n-1-i), 1, 1, color='gray'))
            elif (i, j) == end:
                ax2.text(j + 0.5, n - 1 - i + 0.5, '★', ha='center', va='center', fontsize=16, color='red')
            else:
                a = policy[i, j]
                dx, dy = actions[a][1], -actions[a][0]
                ax2.arrow(j + 0.5, n - 1 - i + 0.5, 0.3 * dx, 0.3 * dy,
                          head_width=0.15, head_length=0.15, fc='black', ec='black')

    plt.tight_layout()
    ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
    image_filename = f'result_{ts}.png'
    image_path = os.path.join(STATIC_FOLDER, image_filename)
    plt.savefig(image_path)
    plt.close()

    image_url = f"/static/{image_filename}"
    return jsonify({"image_url": image_url})

if __name__ == '__main__':
    app.run(debug=True)
