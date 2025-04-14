from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import subprocess, os, sys, signal
from pynput.keyboard import Controller, Key
from datetime import datetime
import time
import atexit

app = Flask(__name__)

process = None  # control script process
train_process = None  # training script process

STATUS_FILE = "robot_status.txt"

# Edit this path as needed
MODEL_ROOT = "outputs/train_single"

def set_status(new_status: str):
    with open(STATUS_FILE, "w") as f:
        f.write(new_status)

def get_status() -> str:
    if not os.path.exists(STATUS_FILE):
        return "idle! sleeping"
    with open(STATUS_FILE, "r") as f:
        status = f.read()
    return status

set_status("")

def cleanup():
    if os.path.exists(STATUS_FILE):
        os.remove(STATUS_FILE)
        print("[Cleanup]")

atexit.register(cleanup)

# ---------- Control Defaults ----------
DEFAULT_ARGS = {
    "record": {
        "robot-path": "lerobot/configs/robot/revobots.yaml",
        "fps": 30,
        "root": "demo_data",
        "repo-id": "demo_dataset",
        "tags": "tutorial",
        "warmup-time-s": 300,
        "episode-time-s": 300,
        "reset-time-s": 1,
        "num-episodes": 2,
        "push-to-hub": 0
    },
    "record-with-marker": {
        "robot-path": "lerobot/configs/robot/revobots.yaml",
        "fps": 30,
        "root": "demo_data",
        "repo-id": "demo_dataset",
        "tags": "tutorial",
        "warmup-time-s": 300,
        "episode-time-s": 300,
        "reset-time-s": 1,
        "num-episodes": 2,
        "push-to-hub": 0
    },
    "replay": {
        "robot-path": "lerobot/configs/robot/revobots.yaml",
        "fps": 30,
        "root": "demo_data",
        "repo-id": "revobots_test",
        "episode": 0
    },
    "teleoperate": {
        "robot-path": "lerobot/configs/robot/revobots.yaml"
    }
}

EDITABLE_KEYS = {
    "record": ["repo-id", "num-episodes"],
    "record-with-marker": ["repo-id", "num-episodes"],
    "replay": ["repo-id", "episode"],
    "teleoperate": []
}

# ---------- Training Defaults ----------
TRAIN_DEFAULT_ARGS = {
    "dataset_repo_id": "demo_dataset",
    "policy": "act_koch_real_added_camera",
    "env": "koch_real",
    "hydra.run.dir": "demo_outputs/train/act_demo_dataset",
    "hydra.job.name": "act_demo_dataset",
    "device": "cuda",
    "wandb.enable": "false"
}

TRAIN_EDITABLE_KEYS = ["dataset_repo_id", "device"]

# ---------- Routes ----------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/inference')
def inference():
    folders = []
    for entry in os.scandir(MODEL_ROOT):
        if entry.is_dir():
            mtime = entry.stat().st_mtime
            folders.append((entry.name, mtime))
    sorted_folders = [name for name, _ in sorted(folders, key=lambda x: x[1], reverse=True)]
    return render_template("inference.html", model_folders=sorted_folders)

@app.route("/run_inference", methods=["POST"])
def run_inference():
    data = request.get_json()
    model_name = data.get("model")
    if not model_name:
        return jsonify({"error": "No model selected"}), 400

    model_path = os.path.join(MODEL_ROOT, model_name, "checkpoints/last/pretrained_model")
    print(model_path)

    # TODO: Replace the next line with your actual command
    command = ["python3", "inference.py", "--model", model_path]

    try:
        # Example: Run command asynchronously
        subprocess.Popen(command)
        return jsonify({"output": f"Inference started for model: {model_name}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/list_datasets', methods=['GET'])
def list_datasets():
    dataset_dir = "demo_data"
    try:
        folders = [
            name for name in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, name))
        ]
        return jsonify({"folders": folders})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_args', methods=['POST'])
def get_args():
    mode = request.json.get("mode")
    args = DEFAULT_ARGS.get(mode, {})
    editable_keys = EDITABLE_KEYS.get(mode, [])
    editable_args = {k: args[k] for k in editable_keys if k in args}
    return jsonify(editable_args)

@app.route('/run_script', methods=['POST'])
def run_script():
    global process
    data = request.json
    mode = data.get("mode")
    user_args = data.get("args", {})

    full_args = DEFAULT_ARGS.get(mode, {}).copy()

    if "repo-id" in EDITABLE_KEYS.get(mode, []) and "repo-id" in user_args:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        user_args["repo-id"] = f"{user_args['repo-id']}_{timestamp}"

    full_args.update(user_args)

    script_path = os.path.abspath("lerobot/scripts/control_robot.py")
    cmd = [sys.executable, script_path, mode.replace("-", "_")]
    for key, value in full_args.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    app.config["CURRENT_CMD"] = cmd

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".") + os.pathsep + env.get("PYTHONPATH", "")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            env=env
        )
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/stream_output')
def stream_output():
    global process
    if not process or process.poll() is not None:
        return "No running process", 400

    def generate():
        for line in iter(process.stdout.readline, ''):
            yield f"data: {line.rstrip()}\n\n"
        process.stdout.close()
        process.wait()

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/send_key', methods=['POST'])
def send_key():
    keyboard = Controller()
    key = request.json.get("key")
    key_map = {
        "ArrowLeft": Key.left,
        "ArrowRight": Key.right,
        "Escape": Key.esc
    }
    if key in key_map:
        keyboard.press(key_map[key])
        keyboard.release(key_map[key])
        return jsonify({"status": "success", "key": key})
    else:
        return jsonify({"status": "error", "message": "Invalid key"}), 400

@app.route('/stop_script', methods=['POST'])
def stop_script():
    global process
    if process and process.poll() is None:
        os.kill(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        return jsonify({"status": "terminated"})
    return jsonify({"status": "no_process"})

@app.route('/stream_status')
def stream_status():
    def generate():
        while True:
            current_step = get_status()
            yield f"data: {current_step}\n\n"
            time.sleep(0.2)
    return Response(generate(), mimetype='text/event-stream')

@app.route('/get_train_args', methods=['POST'])
def get_train_args():
    editable_args = {k: TRAIN_DEFAULT_ARGS[k] for k in TRAIN_EDITABLE_KEYS}
    return jsonify(editable_args)

@app.route('/run_train', methods=['POST'])
def run_train():
    global train_process
    user_args = request.json.get("args", {})
    train_type = request.json.get("train_type", "train")
    full_args = TRAIN_DEFAULT_ARGS.copy()
    full_args.update(user_args)

    if "dataset_repo_id" in user_args:
        dataset_name = user_args["dataset_repo_id"]
        full_args["hydra.run.dir"] = f"outputs/{train_type}/act_{dataset_name}"
        full_args["hydra.job.name"] = f"act_{dataset_name}"

    script_path = os.path.abspath("lerobot/scripts/train.py")
    cmd = ["DATA_DIR=demo_data", sys.executable, script_path]
    for key, value in full_args.items():
        cmd.append(f"{key}={value}")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".") + os.pathsep + env.get("PYTHONPATH", "")

    try:
        train_process = subprocess.Popen(
            " ".join(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            shell=True,
            env=env
        )
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/stream_train_output')
def stream_train_output():
    global train_process
    if not train_process or train_process.poll() is not None:
        return "No training process", 400

    def generate():
        for line in iter(train_process.stdout.readline, ''):
            yield f"data: {line.rstrip()}\n\n"
        train_process.stdout.close()
        train_process.wait()

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

# ---------- Main ----------
if __name__ == '__main__':
    app.run(debug=True)
