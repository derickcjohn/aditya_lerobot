from flask import Flask, request, jsonify, render_template
import threading
from llm_inference import main as robot_main, get_command, is_exit_command

app = Flask(__name__)

# Thread-safe way to interact with the robot
command_queue = []
response_queue = []

@app.route('/')
def home():
    return render_template('robot_chat.html')

@app.route('/api/robot-chat', methods=['POST'])
def robot_chat():
    data = request.get_json()
    command = data.get('command', '')
    if not command:
        return jsonify({'reply': 'No command provided.'}), 400

    # Add the command to the shared queue
    command_queue.append(command)

    # Wait for a response
    while not response_queue:
        pass

    reply = response_queue.pop(0)
    return jsonify({'reply': reply})

# Modify get_command in llm_inference.py to use this wrapper

def get_command_override(use_typing):
    while not command_queue:
        pass
    return command_queue.pop(0)

def reply_to_web(text):
    response_queue.append(text)

# Override methods for web interface
import llm_inference
llm_inference.get_command = get_command_override
llm_inference.say = reply_to_web

# Start robot logic in separate thread
robot_thread = threading.Thread(target=robot_main)
robot_thread.daemon = True
robot_thread.start()

if __name__ == '__main__':
    app.run(debug=True)
