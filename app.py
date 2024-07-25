# """TEXT-TO-SPEECH"""
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# from transformers import pipeline
# import soundfile as sf
# import os
# import time

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Initialize the text-to-speech pipeline
# narrator = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")


# @app.route('/synthesize', methods=['POST'])
# def synthesize_speech():
#     try:
#         # Get the text input from the request
#         data = request.json
#         text = data.get('text', '')

#         if not text:
#             return jsonify({"error": "Text is required"}), 400

#         # Generate speech from the text
#         narrated_txt = narrator(text)

#         # Create a unique filename
#         timestamp = int(time.time())
#         audio_file = f'output_{timestamp}.wav'

#         # Save audio to a file
#         sf.write(audio_file, narrated_txt["audio"]
#                  [0], narrated_txt["sampling_rate"])

#         # Return the audio file and text response
#         return jsonify({
#             "response": text,
#             "audio_file": f"/audio/{audio_file}"
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400


# @app.route('/audio/<filename>', methods=['GET'])
# def get_audio(filename):
#     # Ensure the file exists
#     if not os.path.exists(filename):
#         return jsonify({"error": "File not found"}), 404

#     return send_file(filename, mimetype='audio/wav')


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=8080)





# """SPEECH TO TEXT"""
# from flask import Flask, request, jsonify
# from transformers import pipeline
# from pydub import AudioSegment
# import numpy as np
# import io

# app = Flask(__name__)

# # Load the pipeline with Whisper model
# pipe = pipeline(
#     "automatic-speech-recognition",
#     model="openai/whisper-tiny",
#     chunk_length_s=30
# )


# def audio_file_to_numpy(audio_file):
#     audio_data = audio_file.read()
#     audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))

#     # Convert audio segment to numpy array
#     samples = np.array(audio_segment.get_array_of_samples())
#     # Reshape to the expected format if stereo
#     if audio_segment.channels > 1:
#         samples = samples.reshape(-1, audio_segment.channels).mean(axis=1)
#     # Convert to float32 and normalize
#     samples = samples.astype(np.float32) / np.max(np.abs(samples))
#     return samples


# def transcribe_audio(audio_file):
#     audio_array = audio_file_to_numpy(audio_file)
#     # Whisper model expects numpy arrays as input
#     result = pipe(audio_array, return_timestamps=True)
#     return result


# @app.route("/transcribe", methods=["POST"])
# def transcribe():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     try:
#         transcription = transcribe_audio(file)
#         return jsonify({"transcription": transcription})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(debug=True, port=8080)



"""Speech to text and text to speech."""

# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# from transformers import pipeline
# from pydub import AudioSegment
# import numpy as np
# import io
# import soundfile as sf
# import os
# import time

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Initialize the pipelines
# speech_to_text_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", chunk_length_s=30)
# text_to_speech_pipe = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")

# def audio_file_to_numpy(audio_file):
#     audio_data = audio_file.read()
#     audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
    
#     # Convert audio segment to numpy array
#     samples = np.array(audio_segment.get_array_of_samples())
#     # Reshape to the expected format if stereo
#     if audio_segment.channels > 1:
#         samples = samples.reshape(-1, audio_segment.channels).mean(axis=1)
#     # Convert to float32 and normalize
#     samples = samples.astype(np.float32) / np.max(np.abs(samples))
#     return samples

# def transcribe_audio(audio_file):
#     audio_array = audio_file_to_numpy(audio_file)
#     result = speech_to_text_pipe(audio_array, return_timestamps=True)
#     return result['text']

# def synthesize_speech(text):
#     narrated_txt = text_to_speech_pipe(text)
#     # Create a unique filename
#     timestamp = int(time.time())
#     audio_file = f'output_{timestamp}.wav'
#     # Save audio to a file
#     sf.write(audio_file, narrated_txt["audio"][0], narrated_txt["sampling_rate"])
#     return audio_file

# @app.route("/process", methods=["POST"])
# def process_audio():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     try:
#         # Convert audio to text
#         transcription = transcribe_audio(file)
#         # Append suffix "Hi"
#         text_with_suffix = transcription + " Hi"
#         # Convert text to speech
#         audio_file = synthesize_speech(text_with_suffix)
#         # Return the audio file and text response
#         return jsonify({
#             "response": text_with_suffix,
#             "audio_file": f"/audio/{audio_file}"
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/audio/<filename>', methods=['GET'])
# def get_audio(filename):
#     # Ensure the file exists
#     if not os.path.exists(filename):
#         return jsonify({"error": "File not found"}), 404
#     return send_file(filename, mimetype='audio/wav')

# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=8080)





from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import pipeline
from pydub import AudioSegment
import numpy as np
import io
import soundfile as sf
import os
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the pipelines
speech_to_text_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", chunk_length_s=30)
text_to_speech_pipe = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")

def audio_file_to_numpy(audio_file):
    audio_data = audio_file.read()
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
    
    # Convert audio segment to numpy array
    samples = np.array(audio_segment.get_array_of_samples())
    # Reshape to the expected format if stereo
    if audio_segment.channels > 1:
        samples = samples.reshape(-1, audio_segment.channels).mean(axis=1)
    # Convert to float32 and normalize
    samples = samples.astype(np.float32) / np.max(np.abs(samples))
    return samples

def transcribe_audio(audio_file):
    audio_array = audio_file_to_numpy(audio_file)
    result = speech_to_text_pipe(audio_array, return_timestamps=True)
    return result['text']

def synthesize_speech(text):
    narrated_txt = text_to_speech_pipe(text)
    # Create a unique filename
    timestamp = int(time.time())
    audio_file = f'output_{timestamp}.wav'
    # Save audio to a file
    sf.write(audio_file, narrated_txt["audio"][0], narrated_txt["sampling_rate"])
    return audio_file

@app.route("/process", methods=["POST"])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Convert audio to text
        transcription = transcribe_audio(file)
        # Append suffix "Hi"
        text_with_suffix = transcription
        # Convert text to speech
        audio_file = synthesize_speech(text_with_suffix)
        # Return the audio file and text response
        return jsonify({
            "response": text_with_suffix,
            "audio_file": f"/audio/{audio_file}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/audio/<filename>', methods=['GET'])
def get_audio(filename):
    # Ensure the file exists
    if not os.path.exists(filename):
        return jsonify({"error": "File not found"}), 404
    return send_file(filename, mimetype='audio/wav')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
