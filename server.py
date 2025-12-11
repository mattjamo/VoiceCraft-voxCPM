from flask import Flask, request, jsonify, send_file
import io
import os
import soundfile as sf
import numpy as np
from voxcpm import VoxCPM
import logging
import time

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

# Metrics
metrics = {
    "total_queries": 0,
    "total_words": 0,
    "total_processing_time": 0.0,
    "total_audio_duration_seconds": 0.0
}

def load_model():
    global model
    try:
        logger.info("Loading VoxCPM model...")
        # Set to the 1.5
        model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5")
        logger.info("VoxCPM model loaded successfully.")
        
        # Ensure voices directory exists
        if not os.path.exists("voices"):
            os.makedirs("voices")
            logger.info("Created 'voices' directory.")
            
    except Exception as e:
        logger.error(f"Failed to load VoxCPM model: {e}")
        model = None

@app.route('/health', methods=['GET'])
def health():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    return jsonify({"status": "ok", "message": "Model is ready"}), 200

@app.route('/v1/voices', methods=['GET'])
def list_voices():
    voices = ["default"]
    if os.path.exists("voices"):
        for filename in os.listdir("voices"):
            if filename.lower().endswith(('.wav', '.mp3')):
                voices.append(os.path.splitext(filename)[0])
    return jsonify({"voices": voices})

@app.route('/v1/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics)

@app.route('/v1/system/paths', methods=['GET'])
def get_system_paths():
    # Attempt to find Hugging Face cache
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        if os.name == 'nt':
            hf_home = os.path.join(os.environ.get("USERPROFILE"), ".cache", "huggingface", "hub")
        else:
            hf_home = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    
    # Check if specific model dir exists (best guess)
    model_dir = os.path.join(hf_home, "models--openbmb--VoxCPM1.5")
    
    return jsonify({
        "model_path": model_dir if os.path.exists(model_dir) else hf_home,
        "voices_path": os.path.abspath("voices"),
        "output_path": os.path.abspath("saved_outputs") # For client reference if needed
    })

@app.route('/v1/audio/speech', methods=['POST'])
def text_to_speech():
    if model is None:
        return jsonify({"error": {"message": "Model not loaded", "type": "server_error", "param": None, "code": None}}), 503

    if not request.is_json:
        return jsonify({"error": {"message": "Invalid request, JSON body required", "type": "invalid_request_error", "param": None, "code": None}}), 400

    data = request.get_json()
    input_text = data.get('input')
    
    # Optional parameters
    model_name = data.get('model', 'voxcpm')
    voice = data.get('voice', 'default')
    response_format = data.get('response_format', 'wav')
    stream = data.get('stream', False)
    
    # VoxCPM specific parameters
    cfg_value = data.get('cfg_value', 2.0)
    inference_timesteps = data.get('inference_timesteps', 10)
    
    # Advanced Retry Parameters
    retry_badcase = data.get('retry_badcase', True)
    retry_badcase_max_times = data.get('retry_badcase_max_times', 3)
    retry_badcase_ratio_threshold = data.get('retry_badcase_ratio_threshold', 6.0)
    
    if not input_text:
        return jsonify({"error": {"message": "Missing 'input' field", "type": "invalid_request_error", "param": "input", "code": None}}), 400

    try:
        start_time = time.time()
        
        # Determine prompt path
        prompt_wav_path = None
        prompt_text = None

        if voice and voice != "default":
            # Check for exact file or with extension
            voice_path = os.path.join("voices", f"{voice}.wav")
            if not os.path.exists(voice_path):
                 voice_path = os.path.join("voices", f"{voice}.mp3")
            
            if os.path.exists(voice_path):
                # Check for corresponding text file (transcript)
                text_path = os.path.splitext(voice_path)[0] + ".txt"
                
                if os.path.exists(text_path):
                    prompt_wav_path = voice_path
                    with open(text_path, "r", encoding="utf-8") as f:
                        prompt_text = f.read().strip()
                    logger.info(f"Using voice prompt: {prompt_wav_path}")
                    logger.info(f"Using prompt text: {prompt_text}")
                else:
                    logger.warning(f"Voice '{voice}' found at {voice_path}, but missing transcript file at {text_path}. Using default voice instead.")
                    
            else:
                logger.warning(f"Voice '{voice}' not found, using default.")

        if stream:
            # Streaming response (Raw PCM or chunks)
            # VoxCPM generates streaming chunks as numpy arrays
            def generate():
                for chunk in model.generate_streaming(
                    text=input_text,
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text,
                    cfg_value=float(cfg_value),
                    inference_timesteps=int(inference_timesteps),
                    retry_badcase=retry_badcase,
                    retry_badcase_max_times=int(retry_badcase_max_times),
                    retry_badcase_ratio_threshold=float(retry_badcase_ratio_threshold)
                ):
                    # Convert to int16 PCM
                    audio_int16 = (chunk * 32767).astype(np.int16)
                    yield audio_int16.tobytes()
            
            return app.response_class(generate(), mimetype='audio/pcm')
            
        else:
            logger.info(f"Generating speech for text: {input_text[:50]}...")
            
            wav = model.generate(
                text=input_text,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                cfg_value=float(cfg_value),
                inference_timesteps=int(inference_timesteps),
                retry_badcase=retry_badcase,
                retry_badcase_max_times=int(retry_badcase_max_times),
                retry_badcase_ratio_threshold=float(retry_badcase_ratio_threshold)
            )
            
            buffer = io.BytesIO()
            if response_format == 'pcm':
                 # Convert to int16 PCM
                audio_int16 = (wav * 32767).astype(np.int16)
                buffer.write(audio_int16.tobytes())
                mimetype = 'audio/pcm'
                download_name = 'speech.pcm'
            else:
                # Default to WAV
                sf.write(buffer, wav, model.tts_model.sample_rate, format='WAV')
                mimetype = 'audio/wav'
                download_name = 'speech.wav'
            
            buffer.seek(0)
            
            # Metrics Update
            end_time = time.time()
            process_time = end_time - start_time
            audio_duration = len(wav) / model.tts_model.sample_rate
            
            metrics["total_queries"] += 1
            metrics["total_words"] += len(input_text.split())
            metrics["total_processing_time"] += process_time
            metrics["total_audio_duration_seconds"] += audio_duration
            
            return send_file(
                buffer,
                mimetype=mimetype,
                as_attachment=True,
                download_name=download_name
            )

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return jsonify({"error": {"message": str(e), "type": "server_error", "param": None, "code": None}}), 500

if __name__ == '__main__':
    load_model()
    # OpenAI usually runs on standard ports; 5000 is default Flask
    app.run(host='0.0.0.0', port=5000)
