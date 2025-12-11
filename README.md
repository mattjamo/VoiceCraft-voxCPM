# VoiceCraft - VoxCPM

VoiceCraft is made up of two components a Server to host the AI model and a Client to interact with the server. This allows the user to create and manage voices and then utilize those voices in any application that supports the OpenAI API.

# VoxCPM OpenAI-Compatible API Server

This project provides an OpenAI-compatible API server for the VoxCPM Text-to-Speech model. It allows you to use VoxCPM with standard OpenAI client libraries and tools that support the `/v1/audio/speech` endpoint.

## Features

- **OpenAI Compatibility**: Drop-in replacement for `client.audio.speech.create`.
- **Streaming Support**: Real-time audio streaming (raw PCM) using `stream=True`.
- **High Quality & Low Latency**: Configure via `inference_timesteps`.
- **Customizable generation**: Adjust `cfg_value` and `inference_timesteps` per request.

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: On Windows, you may need `triton-windows` or WSL2 for full GPU support. pip install triton-windows*

2. **Download Model**:
   The model `openbmb/VoxCPM1.5` will be automatically downloaded on the first run.

   > [!NOTE]
   > Models are saved in the Hugging Face cache directory.
   > - Windows: `C:\Users\<YourUsername>\.cache\huggingface\hub`
   > - Linux/Mac: `~/.cache/huggingface/hub`
   > 
   > To change this location, set the `HF_HOME` environment variable before running the server.

## Usage

### Starting the Server
Run the Flask application:
```bash
python server.py
# Server will listen on http://127.0.0.1:5000
```

### API Endpoint: `POST /v1/audio/speech`

**Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model` | string | Model ID (id ignored, accepted for compatibility) | `voxcpm` |
| `input` | string | Text to generate speech for | Required |
| `voice` | string | Voice ID (currently unused) | `default` |
| `response_format` | string | Audio format: `wav` or `pcm` | `wav` |
| `speed` | number | Speed (not yet implemented) | 1.0 |

**Extended Parameters (via `extra_body` in OpenAI client or JSON root):**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `stream` | boolean | Stream response chunk-by-chunk | `False` |
| `cfg_value` | float | Classifier-Free Guidance scale. Higher = text adherence. | `2.0` |
| `inference_timesteps` | int | Number of diffusion steps. Higher = quality, Lower = speed. | `10` |

### Coding Examples

#### Python (OpenAI Library)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5000/v1",
    api_key="sk-dummy" # Not used
)

# Standard Generation
response = client.audio.speech.create(
    model="voxcpm",
    voice="default",
    input="Hello from VoxCPM!",
)
response.stream_to_file("output.wav")

# Streaming (Raw PCM)
response = client.audio.speech.create(
    model="voxcpm",
    voice="default",
    input="Streaming is cool.",
    response_format="pcm",
    extra_body={"stream": True}
)
with open("output.pcm", "wb") as f:
    for chunk in response.iter_bytes():
        f.write(chunk)
```

#### CURL

```bash
curl http://localhost:5000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm",
    "input": "Testing the API.",
    "cfg_value": 4.0
  }' \
  --output test.wav
```

## VoiceCraft Client GUI

A sleek, desktop client is available for easy interaction with the server.

### Prerequisites
Install GUI dependencies:
```bash
pip install PyQt6 pygame pyperclip requests
# Or simply
pip install -r requirements.txt
```

### Running the Client
Make sure the server is running (`python server.py`), then start the client:
```bash
python client_gui.py
```

**Features:**
- **Play Paste**: Automatically grabs text from clipboard, generates audio, and plays it.
- **Parameters**: Adjust CFG Scale and Inference Steps via sidebar sliders.

## Voice Profiles (Cloning)

VoxCPM uses "prompt audio" to clone voices. You require a .wav or .mp3 file and a transcript of the audio.

1.  Navigate to the `voices/` directory in the project root (created on first run).
2.  Drop any **short (3-10s)** `.wav` or `.mp3` file in this folder.
    *   Example: `voices/alice.wav`
3.  **Create a text file** with the same name containing the transcript of the audio.
    *   Example: `voices/alice.txt` containing "The quick brown fox jumps over the lazy dog."
4.  The file name becomes the voice ID (e.g., `alice`).
4.  **GUI**: The voice will appear in the dropdown list (click "Refresh").
5.  **API**: Use `voice="alice"` in your request.

### Alternatively: Create or clone a voice profile using the GUI

1.  **Enter Transcript**: Type the text you intend to speak in the main text input area.
2.  **Voice Name**: Enter a unique name for your voice in the "Voice Name" field (Sidebar).
3.  **Record**: Click the **Record** button and speak clear audio matching the text.
4.  **Save**: Click **Stop** to finish. The audio/text pair is automatically saved to `voices/` and selected.
5.  **Test**: Click **Generate** to hear the result (uses the same text as prompt) or type a new message in the main text input area.

> [!TIP]
> Use a clean, high-quality recording for the best cloning results.

## Testing

Two test scripts are included:
- `test_server.py`: Simple verification using `requests`.
- `test_openai_compatibility.py`: Full verification using the `openai` python library, covering streaming and custom parameters.

## Acknowledgements

This application is powered by **VoxCPM**, developed by [OpenBMB](https://github.com/OpenBMB). We gratefully acknowledge their team for the development of this open-source AI model and their valuable contributions to the community.

- **GitHub Repository**: [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- **Hugging Face Model**: [openbmb/VoxCPM1.5](https://huggingface.co/openbmb/VoxCPM1.5)

## Disclaimer

This project is an independent implementation and is not affiliated with, endorsed by, or connected to any employer, including that of the developer.

Additionally, this software is a personal project and is **not** associated with, sponsored by, or legally linked to any employer in any way.

DO NOT USE THIS SOFTWARE FOR ANY ILLEGAL OR UNETHICAL PURPOSES. THE DEVELOPER IS NOT LIABLE FOR ANY DAMAGE OR LOSS CAUSED BY THE USE OF THIS SOFTWARE. NOR DO I CONDONE OR ENDORSE ANY ILLEGAL OR UNETHICAL USE OF THIS SOFTWARE.
THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

NEVER CLONE AN INDIVIDUALS VOICE WITHOUT THEIR PERMISSION. THIS IS ILLEGAL AND UNETHICAL.
