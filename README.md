# Voice Pacman Server

FastAPI WebSocket server for real-time voice command detection using a trained CNN model.

## Features

- **WebSocket streaming**: Low-latency audio streaming via WebSocket
- **Real-time inference**: Processes audio every 50ms
- **Command detection**: Detects "up", "down", "left", "right" voice commands
- **Confidence thresholding**: Only sends commands with >95% confidence

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Convert PyTorch model to ONNX (for faster inference):
```bash
# Install dev dependencies (includes torch)
uv sync --extra dev

# Convert model to ONNX
python convert_to_onnx.py
```

This will create `training/models/best_pacman_model_with_background_noise.onnx`

**Note**: The server will automatically use the ONNX model if available, or fall back to PyTorch if not.

3. Ensure the model file exists at:
   - `training/models/best_pacman_model_with_background_noise.onnx` (recommended)
   - OR `training/models/best_pacman_model_with_background_noise.pth` (fallback)

4. Run the server:
```bash
python app.py
```

Or with uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API

### WebSocket Endpoint: `/ws`

**Connection**: `ws://localhost:8000/ws`

**Protocol**:
- **Client → Server**: Binary audio chunks (PCM 16-bit, mono, 16kHz)
- **Server → Client**: JSON messages with detected commands

**Response Format**:
```json
{
  "command": "up|down|left|right",
  "confidence": 0.95,
  "timestamp": 1234567890.123
}
```

## Audio Format

- **Sample Rate**: 16000 Hz
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit PCM
- **Format**: Binary (raw PCM data)

## Frontend Integration

Example JavaScript code to connect and send audio:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

// Get audio stream from microphone
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const audioContext = new AudioContext({ sampleRate: 16000 });
const source = audioContext.createMediaStreamSource(stream);
const processor = audioContext.createScriptProcessor(4096, 1, 1);

processor.onaudioprocess = (e) => {
  const inputData = e.inputBuffer.getChannelData(0);
  // Convert float32 to int16 PCM
  const int16Data = new Int16Array(inputData.length);
  for (let i = 0; i < inputData.length; i++) {
    int16Data[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
  }
  // Send binary data
  ws.send(int16Data.buffer);
};

source.connect(processor);
processor.connect(audioContext.destination);

// Receive commands
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Command detected:', data.command, data.confidence);
};
```

## Serwowanie frontendu (najprościej)

- Front: `frontend/client/index.html`
- Backend API info: `GET /api`
- WebSocket: `ws://localhost:8000/ws`

Serwer statycznie serwuje katalog `frontend/client/` pod `/` (tylko jeśli istnieje `frontend/client/index.html`).

## Architecture

- **Model Format**: Uses ONNX Runtime for fast inference (with PyTorch fallback)
- **Audio Streaming**: Client sends small audio chunks (50-100ms) continuously
- **Buffering**: Server maintains a 1-second rolling buffer
- **Inference**: Every 50ms, processes the 1 second of audio in the buffer
- **Command Detection**: Only sends commands above confidence threshold (95%)
