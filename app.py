from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import numpy as np
from pathlib import Path
from audio_processor import AudioStreamProcessor
from constants import ACTION_LABELS, DEBUG, INFERENCE_INTERVAL

app = FastAPI(title="Voice Pacman Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio and receiving command detections.
    
    Protocol:
    - Client sends: Binary audio chunks (PCM 16-bit, mono, 16kHz)
    - Server sends: JSON messages with detected commands
      Format: {"command": "up|down|left|right", "confidence": 0.95, "timestamp": 1234567890.123}
    """
    await websocket.accept()
    
    processor = AudioStreamProcessor()
    print(f"✅ Client connected: {websocket.client}")
    
    try:
        inference_task = asyncio.create_task(
            run_inference_loop(websocket, processor)
        )
        plotting_task = None
        if DEBUG:
            plotting_task = asyncio.create_task(
                run_plotting_loop(processor)
            )
        
        while True:
            data = await websocket.receive_bytes()
            
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            audio_data = audio_data / 32768.0
            
            processor.add_audio_chunk(audio_data)
    
    except WebSocketDisconnect:
        print(f"❌ Client disconnected: {websocket.client}")
    except Exception as e:
        print(f"❌ Error in WebSocket: {e}")
    finally:
        inference_task.cancel()
        if plotting_task is not None:
            plotting_task.cancel()
        try:
            await inference_task
            if plotting_task is not None:
                await plotting_task
        except asyncio.CancelledError:
            pass


async def run_inference_loop(websocket: WebSocket, processor: AudioStreamProcessor):
    try:
        while True:
            probabilities = processor.process_audio()
            
            if probabilities is not None:
                action, confidence = processor.find_action_with_threshold(probabilities)
                
                if action and confidence is not None and action in ACTION_LABELS:
                    message = {
                        "command": action,
                        "confidence": round(confidence, 4),
                    }
                    await websocket.send_json(message)
                    print(f"🎯 Command sent: {action.upper()} ({confidence*100:.1f}%)")
                        
                    processor.clear_buffer()
                elif processor.check_other_detected(probabilities):
                    processor.clear_buffer()
            
            await asyncio.sleep(INFERENCE_INTERVAL)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"❌ Error in inference loop: {e}")
        import traceback
        traceback.print_exc()


async def run_plotting_loop(processor: AudioStreamProcessor):
    """Plot the audio buffer every second."""
    try:
        while True:
            await asyncio.sleep(1.0)  # Plot every second
            processor.plot_buffer()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"❌ Error in plotting loop: {e}")

app.mount("/", StaticFiles(directory="frontend/build/client", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
