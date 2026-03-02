import os
import asyncio
import httpx
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
HUME_KEY = os.getenv("HUME_API_KEY")

@app.post("/qc-score")
async def get_intent_score(file: UploadFile = File(...)):
    headers = {"X-Hume-Api-Key": HUME_KEY}
    
    # 1. Upload the file to Hume's Batch API
    async with httpx.AsyncClient() as client:
        # We send the file as 'multipart/form-data'
        files = {"file": (file.filename, await file.read(), file.content_type)}
        # We tell Hume to use the 'prosody' (voice) model
        data = {"json": '{"models": {"prosody": {}}}'}
        
        response = await client.post(
            "https://api.hume.ai/v0/batch/jobs",
            headers=headers,
            files=files,
            data=data
        )
        
        job_id = response.json().get("job_id")
        if not job_id:
            return {"error": "Failed to start job", "details": response.text}

        # 2. Poll for Completion (Checking every 2 seconds)
        while True:
            status_req = await client.get(f"https://api.hume.ai/v0/batch/jobs/{job_id}", headers=headers)
            status = status_req.json()["state"]["status"]
            
            if status == "COMPLETED":
                break
            elif status == "FAILED":
                return {"error": "Analysis failed"}
            await asyncio.sleep(2)

        # 3. Get the Results
        predictions_req = await client.get(f"https://api.hume.ai/v0/batch/jobs/{job_id}/predictions", headers=headers)
        results = predictions_req.json()
        
        # 4. Extract the Top Emotion (The "Intent")
        # Drilling down into the Hume JSON structure:
        vocal_data = results[0]["results"]["predictions"][0]["models"]["prosody"]["grouped_predictions"][0]["predictions"][0]
        emotions = vocal_data["emotions"]
        top_emotion = sorted(emotions, key=lambda x: x["score"], reverse=True)[0]

        return {
            "filename": file.filename,
            "detected_intent": top_emotion["name"],
            "intent_score": round(top_emotion["score"] * 100, 2)
        }