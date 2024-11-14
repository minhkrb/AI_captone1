from fastapi import FastAPI, HTTPException
import os
from dotenv import load_dotenv
from huggingface_hub import login
from pydantic import BaseModel, Field
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from services.CaculatorPer import CalculateMatchPercentage

load_dotenv("./env/.env")

HF_TOKEN = os.getenv("HF_TOKEN")

login(HF_TOKEN)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

per_model = CalculateMatchPercentage()

class ComparisonContent(BaseModel):
    professionalSkillsCV: str
    educationsCV: str
    languagesCV: str
    certificationsCV: str
    professionalSkillsJob: str
    educationsJob: str
    languagesJob: str
    certificationsJob: str
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.post("/get_percents")
async def get_percents(comparison: ComparisonContent):
    try:
        score_skill = 100 if len(comparison.professionalSkillsJob) == 0 else (
            0 if len(comparison.professionalSkillsCV) == 0 else per_model.calculate_match_percentage(comparison.professionalSkillsCV, comparison.professionalSkillsJob)
        )
        score_edu = 100 if len(comparison.educationsJob) == 0 else (
            0 if len(comparison.educationsCV) == 0 else per_model.calculate_match_percentage(comparison.educationsCV, comparison.educationsJob)
        )
        score_lang = 100 if len(comparison.languagesJob) == 0 else (
            0 if len(comparison.languagesCV) == 0 else per_model.calculate_match_percentage(comparison.languagesCV, comparison.languagesJob)
        )
        score_cer = 100 if len(comparison.certificationsJob) == 0 else (
            0 if len(comparison.certificationsCV) == 0 else per_model.calculate_match_percentage(comparison.certificationsCV, comparison.certificationsJob)
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Data validation error: {str(ve)}")
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Cannot connect to the model service")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

    return {
        "score_skill": score_skill,
        "score_edu": score_edu,
        "score_lang": score_lang,
        "score_cer": score_cer
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
