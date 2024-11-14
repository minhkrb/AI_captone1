from sentence_transformers import SentenceTransformer
import torch

class CalculateMatchPercentage:
    def __init__(self, model_name="truongcongminh/AI_captone1"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text, convert_to_tensor=False):
        return self.model.encode(text, convert_to_tensor=convert_to_tensor)

    def calculate_match_percentage(self, cv, job):
        cv_embedding = self.encode(cv, convert_to_tensor=True)
        job_embedding = self.encode(job, convert_to_tensor=True)
        cosine_sim = torch.nn.functional.cosine_similarity(cv_embedding, job_embedding, dim=0)

        return cosine_sim.item() * 100