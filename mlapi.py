from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

class scoringItem(BaseModel):
    credit_score: int
    age: int
    tenure: int
    balance: float
    products_number: int
    credit_card: int
    active_member: int
    estimated_salary: float


with open('gbmodel.pkl', 'rb') as f:
  model = pickle.load(f)

# y = model.predict(pd.DataFrame())
# print(y)


@app.post('/')
async def scoring_endpoint(item:scoringItem):
    df = pd.DataFrame([item.dict().values()], columns= item.dict().keys())
    yhat = model.predict(df)
    return {'prediction':int(yhat)}