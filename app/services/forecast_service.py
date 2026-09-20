import pandas as pd

def make_prediction(data, model, features):
    input_data = pd.DataFrame(
        [data.model_dump()]
    )
    input_data = input_data[features]
    prediction = model.predict(input_data)
    prediction_value = float(prediction[0])
    return prediction_value