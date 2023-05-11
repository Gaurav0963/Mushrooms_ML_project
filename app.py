import sys
from flask import Flask, request, render_template

from source_code.logger import logging
from source_code.exception import CustomException
from source_code.pipeline.prediction_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


# Route for a home page
@app.route('/')
@app.route('/home')
def index():
    try:
        logging.info("Home Page -> index.html")
        return render_template('index.html')
    except Exception as e:
        raise CustomException(e, sys)


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            cap_shape=request.form.get("cap_shape"),
            cap_surface=request.form.get("cap_surface"),
            cap_color=request.form.get("cap_color"),
            bruises=request.form.get("bruises"),
            odor=request.form.get("odor"),
            gill_attachment=request.form.get("gill_attachment"),
            gill_spacing=request.form.get("gill_spacing"),
            gill_size=request.form.get("gill_size"),
            gill_color=request.form.get("gill_color"),
            stalk_shape=request.form.get("stalk_shape"),
            stalk_root=request.form.get("stalk_root"),
            stalk_surface_above_ring=request.form.get("stalk_surface_above_ring"),
            stalk_surface_below_ring=request.form.get("stalk_surface_below_ring"),
            stalk_color_above_ring=request.form.get("stalk_color_above_ring"),
            stalk_color_below_ring=request.form.get("stalk_color_below_ring"),
            veil_type=request.form.get("veil_type"),
            veil_color=request.form.get("veil_color"),
            ring_number=request.form.get("ring_number"),
            ring_type=request.form.get("ring_type"),
            spore_print_color=request.form.get("spore_print_color"),
            population=request.form.get("population"),
            habitat=request.form.get("habitat")
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.model_predict(pred_df)
        print("after Prediction")
        return render_template('home.html', results=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1000, debug=True)
