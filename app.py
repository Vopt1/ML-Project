from flask import Flask,request,render_template
from src.logger import logging
from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET',"POST"])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Airline = request.form.get('airline'),
            Date_of_Journey = request.form.get('date_of_journey'),
            Source = request.form.get('source'),
            Destination = request.form.get('destination'),
            Dep_Time = request.form.get('dep_time'),
            Arrival_Time = request.form.get('arrival_time'),
            Duration = request.form.get('duration'),
            Total_Stops = request.form.get('total_stops'),
            Additional_Info = request.form.get('Additional_Info'),
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        return render_template('results.html',result=result[0])
    
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)