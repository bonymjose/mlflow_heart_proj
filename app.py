from flask import Flask, render_template, request, redirect, url_for
from flask import Flask, render_template, redirect, url_for, flash
import mlflow
from training import train_model
import pandas as pd


app = Flask(__name__)

MLFLOW_URI = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(MLFLOW_URI)
app.secret_key = "super_secret_key"

@app.route('/')
def index():
    # Initialize as an empty DataFrame
    runs = pd.DataFrame()

    try:
        experiment_id = "0"
        fetched_runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string="")
        
        # If fetched_runs is not empty, update the runs DataFrame
        if not fetched_runs.empty:
            runs = fetched_runs  

    except Exception as e:
        flash(f"An error occurred: {str(e)}", "danger")

    return render_template('index.html', runs=runs)




@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        model_type = request.form['model_type']

        # Fetch hyperparameters
        n_estimators = int(request.form.get('n_estimators', 100))  # Default to 100 if not provided
        max_depth_value = request.form.get('max_depth')
        max_depth = int(max_depth_value) if max_depth_value else None
        C = float(request.form.get('C', 1.0))                      # Default to 1.0 if not provided

        train_model(model_type, n_estimators, max_depth, C)

        # After training, redirect to index or display a success message
        return redirect(url_for('index'))

    # If it's a GET request, display the form
    return render_template('train.html')

@app.route('/deploy/<run_id>')
def deploy(run_id):
    # Pass the run_id to the deploy.html template
    return render_template('deploy.html', run_id=run_id)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
