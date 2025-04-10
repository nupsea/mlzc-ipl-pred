# mlzc-ipl-pred


This project ([mlzc-ipl-pred git repo](https://github.com/nupsea/mlzc-ipl-pred)) analyzes **IPL (Indian Premier League)** cricket match data and builds a model to predict the **win probability** of the chasing team **at any point** during the second innings (post half-game). The approach includes:

- **EDA (Exploratory Data Analysis) and Feature Engineering**  
- **Model Training & Evaluation**  
- **Realistic Sanity Checks** (e.g., no negative scores, no more than 10 wickets)  
- **Inference via Python Script / Dockerized Flask Service / Kubernetes (Local or AWS EKS) / Streamlit UI**  

---

## 0. Quickstart: Play with Streamlit UI

If you’d like to **quickly experiment** with the model via a web interface, follow these steps:

### Prerequisites

- **Python 3.11** (download/install from [python.org](https://www.python.org/downloads/))  
- **Pipenv** (or an equivalent virtual environment manager)
  ```bash
  pip install --user pipenv
  ```

### Steps

1. Clone this repository, open the `ipl_infer` directory.
```bash
git clone <this-repo>
cd ipl_infer
```

2. Install dependencies and activate the virtual environment.
```bash
pipenv install
pipenv shell
```

3. Start the model inference service.
```
(ipl_infer) python predict_ws.py
```
This runs a Flask service on port 9696.

4. Open another terminal, get into the virtual environment and Run the Streamlit app:
```bash
pipenv shell 
(ipl_infer) streamlit run app.py
```

5. Open http://localhost:8501 in your browser to use the UI.
   Fill in match details (target score, current score, balls remaining, etc.) and click Predict.
   The app will display a win probability for the chasing team.
   
---

## Directory Structure

```

.
├── README.md
├── archive/
├── ipl_build
│   ├── DATA
│   │   ├── deliveries.csv
│   │   └── matches.csv
│   ├── Pipfile
│   ├── Pipfile.lock
│   ├── half_match.ipynb
│   └── hm_train.py
├── ipl_infer
│   ├── Dockerfile
│   ├── MODEL
│   │   └── ipl_chase_pred_v1.bin
│   ├── Pipfile
│   ├── Pipfile.lock
│   ├── app.py
│   ├── client.py
│   ├── eks_client.py
│   ├── k_client.py
│   ├── predict.py
│   └── predict_ws.py
└── kube
    ├── eks
    │   ├── chase-deployment.yaml
    │   ├── chase-service.yaml
    │   ├── cloud-deployment.md
    │   ├── image-1.png
    │   └── image.png
    └── local
        ├── README.md
        ├── model-deployment.yaml
        └── model-service.yaml

```

## Workflow Overview

1. **Data & EDA**  
   - In the `ipl_build/DATA` folder, you’ll find raw data (`deliveries.csv`, `matches.csv`).  
   - The Jupyter notebook `half_match.ipynb` shows how we explore the data, engineer features, and evaluate the model.  

2. **Model Training**  
   - Run `hm_train.py` to load data, create features, and train the model.  
   - The script saves the trained model and encoders as `ipl_chase_pred_v1.bin` in the `ipl_infer/MODEL` folder by default.

3. **Inference**  
   - You can make predictions either via a **standalone Python script** (`predict.py`), a **Dockerized Flask web service** (`predict_ws.py`) or making use of Kuberentes or AWS Services. 

4. **Realistic Checks**  
   - The inference code includes **sanity checks** to ensure the input is valid (e.g., no negative current_score, wickets_down ≤ 10, etc.).  
   - If the batting team already surpassed the target or lost all wickets, the code will short-circuit and provide an immediate outcome (100% or 0% win probability).

---

## Setup & Commands

Below is a step-by-step guide to explore deployment and operations, you will/may need the following depending on what you intend to try. 

### Prerequisites
- **Python 3.11**
- **Pipenv**
- **Docker**
- **Kind** (for a local Kubernetes cluster)
- **kubectl** (Kubernetes CLI)
- **AWS Cloud** (AWS account for deploying onto AWS cloud)


### 1. Train the Model (Optional)

Enter the `ipl_build` folder, install dependencies with `Pipenv` and activate the virtual environment to train the model.
(Explore using half_match.ipynb notebook and make changes in `hm_train.py` correspondingly to modify/optimize the model behaviour, else skip to Step 2.)

   ```bash
   cd ipl_build

   pipenv install
   pipenv shell

   (ipl_build) python hm_train.py

   ```
Exit the Pipenv shell when finished.

### 2. Run Inference

Go to `ipl_infer` folder, install dependencies with `Pipenv` and activate the virtual environment to run predictions from the model.

   ```bash
   cd ipl_infer

   pipenv install
   pipenv shell

   (ipl_infer) python predict.py

   ```
This will load the trained model (ipl_chase_pred_v1.bin) and make predictions for the sample input in predict.py.
You should see output resembling:


> Chasing team 'RCB' has a Predicted Win probability of: 69.52%



### 3. Dockerized Flask Web Service
Build the Docker image and run the container, exposing it on port 9696:

```
(ipl_infer) docker build -t anse/ipl-chase-pred .

(ipl_infer) docker run -it --rm -p 9696:9696 anse/ipl-chase-pred
```

<i> <b>Test</b> the service from a separate terminal: </i>

```
# Still in ipl_infer, open a new shell
pipenv shell
(ipl_infer) python client.py
```
This script sends a JSON payload to http://localhost:9696/predict and prints the response:
> Accessing IPLPredictionService Model Endpoint
{"chasing_team":"RCB","win_probability":0.6951505192319618}


### 4. Kubernetes Deployed Service


```
# Ensure you are in kind cluster context 
kind create cluster                         # first time only
kubectl cluster-info --context kind-kind

cd kube
kubectl apply -f model-deployment.yaml
kubectl apply -f model-service.yaml

kubectl port-foward service/chase-pred-svc 8080:80  # Port-forward to test 

```

<i> <b>Test</b> the service from a separate terminal: </i>


```
cd ../ipl_infer
pipenv shell
(ipl_infer) python k_client.py
```

This script now sends a JSON payload to http://localhost:8080/predict and prints the response:

```
>  Accessing Kube Deployed IPLPredictionService 
{"chasing_team":"RCB","win_probability":0.6951505192319617}

```

### 5. AWS Service using EKS

The below link provides you an overview of how this application was deployed onto AWS EKS using a Docker image hosted on ECR and its subsequenting testing.

Refer to [Cloud Deployment](./kube/eks/cloud-deployment.md)



### Notes & Disclaimers

* Data Validations: The prediction code checks for invalid inputs (e.g., negative scores, wickets > 10). If the input is already a decided match (chasing team surpassing target or 10 wickets down), it outputs a direct result.
* Formats & Features: Currently tailored for T20 data. The code may need adaptation for ODIs or other formats (e.g., changing total overs, ball count).
* Further Customization: Additional feature engineering (e.g., rolling stats, match context) can significantly affect accuracy. Refer to half_match.ipynb for more details.
* Production Considerations: For production usage, ensure robust logging, error handling, and possibly a database or queue for handling real-time predictions.


### Acknowledgements

* The dataset used for this project is sourced from Kaggle:
https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
* This project was developed from the learnings of the course: https://github.com/DataTalksClub/machine-learning-zoomcamp

### Contact / Contributions
Feel free to fork this repository or submit pull requests with improvements, bug fixes, or extended features. If you have questions, you can open an issue.

Enjoy experimenting with IPL chase predictions!
