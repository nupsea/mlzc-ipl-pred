# mlzc-ipl-pred

This project analyzes **IPL (Indian Premier League)** cricket match data and builds a model to predict the **win probability** of the chasing team **at any point** during the second innings (post half-game). The approach includes:

- **EDA (Exploratory Data Analysis) and Feature Engineering**  
- **Model Training & Evaluation**  
- **Realistic Sanity Checks** (e.g., no negative scores, no more than 10 wickets)  
- **Inference via Python Script/Dockerized Flask Service/Local or Cloud K8s Deployed Service**  

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
   - In the `ipl_build` folder, you’ll find raw data (`deliveries.csv`, `matches.csv`).  
   - The Jupyter notebook `half_match.ipynb` shows how we explore the data, engineer features, and evaluate the model.  

2. **Model Training**  
   - Run `hm_train.py` to load data, create features, and train the model.  
   - The script saves the trained model and encoders as `ipl_chase_pred_v1.bin` in the `ipl_infer/MODEL` folder by default.

3. **Inference**  
   - You can make predictions either via a **standalone Python script** (`predict.py`) or a **Dockerized Flask web service** (`predict_ws.py`).

4. **Realistic Checks**  
   - The inference code includes **sanity checks** to ensure the input is valid (e.g., no negative current_score, wickets_down ≤ 10, etc.).  
   - If the batting team already surpassed the target or lost all wickets, the code will short-circuit and provide an immediate outcome (100% or 0% win probability).

---

## Setup & Commands

Below is a step-by-step guide to replicating the environment and running the pipeline.


### 0. Prerequisites

To only play with the model with a UI, you will need:
- **Python 3.11**  
- **Pipenv** (or an equivalent virtual environment manager)
  
Go to [Play with App](#6-play-with-streamlit-ui)


Besides, to explore OPS with respect to analysis, training, deployment on local/cloud, you will need the following:

- **Docker**
- **Kind** (for a local Kubernetes cluster)
- **kubectl** (Kubernetes CLI)
- **AWS Cloud** (AWS account for deploying onto AWS cloud)


### 1. Train the Model (Optional)

Enter the `ipl_build` folder, install dependencies with `Pipenv` and activate the virtual environment to train the model.

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


### 6. Play with Streamlit UI

A simple **Streamlit** UI is provided to interactively input match context and get the predicted outcome for a chasing team in IPL T20. 

### How to Run

1. **Install dependencies** in ipl_infer through pipenv if not already done.
```
cd ipl_infer
pipenv install
pipenv shell
```

2. **Start** the model inference service (Flask Web Service, Docker, etc.) on port `9696`.
(Refer to any of the previous local ones.)
```
(ipl_infer) python predict_ws.py
```
3. **Run** the Streamlit app (on another terminal)
```bash
(ipl_infer) streamlit run app.py
```

4. Access the Streamlit UI in your browser at http://localhost:8501 . Fill in the necessary input params and hit on Predict button. 



### Notes & Disclaimers

* Data Validations: The prediction code checks for invalid inputs (e.g., negative scores, wickets > 10). If the input is already a decided match (chasing team surpassing target or 10 wickets down), it outputs a direct result.
* Formats & Features: Currently tailored for T20 data. The code may need adaptation for ODIs or other formats (e.g., changing total overs, ball count).
* Further Customization: Additional feature engineering (e.g., rolling stats, match context) can significantly affect accuracy. Refer to half_match.ipynb for more details.
* Production Considerations: For production usage, ensure robust logging, error handling, and possibly a database or queue for handling real-time predictions.


### Acknowledgements

The dataset used for this project is sourced from Kaggle:
https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020

### Contact / Contributions
Feel free to fork this repository or submit pull requests with improvements, bug fixes, or extended features. If you have questions, you can open an issue.

Enjoy experimenting with IPL chase predictions!
