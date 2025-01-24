## Deploy to kubernetes cluster

### Execution

```
# Ensure you are in kind cluster context 
kind create cluster                         # first time only
kubectl cluster-info --context kind-kind

cd kube/local
kubectl apply -f model-deployment.yaml
kubectl apply -f model-service.yaml

kubectl port-foward service/chase-pred-svc 8080:80  # Port-forward to test 

cd ../ipl_infer
pipenv shell
(ipl_infer) python k_client.py

```
>  Accessing Kube Deployed IPLPredictionService 
{"chasing_team":"RCB","win_probability":0.6951505192319617}


### Handy commands to debug

```
kubectl get deployments     # check for successful deployments 
kubectl get svc             # check for running services

kubectl get pods                # check for running pods
kubectl describe pod <podname>  # debug for errors (port already in use.. )
```









