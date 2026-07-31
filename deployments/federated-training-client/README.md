# Federated training client deployment

The example Deployment runs data preparation on port `8080` and model training
on port `8081` in the same Pod. Both containers mount
`federated-training-client-data` at `/app/data`.

Apply the PVC before the Deployment when testing the raw example:

```sh
kubectl apply -f deployments/federated-training-client/pvc.yaml
kubectl apply -f deployments/federated-training-client/deployment.example.yaml
```

Replace the placeholder source and authentication values first.

This example does not define external routing. The aggregator platform must
publish a single logical service root and route:

- `/prepare` and `/results` to the data-preparation port;
- session, training, weights, metrics, and evaluation paths to the
  model-training port.

Container health and readiness probes can address their Pod ports directly.

