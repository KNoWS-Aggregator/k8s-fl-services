# Federated training client deployment

The example Deployment runs data preparation on port `8080` and model training
on port `8081` in the same Pod. Both containers mount
`federated-training-client-data` at `/app/data`.

Apply the PVC before the Deployment when testing the raw example:

```sh
kubectl apply -f deployments/federated-training-client/pvc.yaml
kubectl apply -f deployments/federated-training-client/deployment.example.yaml
```

Replace the placeholder case-slice URL first. The raw example uses the same
deployment values as the current deployment function: dataset `accellero`,
polling enabled, and an `@hourly` polling schedule.

This example does not define external routing. The aggregator platform must
publish a single logical service root and route:

- `/results` and public `/preparation/status` to the data-preparation port
  (`/preparation/status` maps to internal `/status`);
- public `/status`, session, training, weights, metrics, and evaluation paths to the
  model-training port.

See `service-descriptions/federated-training-client/profile.yaml` and
`deployment-function.yaml` for the current aggregator configuration.

For the raw Deployment (without aggregator routing), access preparation status
directly by forwarding its Pod port:

```sh
kubectl port-forward deployment/federated-training-client 8080:8080
curl http://127.0.0.1:8080/status
```

Container health and readiness probes can address their Pod ports directly.
