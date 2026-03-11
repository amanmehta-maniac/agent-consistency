#!/bin/bash
set -e

NAMESPACE="mltraining-dev"

echo "Creating experiment-60q-code ConfigMap from source files..."
kubectl create configmap experiment-60q-code \
  --namespace="${NAMESPACE}" \
  --from-file=run_60q_experiment.py=../../hotpotqa/run_60q_experiment.py \
  --from-file=agent.py=../../hotpotqa/agent.py \
  --from-file=tools.py=../../hotpotqa/tools.py \
  --from-file=new_60_questions.json=../../hotpotqa/new_60_questions.json \
  --dry-run=client -o yaml | kubectl apply -f -

echo "ConfigMap created/updated."

echo "Applying PVC..."
kubectl apply -f experiment-60q-pvc.yaml

echo "Applying Job..."
kubectl apply -f experiment-60q-job.yaml

echo ""
echo "Deployment complete. Monitor with:"
echo "  kubectl get pods -n ${NAMESPACE} -l job-name=experiment-60q -w"
echo "  kubectl logs -n ${NAMESPACE} -l job-name=experiment-60q -c experiment -f"
echo "  kubectl exec -n ${NAMESPACE} \$(kubectl get pods -n ${NAMESPACE} -l job-name=experiment-60q -o jsonpath='{.items[0].metadata.name}') -c experiment -- cat /results/experiment_60q/experiment_progress.log"
