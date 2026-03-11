#!/bin/bash
set -e

NAMESPACE="mltraining-dev"
CONTEXT="sfc-or-dev-meta-k8s-2"

echo "=== Qwen 2.5 72B Cross-Model Validation Deployment ==="

echo "Step 1: Clean up old jobs to free GPU quota..."
kubectl --context="${CONTEXT}" delete job llama-70b-hidden-states -n "${NAMESPACE}" --ignore-not-found
kubectl --context="${CONTEXT}" delete job experiment-60q -n "${NAMESPACE}" --ignore-not-found
kubectl --context="${CONTEXT}" delete job qwen-cross-model -n "${NAMESPACE}" --ignore-not-found

echo "Step 2: Creating qwen-cross-model-code ConfigMap from source files..."
kubectl --context="${CONTEXT}" create configmap qwen-cross-model-code \
  --namespace="${NAMESPACE}" \
  --from-file=run_cross_model_experiment.py=../../hotpotqa/run_cross_model_experiment.py \
  --from-file=agent.py=../../hotpotqa/agent.py \
  --from-file=tools.py=../../hotpotqa/tools.py \
  --from-file=cross_model_20q.json=../../hotpotqa/cross_model_20q.json \
  --dry-run=client -o yaml | kubectl --context="${CONTEXT}" apply -f -

echo "Step 3: Applying PVC..."
kubectl --context="${CONTEXT}" apply -f qwen-cross-model-pvc.yaml

echo "Step 4: Applying Job..."
kubectl --context="${CONTEXT}" apply -f qwen-cross-model-job.yaml

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Monitor with:"
echo "  kubectl --context=${CONTEXT} get pods -n ${NAMESPACE} -l job-name=qwen-cross-model -w"
echo "  kubectl --context=${CONTEXT} logs -n ${NAMESPACE} -l job-name=qwen-cross-model -c qwen-server -f"
echo "  kubectl --context=${CONTEXT} logs -n ${NAMESPACE} -l job-name=qwen-cross-model -c experiment -f"
echo ""
echo "When done, copy results:"
echo "  POD=\$(kubectl --context=${CONTEXT} get pods -n ${NAMESPACE} -l job-name=qwen-cross-model -o jsonpath='{.items[0].metadata.name}')"
echo "  kubectl --context=${CONTEXT} cp ${NAMESPACE}/\${POD}:/results/qwen_cross_model ../../hotpotqa/qwen_cross_model_results -c experiment"
