# this script starts the cluster on microk8s
# it assumes that the cluster is already installed
# and that the user has access to the cluster
# it also assumes that the user has access to the
# docker registry
## create the namespace iotstack
kubectl create namespace iotstack
kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.13.7/config/manifests/metallb-native.yaml
# check if skaffold is installed and install it if not
if ! [ -x "$(command -v skaffold)" ];
then
    echo "skaffold could not be found"
    echo "installing skaffold"
    curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64
    chmod +x skaffold
    sudo mv skaffold /usr/local/bin
fi

# run skaffold to build the application and check for errors
skaffold build

# skaffold dev --namespace iotstack --status-check
skaffold run --namespace iotstack --status-check \
                                --kube-context kind-diona
kubectl apply -f metallb.yaml

# check if the application is running
kubectl get pods --namespace iotstack

