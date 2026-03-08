import torch, gpytorch, time, sys

torch.manual_seed(42)
n, d, m = 256, 26, 128
X = torch.randn(n, d)
y = torch.randint(0, 2, (n,)).float()

class SVGP(gpytorch.models.ApproximateGP):
    def __init__(self, ip):
        vd = gpytorch.variational.CholeskyVariationalDistribution(ip.size(0))
        vs = gpytorch.variational.VariationalStrategy(self, ip, vd, learn_inducing_locations=True)
        super().__init__(vs)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

ip = X[:m].clone()
model = SVGP(ip)
lik = gpytorch.likelihoods.BernoulliLikelihood()
model.train(); lik.train()
opt = torch.optim.Adam(list(model.parameters()) + list(lik.parameters()), lr=0.01)
mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=n)

print("Starting test...", flush=True)
for i in range(3):
    t0 = time.time()
    opt.zero_grad()
    out = model(X)
    loss = -mll(out, y)
    loss.backward()
    opt.step()
    elapsed = time.time() - t0
    print(f"batch {i}: loss={loss.item():.4f} time={elapsed:.2f}s", flush=True)
print("PASSED")
