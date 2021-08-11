using AccuracyAtTopDual
using DatasetProvider
using EvalMetrics
using Plots

# data preparation
T = Float32;
dataset = Dataset(
    DatasetProvider.Spambase;
    asmatrix = true,
    shuffle = true,
    binarize = true,
    poslabels = 1,
    seed = 123,
);
train, test = load(TrainTest(0.8), dataset);
Xtrain, ytrain = Array{T}(train[1]), train[2];
Xtest, ytest = Array{T}(test[1]), test[2];

# training
model = TopPush(; C = 1, ϑ = 1, surrogate = Quadratic, T)
kernel = KernelType(Gaussian; γ = 1, scale = true, precomputed = false, T)

h = solve!(model, Xtrain, ytrain, kernel; maxiter = 20000, seed = 123, ε = 1e-4)

# evaluation
s_train = h.solution[:train].s;
s_test = predict(model, Xtrain, ytrain, kernel, Xtest);

plot(
    rocplot(ytrain, s_train; title = "Train"),
    rocplot(ytest, s_test; title = "Test");
    xlims = (1e-5, 1),
    xscale = :log10,
    fill = false,
    size = (700, 400),
)
