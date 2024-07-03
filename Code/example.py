import numpy as np
import torch
from pymor.basic import *
import matplotlib.pyplot as plt
from pymor.algorithms.timestepping import TimeStepper
from numpy import linalg as LA
import time

plt.style.use('style.mplstyle')

# global values
trainingTime = 0
FOMEvaluations = 0
surrogateEvaluations = 0
innerIterations = 0


class CrankNicolsonTimeStepper(TimeStepper):
    """Crank-Nicolson time-stepper.

    Solves equations of the form ::

        M * d_t u + A(u, mu, t) = F(mu, t).

    Parameters
    ----------
    nt
        The number of time-steps the time-stepper will perform.
    solver_options
        The |solver_options| used to invert `M + dt*A`.
        The special values `'mass'` and `'operator'` are
        recognized, in which case the solver_options of
        M (resp. A) are used.
    """

    def __init__(self, nt, solver_options='operator'):
        self.__auto_init(locals())

    def estimate_time_step_count(self, initial_time, end_time):
        return self.nt

    def iterate(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None, num_values=None):

        from pymor.operators.interface import Operator
        from pymor.parameters.base import Mu
        from pymor.vectorarrays.interface import VectorArray
        from pymor.parameters.functionals import ConstantParameterFunctional
        from pymor.algorithms.timestepping import _depends_on_time

        A, F, M, U0, t0, t1, nt = operator, rhs, mass, initial_data, initial_time, end_time, self.nt

        assert isinstance(A, Operator)
        assert isinstance(F, (type(None), Operator, VectorArray))
        assert isinstance(M, (type(None), Operator))
        assert A.source == A.range
        num_values = num_values or nt + 1
        dt = (t1 - t0) / nt
        DT = (t1 - t0) / (num_values - 1)

        if F is None:
            F_time_dep = False
        elif isinstance(F, Operator):
            assert F.source.dim == 1
            assert F.range == A.range
            F_time_dep = _depends_on_time(F, mu)
            if not F_time_dep:
                dt_F0 = F.as_vector(mu) * dt * 0.5
                dt_F1 = F.as_vector(mu) * dt * 0.5
        else:
            assert len(F) == 1
            assert F in A.range
            F_time_dep = False
            dt_F0 = F * dt * 0.5
            dt_F1 = F * dt * 0.5

        if M is None:
            from pymor.operators.constructions import IdentityOperator
            M = IdentityOperator(A.source)

        assert A.source == M.source == M.range
        assert not M.parametric
        assert U0 in A.source
        assert len(U0) == 1

        num_ret_values = 1
        yield U0, t0

        options = (A.solver_options if self.solver_options == 'operator' else
                   M.solver_options if self.solver_options == 'mass' else
                   self.olver_options)
        M_dt_A_p = (M + A * dt * ConstantParameterFunctional(1/2)).with_(solver_options=options)
        if not _depends_on_time(M_dt_A_p, mu):
            M_dt_A_p = M_dt_A_p.assemble(mu)
        M_dt_A_m = (M - A * dt * ConstantParameterFunctional(1/2)).with_(solver_options=options)
        if not _depends_on_time(M_dt_A_m, mu):
            M_dt_A_m = M_dt_A_m.assemble(mu)

        t = t0
        U = U0.copy()
        if mu is None:
            mu = Mu()

        for n in range(nt):
            mu0 = mu.with_(t=t)
            t += dt
            mu1 = mu.with_(t=t)
            rhs = M_dt_A_m.apply(U, mu=mu0)
            if F_time_dep:
                dt_F0 = F.as_vector(mu0) * dt * 0.5
                dt_F1 = F.as_vector(mu1) * dt * 0.5

            if F:
                rhs += dt_F0 + dt_F1
            U = M_dt_A_p.apply_inverse(rhs, mu=mu1, initial_guess=U)
            while t - t0 + (min(dt, DT) * 0.5) >= num_ret_values * DT:
                num_ret_values += 1
                yield U, t


def parabolic_equation(q, T, grid_intervals=50, nt=10):
    """Return a parabolic equation.

    Parameters
    ----------
    q
        Control function q
    T
        Stopping time
    grid_intervals
        Number of intervals in each direction of the two-dimensional |RectDomain|.
    nt
        Number of time steps.

    Returns
    -------
    fom
        Parabolic equation problem as an |InstationaryModel|.
    """
    from pymor.analyticalproblems.domaindescriptions import RectDomain
    from pymor.analyticalproblems.elliptic import StationaryProblem
    from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction
    from pymor.analyticalproblems.instationary import InstationaryProblem
    from pymor.discretizers.builtin import discretize_instationary_cg

    # setup analytical problem
    domain = RectDomain()

    diffusion = ConstantFunction(1, 2)

    f = ExpressionFunction('-pi**4*exp(a[0]*pi**2*T)*sin(pi*x[0])*sin(pi*x[1])', dim_domain=2, parameters={'a': 1}, values={'T': T})

    rhs = f+q

    problem = InstationaryProblem(

        StationaryProblem(domain, rhs, diffusion),

        T=T,

        initial_data=ExpressionFunction('-1/(2+a[0])*pi**2*sin(pi*x[0])*sin(pi*x[1])', dim_domain=2, parameters={'a': 1})
    )

    # discretize using continuous finite elements
    fom, data = discretize_instationary_cg(analytical_problem=problem, diameter=1./grid_intervals, time_stepper=CrankNicolsonTimeStepper(nt=nt))

    return fom, data


# objective functional
def J(param, q_shape, a, T, grid_intervals=50, nt=10):
    global FOMEvaluations
    FOMEvaluations += 1
    alpha = np.pi**(-4)
    # setting the control function q
    # q_shape = [ExpressionFunction('sin(pi*x[0])*sin(pi*x[1])', dim_domain=2)]
    nb = len(q_shape)
    assert len(param) == nb*(nt+1)
    q = ExpressionFunction('0', dim_domain=2)
    for j in range(nb):
        paramFunc = ExpressionFunction('param*(t[0]<=0)*(0<=t[0])', dim_domain=2, parameters={'t': 1}, values={'param': param[0]})
        for i in range(nt):
            paramFunc = paramFunc+ExpressionFunction('((j+1-nt/T*t[0])*param1+(nt/T*t[0]-j)*param2)*(nt/T*t[0]<=j+1)*(j<nt/T*t[0])', dim_domain=2, parameters={'t': 1}, values={'j': i, 'param1': param[i+j*(nt+1)], 'param2': param[i+1+j*(nt+1)], 'nt': nt, 'T': T})
        q = q+paramFunc*q_shape[j]
    # solving the parabolic equation
    fom, data = parabolic_equation(q, T, grid_intervals, nt)
    u = fom.solve({'a': a})
    # calculating the objective functional value
    uHat = ExpressionFunction('(a[0]**2-5)/(2+a[0])*pi**2*exp(a[0]*pi**2*t[0])*sin(pi*x[0])*sin(pi*x[1])+2*pi**2*exp(a[0]*pi**2*T)*sin(pi*x[0])*sin(pi*x[1])', dim_domain=2, parameters={'a': 1, 't': 1}, values={'T': T})
    uMinusUHath = u.space.empty(reserve=nt+1)
    qh = u.space.empty(reserve=nt+1)
    grid = data['grid']
    for i in range(nt+1):
        qh.append(u.space.from_numpy(q.evaluate(grid.centers(grid.dim), q.parameters.parse({'t': i/nt*T}))))
        uMinusUHath.append(u[i][0]-u.space.from_numpy(uHat.evaluate(grid.centers(grid.dim), uHat.parameters.parse({'a': a, 't': i/nt*T}))))
    y1Int = 0
    y2Int = 0
    for t in range(nt):
        y1Int += T/(3*nt) * (uMinusUHath[t].norm2(product=fom.l2_product) + uMinusUHath[t].inner(uMinusUHath[t+1], product=fom.l2_product) + uMinusUHath[t+1].norm2(product=fom.l2_product))
        y2Int += T/(3*nt) * (qh[t].norm2(product=fom.l2_product) + qh[t].inner(qh[t+1], product=fom.l2_product) + qh[t+1].norm2(product=fom.l2_product))
    y1Int = y1Int[0][0]
    y2Int = y2Int[0][0]
    out = 1/2*y1Int+alpha/2*y2Int
    return out, fom, data, u


def L_BFGS_B_minimizer(init, eps, k_1, a, T, grid_intervals, nt, q_shape):
    from scipy.optimize import minimize
    startLBFGSB = time.time()
    opt = minimize(lambda mu: J(mu, q_shape, a, T, grid_intervals, nt)[0], init, method='L-BFGS-B', options={'maxiter': k_1, 'gtol': eps})
    endLBFGSB = time.time()
    LBFGSBTime = (endLBFGSB-startLBFGSB)/60
    return opt.x, opt.fun, LBFGSBTime


def lineSearch(F, q_k, F_k, d_k, beta, r, eps, nu_1, proj):
    beta_k = beta
    q_k_next = proj(q_k+beta_k*d_k)
    F_k_next = F(q_k_next)
    nu = 0
    while (F_k_next-F_k <= eps and nu < nu_1):
        beta_k = r*beta_k
        q_k_next = proj(q_k+beta_k*d_k)
        F_k_next = F(q_k_next)
        nu = nu+1
    return q_k_next, F_k_next


def initCov(N_q, var, correlationCoeff, nt, nb):
    C_k_next = np.zeros((N_q, N_q))
    for k in range(nb):
        for i in range(nt+1):
            for j in range(nt+1):
                C_k_next[k*(nt+1)+i, k*(nt+1)+j] = var[k]**2*correlationCoeff**(np.abs(i-j))*1/(1-correlationCoeff**2)
    return C_k_next


def updateCov(q_k, T_k, C_k, F_k, beta):
    N_q = len(q_k)
    N = len(T_k)
    d_k_cov = np.zeros((N_q, N_q))
    for m in range(N):
        d_k_cov = d_k_cov+(T_k[m][1]-F_k)*((T_k[m][0]-q_k).reshape((N_q, 1))*(T_k[m][0]-q_k).reshape((1, N_q))-C_k)
    d_k_cov = d_k_cov / N
    C_k_diag = np.zeros(N_q)
    d_k_cov_diag = np.zeros(N_q)
    for i in range(N_q):
        C_k_diag[i] = C_k[i, i]
        d_k_cov_diag[i] = d_k_cov[i, i]
    beta_iter = beta
    while (np.min(C_k_diag+beta_iter*d_k_cov_diag) <= 0):
        beta_iter /= 2
    return C_k + beta_iter*d_k_cov


def optStep(F, q_k, N, k, T_k, C_k, F_k, beta_1, beta_2, r, eps, nu_1, var, correlationCoeff, nt, nb, proj=lambda mu: mu, FOM=True):
    global innerIterations
    if not FOM:
        innerIterations += 1
    N_q = len(q_k)
    C_k_next = np.zeros((N_q, N_q))
    if k == 0:
        if C_k is None:
            C_k_next = initCov(len(q_k), var, correlationCoeff, nt, nb)
        else:
            C_k_next = C_k.copy()
    else:
        C_k_next = updateCov(q_k, T_k, C_k, F_k, beta_2)
    sample = np.random.multivariate_normal(q_k, C_k_next, size=N)
    T_k_next = []
    for i in range(N):
        T_k_next.append([proj(sample[i]), F(proj(sample[i]))])
        # T_k_next.append([sample[i], F(sample[i])])
    C_F = np.zeros(N_q)
    for m in range(N):
        C_F = C_F+(T_k_next[m][0]-q_k)*(T_k_next[m][1]-F_k)
    C_F = 1/(N-1)*C_F
    d_k = np.zeros(N_q)
    q_k_next, F_k_next = q_k.copy(), F_k
    if not np.all(C_F == 0):
        d_k = C_F/np.max(np.abs(C_F))
        q_k_next, F_k_next = lineSearch(F, q_k, F_k, d_k, beta_1, r, eps, nu_1, proj)

    return q_k_next, T_k_next, C_k_next, F_k_next


def enOpt(F, q_0, N, eps, k_1, beta_1, beta_2, r, nu_1, var, correlationCoeff, T, nt, nb, proj=lambda mu: mu, Cov=None, FOM=True):
    F_k_prev = F(q_0)
    functionValues = [F_k_prev]
    q_k, T_k, C_k, F_k = optStep(F, q_0, N, 0, [], Cov, F_k_prev, beta_1, beta_2, r, eps, nu_1, var, correlationCoeff, nt, nb, proj, FOM)
    functionValues.append(F_k)
    if (not FOM and showInnerIterationPlots) or (FOM and showOuterIterationPlots):
        t = np.linspace(0, T, nt+1)
        for i in range(nb):
            plt.plot(t, q_k[i*(nt+1):(i+1)*(nt+1)], label=r'$\mathbf{q}_{k}$')
            plt.title('EnOpt: shape functional {}, iteration {}'.format(i+1, 1))
            plt.xlabel('Time')
            plt.ylabel('Control variable')
            plt.legend()
            plt.show()
    k = 1
    while (F_k > F_k_prev+eps and k < k_1):
        F_k_prev = F_k
        q_k, T_k, C_k, F_k = optStep(F, q_k, N, k, T_k, C_k, F_k, beta_1, beta_2, r, eps, nu_1, var, correlationCoeff, nt, nb, proj, FOM)
        functionValues.append(F_k)
        if (not FOM and showInnerIterationPlots) or (FOM and showOuterIterationPlots):
            t = np.linspace(0, T, nt+1)
            for i in range(nb):
                plt.plot(t, q_k[i*(nt+1):(i+1)*(nt+1)], label=r'$\mathbf{q}_{k}$')
                plt.title('EnOpt: shape functional {}, iteration {}'.format(i+1, k+1))
                plt.xlabel('Time')
                plt.ylabel('Control variable')
                plt.legend()
                plt.show()
        k = k+1
    return q_k, functionValues


def FOM_EnOpt(q_0, N, eps, k_1, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt, q_shape):
    q, FOMValues = enOpt(lambda mu: -J(mu, q_shape, a, T, grid_intervals, nt)[0], q_0, N, eps, k_1, beta_1, beta_2, r, nu_1, var, correlationCoeff, T, nt, len(q_shape))
    return q, -np.array(FOMValues)


def projection(x, u_k, d_k):
    upp = u_k + d_k
    low = u_k - d_k
    return np.maximum(np.minimum(x, upp), low)


def testDNN(DNN, x, y, loss_fn):
    DNN.eval()
    with torch.inference_mode():
        pred = DNN(x).reshape(len(y))
        loss = loss_fn(pred, y)
        return loss.cpu().detach().numpy()


def trainDNN(DNN, x_train, y_train, x_val, y_val, loss_fn, optimizer, epochs, earlyStop):
    wait = 0
    minimal_validation_loss = testDNN(DNN, x_val, y_val, loss_fn)
    val_iteration = [minimal_validation_loss]
    torch.save(DNN.state_dict(), 'checkpoint.pth')
    # Training
    for epoch in np.arange(1, epochs+1):
        DNN.train()

        def closure():
            y_pred = DNN(x_train).reshape(len(y_train))
            loss = loss_fn(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            return loss
        optimizer.step(closure)

        # Testing
        val_loss = testDNN(DNN, x_val, y_val, loss_fn)
        val_iteration.append(val_loss)
        if epoch % 1 == 0:
            print('Epoch: {}, validation loss: {}'.format(epoch, val_loss))
        if val_loss < minimal_validation_loss:
            wait = 0
            minimal_validation_loss = val_loss
            torch.save(DNN.state_dict(), 'checkpoint.pth')
        else:
            wait += 1
        if wait >= earlyStop:
            DNN.load_state_dict(torch.load('checkpoint.pth'))
            if showPlots:
                plt.plot(range(len(val_iteration)), val_iteration)
                plt.title('DNN training: validation loss')
                plt.xlabel('Training iteration')
                plt.ylabel('Validation loss')
                plt.show()
                if len(val_iteration) > earlyStop+5:
                    plt.plot(np.arange(len(val_iteration)-(earlyStop+5), len(val_iteration)), val_iteration[(len(val_iteration)-(earlyStop+5)):])
                    plt.title('DNN training: validation loss, last iterations')
                    plt.xlabel('Training iteration')
                    plt.ylabel('Validation loss')
                    plt.show()
            return val_iteration
    if showPlots:
        plt.plot(range(len(val_iteration)), val_iteration, label='DNN training: validation loss')
        plt.title('DNN training: validation loss')
        plt.xlabel('Training iteration')
        plt.ylabel('Validation loss')
        plt.show()
        if len(val_iteration) > earlyStop+5:
            plt.plot(np.arange(len(val_iteration)-(earlyStop+5), len(val_iteration)), val_iteration[(len(val_iteration)-(earlyStop+5)):])
            plt.title('DNN training: validation loss, last iterations')
            plt.xlabel('Training iteration')
            plt.ylabel('Validation loss')
            plt.show()
    return val_iteration


def constructDNN(sample, V_DNN, minIn, maxIn):
    from pymor.models.neural_network import FullyConnectedNN
    from torch import nn
    DNNStructure = V_DNN[0]
    activFunc = V_DNN[1]
    restarts = V_DNN[2]
    epochs = V_DNN[3]
    earlyStop = V_DNN[4]
    trainFrac = V_DNN[5]
    learning_rate = V_DNN[6]
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # scaling
    normSample = np.zeros((len(sample), len(sample[0][0])))
    normVal = np.zeros(len(sample))
    for i in range(len(sample)):
        normSample[i, :] = sample[i][0]
        normVal[i] = sample[i][1]
    minOut = np.min(normVal)
    maxOut = np.max(normVal)
    normSample = (normSample-minIn)/(maxIn-minIn)
    normVal = (normVal-minOut)/(maxOut-minOut)
    x = torch.from_numpy(normSample).to(torch.float32)
    y = torch.from_numpy(normVal).to(torch.float32)
    # Create train/test split
    trainSplit = int(trainFrac * len(x))
    x_train, y_train = x[:trainSplit], y[:trainSplit]
    x_val, y_val = x[trainSplit:], y[trainSplit:]

    DNN = FullyConnectedNN(DNNStructure, activation_function=activFunc)

    DNN.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.LBFGS(DNN.parameters(), lr=learning_rate, line_search_fn='strong_wolfe')

    x_train = x_train.to(device)
    x_val = x_val.to(device)
    y_train = y_train.to(device)
    y_val = y_val.to(device)
    val_iteration = trainDNN(DNN, x_train, y_train, x_val, y_val, loss_fn, optimizer, epochs, earlyStop)
    DNN_eval = testDNN(DNN, x_val, y_val, loss_fn)
    val_loss = [DNN_eval]
    for i in range(restarts):
        DNN_i = FullyConnectedNN(DNNStructure, activation_function=activFunc)
        DNN_i.to(device)
        optimizer = torch.optim.LBFGS(DNN_i.parameters(), lr=learning_rate, line_search_fn='strong_wolfe')
        val_iteration_i = trainDNN(DNN_i, x_train, y_train, x_val, y_val, loss_fn, optimizer, epochs, earlyStop)
        DNN_i_eval = testDNN(DNN_i, x_val, y_val, loss_fn)
        val_loss.append(DNN_i_eval)
        if DNN_i_eval < DNN_eval:
            DNN_eval = DNN_i_eval
            DNN = DNN_i
            val_iteration = val_iteration_i
    if showPlots:
        plt.bar(range(len(val_loss)), val_loss)
        plt.title('Validation loss per restart')
        plt.xlabel('Restart')
        plt.ylabel('Validation loss')
        plt.show()

    if showPlots:
        with torch.inference_mode():

            sortedDNNOutputs = []
            sortedFOMOutputs = []
            sortSample = []
            for i in range(len(normSample)):
                sortSample.append([normSample[i, :], normVal[i]])
            for i in range(len(normSample)):
                jMin = 0
                sampleMin = sortSample[0][1]
                for j in range(len(sortSample)):
                    if sortSample[j][1] < sampleMin:
                        jMin = j
                        sampleMin = sortSample[j][1]
                sortedFOMOutputs.append(sampleMin)
                sortedDNNOutputs.append(DNN(torch.from_numpy(sortSample[jMin][0]).to(torch.float32).to(device)).cpu().detach().numpy()[0])
                sortSample.pop(jMin)
            plt.bar(range(len(sortedDNNOutputs)), sortedDNNOutputs)
            plt.title('Inputs sorted by their FOM objective functional values')
            plt.xlabel('Entry of the sorted list')
            plt.ylabel('DNN output value')
            plt.show()
            plt.bar(range(len(sortedFOMOutputs)), sortedFOMOutputs)
            plt.title('Inputs sorted by their FOM objective functional values')
            plt.xlabel('Entry of the sorted list')
            plt.ylabel('FOM objective functional value')
            plt.show()

    def f(x_inp):
        global surrogateEvaluations
        surrogateEvaluations += 1
        scaledInput = torch.from_numpy((x_inp-minIn)/(maxIn-minIn)).to(torch.float32).to(device)
        with torch.inference_mode():
            scaledOutput = DNN(scaledInput).cpu()
        print('Scaled DNN output: {}'.format(scaledOutput.numpy()[0]))
        return scaledOutput.numpy()[0]*(maxOut-minOut)+minOut
    DNN_train_loss = testDNN(DNN, x_train, y_train, loss_fn)
    return f, DNN_eval, DNN_train_loss, val_iteration


def AML_EnOpt(F, q_0, N, eps_o, eps_i, k_1_o, k_1_i, k_tr, V_DNN, delta_init, beta_1, beta_2, r, nu_1, var, correlationCoeff, T, nt, nb):
    global trainingTime
    global addDNNStruct
    # initialize the surrogate functional
    surrogateValue = 0
    surrogateEval = []
    surrogateTrain = []
    N_q = len(q_0)
    F_k = F(q_0)
    FOMValues = [F_k]
    surrogateValuesOuterIteration = []
    F_k_next = F_k
    q_k_tilde, T_k, C_k, F_k_tilde = optStep(F, q_0, N, 0, [], None, F_k, beta_1, beta_2, r, eps_o, nu_1, var, correlationCoeff, nt, nb)
    t = np.linspace(0, T, num=nt+1)
    if showPlots:
        for i in range(nb):
            plt.plot(t, q_k_tilde[i*(nt+1):(i+1)*(nt+1)], label=r'$\tilde{\mathbf{q}}_k$')
            plt.title('AML-EnOpt: shape functional {}, iteration {}'.format(i+1, 0))
            plt.xlabel('Time')
            plt.ylabel('Control variable')
            plt.legend()
            plt.show()
    k = 1
    q_k = q_0
    q_k_next = q_k.copy()
    failures_iter = []
    failures = 0
    delta = delta_init
    while (F_k_tilde > F_k+eps_o and k < k_1_o):
        T_k_x = np.zeros((N, N_q))
        for i in range(N):
            T_k_x[i, :] = T_k[i][0]
        minIn = np.zeros(N_q)
        maxIn = np.zeros(N_q)
        for i in range(N_q):
            minIn[i] = np.min(T_k_x[:, i])
            maxIn[i] = np.max(T_k_x[:, i])

        d_k = np.abs(q_k-q_k_tilde)

        DNN_eval = 0
        tr = 1
        while F_k_next <= F_k+eps_o:
            assert tr <= k_tr
            startTraining = time.time()
            F_ML_k, DNN_eval, DNN_train_loss, val_iteration = constructDNN(T_k, V_DNN, minIn, maxIn)
            endTraining = time.time()
            trainingTime = trainingTime + (endTraining - startTraining)
            F_ML_k_q_k = F_ML_k(q_k)
            if inspectDNNStructures:
                F_ML_k_list = [F_ML_k_q_k-F_k]
                DNN_eval_list = [DNN_eval]
                val_iteration_list = []
                V_DNN_i = V_DNN.copy()
                for i in range(len(addDNNStruct)):
                    V_DNN_i[0] = addDNNStruct[i]
                    F_ML_k_i, DNN_eval_i, DNN_train_loss_i, val_iteration_i = constructDNN(T_k, V_DNN_i, minIn, maxIn)
                    F_ML_k_list.append(F_ML_k_i(q_k)-F_k)
                    DNN_eval_list.append(DNN_eval_i)
                    val_iteration_list.append(val_iteration_i)
                allDNNStruct = [f'{V_DNN[0][1]}']
                for i in addDNNStruct:
                    allDNNStruct.append(f'{i[1]}')
                plt.bar(allDNNStruct, F_ML_k_list)
                plt.title('Difference between the surrogate functional value and the FOM objective functional value \n for different DNN structures during TR-method {}, outer iteration {}'.format(tr, k))
                plt.xlabel('DNN Structure')
                plt.ylabel('Surrogate functional value')
                plt.show()
                plt.bar(allDNNStruct, DNN_eval_list)
                plt.title('Validation loss for different DNN structures \n during TR-method {}, outer iteration {}'.format(tr, k))
                plt.xlabel('DNN Structure')
                plt.ylabel('Validation loss')
                plt.show()
                plt.plot(range(len(val_iteration)), val_iteration, label='DNN structure: {}'.format(V_DNN[0][1:-1]))
                for i in range(len(val_iteration_list)):
                    plt.plot(range(len(val_iteration_list[i])), val_iteration_list[i], label='DNN structure: {}'.format(addDNNStruct[i][1:-1]))
                plt.title('Validation loss progression for different DNN structures \n during TR-method {}, outer iteration {}'.format(tr, k))
                plt.xlabel('Training iteration')
                plt.ylabel('Validation loss')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                plt.show()
                minimum_eval_length = len(val_iteration)-(V_DNN[4]+1)
                for i in range(len(val_iteration_list)):
                    eval_length_i = len(val_iteration_list[i])-(V_DNN[4]+1)
                    if eval_length_i < minimum_eval_length:
                        minimum_eval_length = eval_length_i
                if minimum_eval_length > 0:
                    plt.plot(np.arange(minimum_eval_length, len(val_iteration)), val_iteration[minimum_eval_length:], label='DNN structure: {}'.format(V_DNN[0][1:-1]))
                    for i in range(len(val_iteration_list)):
                        plt.plot(np.arange(minimum_eval_length, len(val_iteration_list[i])), val_iteration_list[i][minimum_eval_length:], label='DNN structure: {}'.format(addDNNStruct[i][1:-1]))
                    plt.title('Last iterations of the validation loss progression for different \n DNN structures during TR-method {}, outer iteration {}'.format(tr, k))
                    plt.xlabel('Training iteration')
                    plt.ylabel('Validation loss')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                    plt.show()
            deltaList = [delta]
            trustRegionFlag = True
            while trustRegionFlag:
                d_k_iter = delta * d_k
                q_k_next, surrogateValuesInnerIteration = enOpt(F_ML_k, q_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var, correlationCoeff, T, nt, nb, proj=lambda mu: projection(mu, q_k, d_k_iter), Cov=C_k, FOM=False)
                F_k_next = F(q_k_next)
                surrogateValue = surrogateValuesInnerIteration[-1]
                rho_k = (F_k_next-F_k)/(surrogateValue-F_ML_k_q_k)
                if rho_k < 0.25:
                    delta *= 0.25
                else:
                    if rho_k > 0.75 and np.any(np.abs(q_k-q_k_next)-d_k_iter == 0):
                        delta *= 2
                if rho_k > 0:
                    trustRegionFlag = False
                deltaList.append(delta)
            if showPlots:
                plt.plot(range(len(deltaList)), deltaList)
                plt.title(r'$\delta$'+' during TR-method {}, outer iteration {}'.format(tr, k))
                plt.xlabel('TR iteration')
                plt.ylabel(r'$\delta$')
                plt.show()
            tr += 1

        failures += tr-2
        surrogateEval.append(DNN_eval)
        surrogateTrain.append(DNN_train_loss)
        FOMValues.append(F_k_next)
        surrogateValuesOuterIteration.append(surrogateValue)
        failures_iter.append(failures)
        if showOuterIterationPlots:
            for i in range(nb):
                plt.plot(t, q_k_next[i*(nt+1):(i+1)*(nt+1)], label=r'$\mathbf{q}^\mathrm{next}_k$')
                plt.title('AML-EnOpt: shape functional {}, iteration {}'.format(i+1, k))
                plt.xlabel('Time')
                plt.ylabel('Control variable')
                plt.legend()
                plt.show()
        if showPlots:
            for i in range(nb):
                for j in range(len(T_k)):
                    plt.plot(t, T_k[j][0][i*(nt+1):(i+1)*(nt+1)])
                plt.title('Samples: shape functional {}, outer iteration {} '.format(i+1, k))
                plt.xlabel('Time')
                plt.ylabel('Control variable')
                plt.show()
            T_k_min_q = T_k[0][0].copy()
            T_k_min_dist = LA.norm(T_k_min_q-q_k)
            T_k_max_q = T_k[0][0].copy()
            T_k_max_dist = LA.norm(T_k_max_q-q_k)
            for i in range(len(T_k)):
                dist = LA.norm(T_k[i][0]-q_k)
                if dist < T_k_min_dist:
                    T_k_min_dist = dist
                    T_k_min_q = T_k[i][0].copy()
                if dist > T_k_max_dist:
                    T_k_max_dist = dist
                    T_k_max_q = T_k[i][0].copy()
            for i in range(nb):
                plt.plot(t, q_k[i*(nt+1):(i+1)*(nt+1)], label=r'$\mathbf{q}_k$')
                plt.plot(t, T_k_min_q[i*(nt+1):(i+1)*(nt+1)], label=r'$T_k$ min')
                plt.plot(t, T_k_max_q[i*(nt+1):(i+1)*(nt+1)], label=r'$T_k$ max')
                plt.plot(t, q_k_tilde[i*(nt+1):(i+1)*(nt+1)], label=r'$\tilde{\mathbf{q}}_k$')
                plt.plot(t, q_k_next[i*(nt+1):(i+1)*(nt+1)], label=r'$\mathbf{q}^\mathrm{next}_k$')
                plt.title('AML-EnOpt: shape functional {}, outer iteration {}'.format(i+1, k))
                plt.xlabel('Time')
                plt.ylabel('Control variable')
                plt.legend()
                plt.show()
                plt.plot(t, q_k[i*(nt+1):(i+1)*(nt+1)], label=r'$\mathbf{q}_k$')
                plt.plot(t, T_k_min_q[i*(nt+1):(i+1)*(nt+1)], label=r'$T_k$ min')
                plt.plot(t, T_k_max_q[i*(nt+1):(i+1)*(nt+1)], label=r'$T_k$ max')
                plt.plot(t, q_k_tilde[i*(nt+1):(i+1)*(nt+1)], label=r'$\tilde{\mathbf{q}}_k$')
                plt.title('AML-EnOpt: shape functional {}, outer iteration {}'.format(i+1, k))
                plt.xlabel('Time')
                plt.ylabel('Control variable')
                plt.legend()
                plt.show()
            names = [r'$T_k$ min', r'$T_k$ max', r'$\tilde{\mathbf{q}}_k$', r'$\mathbf{q}^\mathrm{next}_k$']
            values = [T_k_min_dist, T_k_max_dist, LA.norm(q_k_tilde-q_k), LA.norm(q_k_next-q_k)]
            plt.bar(names, values)
            plt.title(r'$L^2$-distance to $\mathbf{q}_k$,'+' iteration {}'.format(k))
            plt.xlabel('Control vector')
            plt.ylabel(r'$L^2$-distance to $\mathbf{q}_k$')
            plt.show()
            names = [r'$T_k$ min', r'$T_k$ max', r'$\tilde{\mathbf{q}}_k$']
            values = [T_k_min_dist, T_k_max_dist, LA.norm(q_k_tilde-q_k)]
            plt.bar(names, values)
            plt.title(r'$L^2$-distance to $\mathbf{q}_k$,'+' iteration {}'.format(k))
            plt.xlabel('Control vector')
            plt.ylabel(r'$L^2$-distance to $\mathbf{q}_k$')
            plt.show()
        q_k_tilde, T_k, C_k, F_k_tilde = optStep(F, q_k_next, N, k, T_k, C_k, F_k_next, beta_1, beta_2, r, eps_o, nu_1, var, correlationCoeff, nt, nb)
        if showPlots:
            covList = []
            for i in range(len(C_k)):
                covList.append(C_k[i, i])
            tCov = range(len(C_k))
            for i in range(nb):
                plt.bar(tCov[i*(nt+1):(i+1)*(nt+1)], covList[i*(nt+1):(i+1)*(nt+1)])
                plt.title('Variance: shape functional {}, outer iteration {}'.format(i+1, k))
                plt.xlabel('Time step')
                plt.ylabel('Variance')
                plt.show()
            for i in range(nb):
                plt.plot(t, q_k_tilde[i*(nt+1):(i+1)*(nt+1)], label=r'$\tilde{\mathbf{q}}_k$')
                plt.title('AML-EnOpt: shape functional {}, iteration {}'.format(i+1, k))
                plt.xlabel('Time')
                plt.ylabel('Control variable')
                plt.legend()
                plt.show()
        F_k = F_k_next
        q_k = q_k_next.copy()
        k = k+1
    if showPlots:
        plt.plot(np.arange(1, len(failures_iter)+1), failures_iter)
        plt.title('Number of cumulated failures of FOM objective functional value improvements')
        plt.xlabel('Iteration')
        plt.ylabel('Cumulated failures')
        plt.show()
    print('FOM objective functional value improvement failures: {}'.format(failures))
    return q_k, FOMValues, surrogateValuesOuterIteration, surrogateEval, surrogateTrain


def ROM_EnOpt(q_0, N, eps_o, eps_i, k_1_o, k_1_i, k_tr, V_DNN, delta_init, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt, q_shape):
    q, FOMValues, surrogateValuesOuterIteration, surrogateEval, surrogateTrain = AML_EnOpt(lambda mu: -J(mu, q_shape, a, T, grid_intervals, nt)[0], q_0, N, eps_o, eps_i, k_1_o, k_1_i, k_tr, V_DNN, delta_init, beta_1, beta_2, r, nu_1, var, correlationCoeff, T, nt, len(q_shape))
    return q, -np.array(FOMValues), -np.array(surrogateValuesOuterIteration), surrogateEval, surrogateTrain


#q_shape = [ExpressionFunction('1', dim_domain=2)]
#for i in np.arange(1, 2):
#    for j in np.arange(1, 2):
#        q_shape.append(ExpressionFunction('cos(2*pi*n*x[0])*cos(2*pi*m*x[1])', dim_domain=2, values={'n': i, 'm': j}))
#        q_shape.append(ExpressionFunction('cos(2*pi*n*x[0])*sin(2*pi*m*x[1])', dim_domain=2, values={'n': i, 'm': j}))
#        q_shape.append(ExpressionFunction('sin(2*pi*n*x[0])*cos(2*pi*m*x[1])', dim_domain=2, values={'n': i, 'm': j}))
#        q_shape.append(ExpressionFunction('sin(2*pi*n*x[0])*sin(2*pi*m*x[1])', dim_domain=2, values={'n': i, 'm': j}))
q_shape = [ExpressionFunction('sin(pi*x[0])*sin(pi*x[1])', dim_domain=2)]
T = 0.1
nt = 50
nb = len(q_shape)
grid_intervals = 50
a = -np.sqrt(5)
init = np.zeros(nb*(nt+1))-40


showOuterIterationPlots = False
showInnerIterationPlots = False
showPlots = False
inspectDNNStructures = False
N = 100
eps = 1e-14
eps_LBFGSB = 1e-7
k_1 = 1000
beta_1 = 1
beta_2 = 0.001
r = 0.5
nu_1 = 10
var = [0.01]
assert len(var) == len(q_shape)
correlationCoeff = 0.99


# optimized control functional using the AML EnOpt minimizer
delta_init = 100
eps_o = 1e-14
eps_i = 1e-14
k_1_o = k_1
k_1_i = k_1
k_tr = 5
# V_DNN: neurons per hidden layer, activation function (like torch.tanh), number of restarts, number of epochs, early stop, trainFrac, learning rate
# V_DNN = [[nb*(nt+1), 25, 25, 1], torch.tanh, 2, 1000, 15, 0.8, 1e-2]
V_DNN = [[nb*(nt+1), 25, 25, 1], torch.tanh, 10, 1000, 15, 0.8, 1e-2]
addDNNStruct = [[nb*(nt+1), 20, 20, 1], [nb*(nt+1), 25, 25, 1], [nb*(nt+1), 30, 30, 1], [nb*(nt+1), 35, 35, 1], [nb*(nt+1), 50, 50, 1], [nb*(nt+1), 100, 100, 1], [nb*(nt+1), 250, 250, 1], [nb*(nt+1), 500, 500, 1], [nb*(nt+1), 1000, 1000, 1]]


def evalFOM_EnOpt(init, N, eps, k_1, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt, q_shape):
    global FOMEvaluations
    nb = len(q_shape)
    method = 'FOM-EnOpt'
    FOMEvaluationsStart = FOMEvaluations
    startAlg = time.time()
    q, FOMValues = FOM_EnOpt(init, N, eps, k_1, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt, q_shape)
    endAlg = time.time()
    runTimeTotal = (endAlg-startAlg)/60
    outerIterationsTotal = len(FOMValues)-1
    FOMEvaluationsEnd = FOMEvaluations
    FOMEvaluationsTotal = FOMEvaluationsEnd-FOMEvaluationsStart
    for i in range(nb):
        plt.plot(np.linspace(0, T, num=nt+1), q[i*(nt+1):(i+1)*(nt+1)], label=r'$\mathbf{q}$')
        plt.title('FOM-EnOpt output: shape functional {}'.format(i+1))
        plt.xlabel('Time')
        plt.ylabel('Control variable')
        plt.legend()
        plt.show()
    print('Output: {}\n'.format(q))
    print('Method: {}\n'.format(method))
    print('FOM objective functional values: {}\n'.format(FOMValues))
    print('FOM objective functional output value: {}\n'.format(FOMValues[-1]))
    print('Number of outer iterations: {}\n'.format(outerIterationsTotal))
    print('Number of FOM evaluations: {}\n'.format(FOMEvaluationsTotal))
    print('Total run time (minutes): {}\n'.format(runTimeTotal))
    return q, method, FOMValues, outerIterationsTotal, FOMEvaluationsTotal, runTimeTotal


def evalROM_EnOpt(init, N, eps_o, eps_i, k_1_o, k_1_i, k_tr, V_DNN, delta_init, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt, q_shape):
    global innerIterations
    global FOMEvaluations
    global surrogateEvaluations
    global trainingTime
    nb = len(q_shape)
    method = 'AML-EnOpt'
    innerIterationsStart = innerIterations
    FOMEvaluationsStart = FOMEvaluations
    surrogateEvaluationsStart = surrogateEvaluations
    trainingTimeStart = trainingTime
    startAlg = time.time()
    q, FOMValues, surrogateValues, surrogateEval, surrogateTrain = ROM_EnOpt(init, N, eps_o, eps_i, k_1_o, k_1_i, k_tr, V_DNN, delta_init, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt, q_shape)
    endAlg = time.time()
    runTimeTotal = (endAlg-startAlg)/60
    outerIterationsTotal = len(FOMValues)-1
    innerIterationsEnd = innerIterations
    innerIterationsTotal = innerIterationsEnd-innerIterationsStart
    FOMEvaluationsEnd = FOMEvaluations
    FOMEvaluationsTotal = FOMEvaluationsEnd-FOMEvaluationsStart
    surrogateEvaluationsEnd = surrogateEvaluations
    surrogateEvaluationsTotal = surrogateEvaluationsEnd-surrogateEvaluationsStart
    trainingTimeEnd = trainingTime
    trainingTimeTotal = (trainingTimeEnd-trainingTimeStart)/60
    for i in range(nb):
        plt.plot(np.linspace(0, T, num=nt+1), q[i*(nt+1):(i+1)*(nt+1)], label=r'$\mathbf{q}$')
        plt.title('AML-EnOpt output: shape functional {}'.format(i+1))
        plt.xlabel('Time')
        plt.ylabel('Control variable')
        plt.legend()
        plt.show()
    print('Output: {}\n'.format(q))
    print('Method: {}\n'.format(method))
    print('FOM objective functional values: {}\n'.format(FOMValues))
    print('FOM objective functional output value: {}\n'.format(FOMValues[-1]))
    print('Surrogate functional values: {}\n'.format(surrogateValues))
    print('Number of outer iterations: {}\n'.format(outerIterationsTotal))
    print('Number of inner iterations: {}\n'.format(innerIterationsTotal))
    print('Number of FOM evaluations: {}\n'.format(FOMEvaluationsTotal))
    print('Number of surrogate evalutations: {}\n'.format(surrogateEvaluationsTotal))
    print('MSE validation loss :{}\n'.format(surrogateEval))
    print('MSE training loss :{}\n'.format(surrogateTrain))
    print('Training time (minutes): {}\n'.format(trainingTimeTotal))
    print('Total run time (minutes): {}\n'.format(runTimeTotal))
    return q, method, FOMValues, surrogateValues, outerIterationsTotal, innerIterationsTotal, FOMEvaluationsTotal, surrogateEvaluationsTotal, surrogateEval, surrogateTrain, trainingTimeTotal, runTimeTotal


def compareEnOpt(init, N, eps, eps_o, eps_i, eps_LBFGSB, k_1, k_1_o, k_1_i, V_DNN, delta_init, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt, q_shape, analytical=True):
    nb = len(q_shape)
    t = np.linspace(0, T, nt+1)
    q, method, FOMValues, outerIterationsTotal, FOMEvaluationsTotal, runTimeTotal = evalFOM_EnOpt(init, N, eps, k_1, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt, q_shape)
    qAML, methodAML, FOMValuesAML, surrogateValuesAML, outerIterationsTotalAML, innerIterationsTotalAML, FOMEvaluationsTotalAML, surrogateEvaluationsTotalAML, surrogateEvalAML, surrogateTrainAML, trainingTimeTotalAML, runTimeTotalAML = evalROM_EnOpt(init, N, eps_o, eps_i, k_1_o, k_1_i, k_tr, V_DNN, delta_init, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt, q_shape)
    qLBFGSB, FOMValueLBFGSB, runTimeTotalLBFGSB = L_BFGS_B_minimizer(init, eps_LBFGSB, k_1, a, T, grid_intervals, nt, q_shape)
    print('FOM-EnOpt output: {}\n'.format(q))
    print('FOM-EnOpt FOM objective functional values: {}\n'.format(FOMValues))
    print('FOM-EnOpt FOM objective functional output value: {}\n'.format(FOMValues[-1]))
    print('FOM-EnOpt number of outer iterations: {}\n'.format(outerIterationsTotal))
    print('FOM-EnOpt number of FOM evaluations: {}\n'.format(FOMEvaluationsTotal))
    print('FOM-EnOpt total run time (minutes): {}\n'.format(runTimeTotal))
    print('\n')
    print('AML-EnOpt output: {}\n'.format(qAML))
    print('AML-EnOpt FOM objective functional values: {}\n'.format(FOMValuesAML))
    print('AML-EnOpt FOM objective functional output value: {}\n'.format(FOMValuesAML[-1]))
    print('AML-EnOpt surrogate functional values: {}\n'.format(surrogateValuesAML))
    if len(surrogateValuesAML) > 0:
        print('AML-EnOpt last surrogate functional value: {}\n'.format(surrogateValuesAML[-1]))
    print('AML-EnOpt number of outer iterations: {}\n'.format(outerIterationsTotalAML))
    print('AML-EnOpt number of inner iterations: {}\n'.format(innerIterationsTotalAML))
    print('AML-EnOpt number of FOM evaluations: {}\n'.format(FOMEvaluationsTotalAML))
    print('AML-EnOpt number of surrogate evalutations: {}\n'.format(surrogateEvaluationsTotalAML))
    print('AML-EnOpt MSE validation loss :{}\n'.format(surrogateEvalAML))
    print('AML-EnOpt MSE training loss :{}\n'.format(surrogateTrainAML))
    print('AML-EnOpt training time (minutes): {}\n'.format(trainingTimeTotalAML))
    print('AML-EnOpt total run time (minutes): {}\n'.format(runTimeTotalAML))
    print('\n')
    print('L-BFGS-B output: {}\n'.format(qLBFGSB))
    print('L-BFGS-B FOM objective functional values: {}\n'.format(FOMValueLBFGSB))
    print('L-BFGS-B total run time (minutes): {}\n'.format(runTimeTotalLBFGSB))
    print('\n')
    qAnalytical = np.zeros((nb*(nt+1)))
    out, fom, data, u = J(q, q_shape, a, T, grid_intervals, nt)
    outAML, fomAML, dataAML, uAML = J(qAML, q_shape, a, T, grid_intervals, nt)
    fom.visualize(u, title='FOM-EnOpt u')
    fomAML.visualize(uAML, title='AML-EnOpt u')
    if analytical:
        assert nb == 1
        for i in range(nt+1):
            qAnalytical[i] = -np.pi**4*(np.exp(a*np.pi**2*i/nt*T)-np.exp(a*np.pi**2*T))
        uBar = ExpressionFunction('-1/(2+a[0])*pi**2*exp(a[0]*pi**2*t[0])*sin(pi*x[0])*sin(pi*x[1])', dim_domain=2, parameters={'a': 1, 't': 1})
        uBarh = u.space.empty(reserve=nt+1)
        for i in range(nt+1):
            uBarh.append(u.space.from_numpy(uBar.evaluate(data['grid'].centers(data['grid'].dim), uBar.parameters.parse({'a': a, 't': i/nt*T}))))
        err = uBarh-u
        supErr = np.max(err.sup_norm())
        relSupErr = np.max(err.sup_norm())/np.max(u.sup_norm())
        errAML = uBarh-uAML
        supErrAML = np.max(errAML.sup_norm())
        relSupErrAML = np.max(errAML.sup_norm())/np.max(u.sup_norm())
        print('Analytical FOM objective functional value: {}'.format(J(qAnalytical, q_shape, a, T, grid_intervals, nt)[0]))
        print('FOM-EnOpt sup-norm error of the state variable: {}'.format(supErr))
        print('FOM-EnOpt relative sup-norm error of the state variable: {}'.format(relSupErr))
        print('AML-EnOpt sup-norm error of the state variable: {}'.format(supErrAML))
        print('AMl-EnOpt relative sup-norm error of the state variable: {}'.format(relSupErrAML))
        fom.visualize(err, title='FOM-EnOpt: error of the state variable')
        fom.visualize(errAML, title='AML-EnOpt: error of the state variable')
    if len(surrogateValuesAML) > 0:
        plt.plot(range(len(FOMValues)), FOMValues, label='FOM-EnOpt: obj. functional value')
        plt.plot(np.arange(1, len(surrogateValuesAML)+1), surrogateValuesAML, label='AML-EnOpt: surr. functional value')
        plt.plot(range(len(FOMValuesAML)), FOMValuesAML, label='AML-EnOpt: obj. functional value')
        plt.title('Comparison of the functional values')
        plt.xlabel('Outer iteration')
        plt.ylabel('Functional value')
        plt.legend()
        plt.show()
        plt.plot(range(len(FOMValues)), FOMValues-FOMValues[-1], label='FOM-EnOpt: obj. functional value')
        plt.plot(np.arange(1, len(surrogateValuesAML)+1), surrogateValuesAML-FOMValues[-1], label='AML-EnOpt: surr. functional value')
        plt.plot(range(len(FOMValuesAML)), FOMValuesAML-FOMValues[-1], label='AML-EnOpt: obj. functional value')
        plt.yscale('symlog', linthresh=np.abs(FOMValues[-1]-FOMValuesAML[-1]))
        plt.title('Comparison of the functional values, translated and\n scaled with a symmetrical logarithmic scale')
        plt.xlabel('Outer iteration')
        plt.ylabel('Functional value')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
        plt.show()
        plt.plot(np.arange(1, len(surrogateValuesAML)+1), surrogateValuesAML, label='AML-EnOpt: surr. functional value', c='tab:orange')
        plt.plot(range(len(FOMValuesAML)), FOMValuesAML, label='AML-EnOpt: obj. functional value', c='tab:green')
        plt.title('Comparison of the functional values')
        plt.xlabel('Outer iteration')
        plt.ylabel('Functional value')
        plt.legend()
        plt.show()
        plt.plot(np.arange(1, len(surrogateValuesAML)+1), surrogateValuesAML-FOMValuesAML[-1], label='AML-EnOpt: surr. functional value', c='tab:orange')
        plt.plot(range(len(FOMValuesAML)), FOMValuesAML-FOMValuesAML[-1], label='AML-EnOpt: obj. functional value', c='tab:green')
        plt.yscale('symlog', linthresh=np.abs(FOMValuesAML[-1]-surrogateValuesAML[-1]))
        plt.title('Comparison of the functional values, translated and\n scaled with a symmetrical logarithmic scale')
        plt.xlabel('Outer iteration')
        plt.ylabel('Functional value')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
        plt.show()
        if len(FOMValuesAML)>5:
            plt.plot(np.arange(len(surrogateValuesAML)-4, len(surrogateValuesAML)+1), surrogateValuesAML[(len(surrogateValuesAML)-5):], label='AML-EnOpt: surr. functional value', c='tab:orange')
            plt.plot(np.arange(len(FOMValuesAML)-5, len(FOMValuesAML)), FOMValuesAML[(len(FOMValuesAML)-5):], label='AML-EnOpt: obj. functional value', c='tab:green')
            plt.title('Comparison of the functional values, last iterations')
            plt.xlabel('Outer iteration')
            plt.ylabel('Functional value')
            plt.legend()
            plt.show()
    for i in range(nb):
        plt.plot(t, q[i*(nt+1):(i+1)*(nt+1)], label='FOM-EnOpt')
        plt.plot(t, qAML[i*(nt+1):(i+1)*(nt+1)], label='AML-EnOpt')
        if analytical:
            plt.plot(t, qAnalytical, label='Analytical')
        plt.plot(t, init[i*(nt+1):(i+1)*(nt+1)], label='Initialization')
        plt.title('Optimal solutions: shape functional {}'.format(i+1))
        plt.xlabel('Time')
        plt.ylabel('Control variable')
        plt.legend()
        plt.show()
        plt.plot(t, q[i*(nt+1):(i+1)*(nt+1)], label='FOM-EnOpt')
        plt.plot(t, qAML[i*(nt+1):(i+1)*(nt+1)], label='AML-EnOpt')
        plt.plot(t, qLBFGSB[i*(nt+1):(i+1)*(nt+1)], label='L-BFGS-B')
        if analytical:
            plt.plot(t, qAnalytical, label='Analytical')
        plt.plot(t, init[i*(nt+1):(i+1)*(nt+1)], label='Initialization')
        plt.title('Optimal solutions: shape functional {}'.format(i+1))
        plt.xlabel('Time')
        plt.ylabel('Control variable')
        plt.legend()
        plt.show()
        plt.plot(t, q[i*(nt+1):(i+1)*(nt+1)], label='FOM-EnOpt')
        plt.title('FOM-EnOpt solution: shape functional {}'.format(i+1))
        plt.xlabel('Time')
        plt.ylabel('Control variable')
        plt.legend()
        plt.show()
        plt.plot(t, qAML[i*(nt+1):(i+1)*(nt+1)], label='AML-EnOpt')
        plt.title('AML-EnOpt solution: shape functional {}'.format(i+1))
        plt.xlabel('Time')
        plt.ylabel('Control variable')
        plt.legend()
        plt.show()
        plt.plot(t, qLBFGSB[i*(nt+1):(i+1)*(nt+1)], label='L-BFGS-B')
        plt.title('L-BFGS-B solution: shape functional {}'.format(i+1))
        plt.xlabel('Time')
        plt.ylabel('Control variable')
        plt.legend()
        plt.show()
        if analytical:
            plt.plot(t, qAnalytical, label='Analytical')
            plt.title('Analytical solution: shape functional {}'.format(i+1))
            plt.xlabel('Time')
            plt.ylabel('Control variable')
            plt.legend()
            plt.show()
    if analytical:
        plt.plot(t, q-qAnalytical, label='FOM-EnOpt')
        plt.plot(t, qAML-qAnalytical, label='AML-EnOpt')
        plt.title('Difference between the EnOpt and analytical solutions')
        plt.xlabel('Time')
        plt.ylabel('Control variable diff.')
        plt.legend()
        plt.show()
        print('FOM-EnOpt error: {}'.format(q-qAnalytical))
        print('Average absolute FOM-EnOpt error: {}'.format(np.sum(np.abs(q-qAnalytical))/len(q)))
        print('AML-EnOpt error: {}'.format(qAML-qAnalytical))
        print('Average absolute AML-EnOpt error: {}'.format(np.sum(np.abs(qAML-qAnalytical))/len(qAML)))
        plt.plot(t, qLBFGSB-qAnalytical, label='L-BFGS-B')
        plt.title('Difference between the L-BFGS-B and analytical solutions')
        plt.xlabel('Time')
        plt.ylabel('Control variable diff.')
        plt.legend()
        plt.show()
        print('L-BFGS-B error: {}'.format(qLBFGSB-qAnalytical))
        print('Average absolute L-BFGS-B error: {}'.format(np.sum(np.abs(qLBFGSB-qAnalytical))/len(qLBFGSB)))
    FOM_LBFGSB_diff = q-qLBFGSB
    RON_LBFGSB_diff = qAML-qLBFGSB
    for i in range(nb):
        plt.plot(t, FOM_LBFGSB_diff[i*(nt+1):(i+1)*(nt+1)], label='FOM-EnOpt')
        plt.plot(t, RON_LBFGSB_diff[i*(nt+1):(i+1)*(nt+1)], label='AML-EnOpt')
        plt.title('Difference between the EnOpt and L-BFGS-B solutions')
        plt.xlabel('Time')
        plt.ylabel('Control variable diff.')
        plt.legend()
        plt.show()
        print('FOM-EnOpt L-BFGS-B error, shape functional {}: {}'.format(i+1, q-qLBFGSB))
        print('Average absolute FOM-EnOpt L-BFGS-B error, shape functional {}: {}'.format(i, np.sum(np.abs(q-qLBFGSB))/len(q)))
        print('AML-EnOpt L-BFGS-B error, shape functional {}: {}'.format(i+1, qAML-qLBFGSB))
        print('Average absolute AML-EnOpt L-BFGS-B error, shape functional {}: {}'.format(i, np.sum(np.abs(qAML-qLBFGSB))/len(qAML)))
    return q, method, FOMValues, outerIterationsTotal, FOMEvaluationsTotal, runTimeTotal, qAML, methodAML, FOMValuesAML, surrogateValuesAML, outerIterationsTotalAML, innerIterationsTotalAML, FOMEvaluationsTotalAML, surrogateEvaluationsTotalAML, surrogateEvalAML, surrogateTrainAML, trainingTimeTotalAML, runTimeTotalAML, qLBFGSB, FOMValueLBFGSB, runTimeTotalLBFGSB


def compare_AML_EnOpt_DNN(addDNNStruct, init, N, eps_o, eps_i, k_1_o, k_1_i, V_DNN, delta_init, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt, q_shape):
    DNNStruct = [V_DNN[0]]
    DNNStruct.extend(addDNNStruct)
    q = []
    method = []
    FOMValues = []
    surrogateValues = []
    outerIterationsTotal = []
    innerIterationsTotal = []
    FOMEvaluationsTotal = []
    surrogateEvaluationsTotal = []
    surrogateEval = []
    surrogateTrain = []
    trainingTimeTotal = []
    runTimeTotal = []
    V_DNN_i = V_DNN.copy()
    for i in range(len(DNNStruct)):
        V_DNN_i[0] = DNNStruct[i]
        qIter, methodIter, FOMValuesIter, surrogateValuesIter, outerIterationsTotalIter, innerIterationsTotalIter, FOMEvaluationsTotalIter, surrogateEvaluationsTotalIter, surrogateEvalIter, surrogateTrainIter, trainingTimeTotalIter, runTimeTotalIter = evalROM_EnOpt(init, N, eps_o, eps_i, k_1_o, k_1_i, k_tr, V_DNN_i, delta_init, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt, q_shape)
        q.append(qIter)
        method.append(methodIter)
        FOMValues.append(FOMValuesIter)
        surrogateValues.append(surrogateValuesIter)
        outerIterationsTotal.append(outerIterationsTotalIter)
        innerIterationsTotal.append(innerIterationsTotalIter)
        FOMEvaluationsTotal.append(FOMEvaluationsTotalIter)
        surrogateEvaluationsTotal.append(surrogateEvaluationsTotalIter)
        surrogateEval.append(surrogateEvalIter)
        surrogateTrain.append(surrogateTrainIter)
        trainingTimeTotal.append(trainingTimeTotalIter)
        runTimeTotal.append(runTimeTotalIter)
    for i in range(len(DNNStruct)):
        plt.plot(range(len(FOMValues[i])), FOMValues[i], label='{}'.format(DNNStruct[i][1:-1]))
        print('\nOutput for the DNN structure {}: {}\n'.format(DNNStruct[i][1:-1], q[i]))
        print('Method for the DNN structure {}: {}\n'.format(DNNStruct[i][1:-1], method[i]))
        print('FOM objective functional values for the DNN structure {}: {}\n'.format(DNNStruct[i][1:-1], FOMValues[i]))
        print('FOM objective functional output value for the DNN structure {}: {}\n'.format(DNNStruct[i][1:-1], FOMValues[i][-1]))
        print('Surrogate functional values for the DNN structure {}: {}\n'.format(DNNStruct[i][1:-1], surrogateValues[i]))
        print('Last surrogate functional value for the DNN structure {}: {}\n'.format(DNNStruct[i][1:-1], surrogateValues[i][-1]))
        print('Number of outer iterations for the DNN structure {}: {}\n'.format(DNNStruct[i][1:-1], outerIterationsTotal[i]))
        print('Number of inner iterations for the DNN structure {}: {}\n'.format(DNNStruct[i][1:-1], innerIterationsTotal[i]))
        print('Number of FOM evaluations for the DNN structure {}: {}\n'.format(DNNStruct[i][1:-1], FOMEvaluationsTotal[i]))
        print('Number of surrogate evalutations for the DNN structure {}: {}\n'.format(DNNStruct[i][1:-1], surrogateEvaluationsTotal[i]))
        print('Training time (minutes) for the DNN structure {}: {}\n'.format(DNNStruct[i][1:-1], trainingTimeTotal[i]))
        print('Total run time (minutes) for the DNN structure {}: {}\n'.format(DNNStruct[i][1:-1], runTimeTotal[i]))
        print('MSE validation loss for the DNN structure {}: {}'.format(DNNStruct[i][1:-1], surrogateEval[i]))
        print('Minimum MSE validation loss for the DNN structure {}: {}'.format(DNNStruct[i][1:-1], np.min(surrogateEval[i])))
        print('Maximum MSE validation loss for the DNN structure {}: {}'.format(DNNStruct[i][1:-1], np.max(surrogateEval[i])))
        print('Average MSE validation loss for the DNN structure {}: {}\n'.format(DNNStruct[i][1:-1], np.sum(surrogateEval[i])/len(surrogateEval[i])))
        print('MSE training loss for the DNN structure {}: {}'.format(DNNStruct[i][1:-1], surrogateTrain[i]))
        print('Minimum MSE training loss for the DNN structure {}: {}'.format(DNNStruct[i][1:-1], np.min(surrogateTrain[i])))
        print('Maximum MSE training loss for the DNN structure {}: {}'.format(DNNStruct[i][1:-1], np.max(surrogateTrain[i])))
        print('Average MSE training loss for the DNN structure {}: {}\n'.format(DNNStruct[i][1:-1], np.sum(surrogateTrain[i])/len(surrogateTrain[i])))
    plt.title('Comparison of the FOM objective functional values \n for different DNN structures')
    plt.xlabel('Iteration')
    plt.ylabel('FOM objective functional value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    minIter = np.min(outerIterationsTotal)
    if minIter > 1:
        for i in range(len(DNNStruct)):
            plt.plot(np.arange(minIter-1, len(FOMValues[i])), FOMValues[i][(minIter-1):], label='{}'.format(DNNStruct[i][1:-1]))
        plt.title('Comparison of the FOM objective functional values \n for different DNN structures, last iterations')
        plt.xlabel('Iteration')
        plt.ylabel('FOM objective functional value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.show()
    return q, method, FOMValues, surrogateValues, outerIterationsTotal, innerIterationsTotal, FOMEvaluationsTotal, surrogateEvaluationsTotal, surrogateEval, surrogateTrain, trainingTimeTotal, runTimeTotal


# performance test that runs the FOM_EnOpt algorithm rep times
def testFOM_EnOpt(rep, init, N, eps, k_1, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt, q_shape):
    global showOuterIterationPlots
    global showInnerIterationPlots
    global showPlots
    showOuterIterationPlotsSave = showOuterIterationPlots
    showInnerIterationPlotsSave = showInnerIterationPlots
    showPlotsSave = showPlots
    showOuterIterationPlots = False
    showInnerIterationPlots = False
    showPlots = False
    q, method, FOMValues, FOMValuesOut, outerIterationsTotal, FOMEvaluationsTotal, runTimeTotal = [], [], [], [], [], [], []
    for i in range(rep):
        qIter, methodIter, FOMValuesIter, outerIterationsTotalIter, FOMEvaluationsTotalIter, runTimeTotalIter = evalFOM_EnOpt(init, N, eps, k_1, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt, q_shape)
        q.append(qIter)
        method.append(methodIter)
        FOMValues.append(FOMValuesIter)
        FOMValuesOut.append(FOMValuesIter[-1])
        outerIterationsTotal.append(outerIterationsTotalIter)
        FOMEvaluationsTotal.append(FOMEvaluationsTotalIter)
        runTimeTotal.append(runTimeTotalIter)
    print('Output: {}\n'.format(q))
    print('Method: {}\n'.format(method))
    print('FOM objective functional values: {}\n'.format(FOMValues))
    for i in range(rep):
        print('FOM objective functional output value, repetition {}: {}'.format(i, FOMValuesOut[i]))
    print('')
    print('Minimum output FOM objective functional values: {}'.format(np.min(FOMValuesOut)))
    print('Maximum output FOM objective functional values: {}'.format(np.max(FOMValuesOut)))
    print('Average output FOM objective functional values: {}\n'.format(np.sum(FOMValuesOut)/len(FOMValuesOut)))
    print('Number of outer iterations: {}\n'.format(outerIterationsTotal))
    print('Number of FOM evaluations: {}\n'.format(FOMEvaluationsTotal))
    print('Total run time (minutes): {}\n'.format(runTimeTotal))
    print('Minimum total run time (minutes): {}\n'.format(np.min(runTimeTotal)))
    print('Maximum total run time (minutes): {}\n'.format(np.max(runTimeTotal)))
    print('Average total run time (minutes): {}\n'.format(np.sum(runTimeTotal)/len(runTimeTotal)))
    showOuterIterationPlots = showOuterIterationPlotsSave
    showInnerIterationPlots = showInnerIterationPlotsSave
    showPlots = showPlotsSave


# performance test that runs the ROM_EnOpt algorithm rep times
def testROM_EnOpt(rep, init, N, eps_o, eps_i, k_1_o, k_1_i, V_DNN, delta_init, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt, q_shape):
    global showOuterIterationPlots
    global showInnerIterationPlots
    global showPlots
    global inspectDNNStructures
    showOuterIterationPlotsSave = showOuterIterationPlots
    showInnerIterationPlotsSave = showInnerIterationPlots
    showPlotsSave = showPlots
    inspectDNNStructuresSave = inspectDNNStructures
    showOuterIterationPlots = False
    showInnerIterationPlots = False
    showPlots = False
    inspectDNNStructures = False
    q, method, FOMValues, FOMValuesOut, surrogateValues, surrogateValuesOut, outerIterationsTotal, innerIterationsTotal, FOMEvaluationsTotal, surrogateEvaluationsTotal, trainingTimeTotal, runTimeTotal = [], [], [], [], [], [], [], [], [], [], [], []
    for i in range(rep):
        qIter, methodIter, FOMValuesIter, surrogateValuesIter, outerIterationsTotalIter, innerIterationsTotalIter, FOMEvaluationsTotalIter, surrogateEvaluationsTotalIter, surrogateEvalIter, surrogateTrainIter, trainingTimeTotalIter, runTimeTotalIter = evalROM_EnOpt(init, N, eps_o, eps_i, k_1_o, k_1_i, k_tr, V_DNN, delta_init, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt, q_shape)
        q.append(qIter)
        method.append(methodIter)
        FOMValues.append(FOMValuesIter)
        FOMValuesOut.append(FOMValuesIter[-1])
        surrogateValues.append(surrogateValuesIter)
        surrogateValuesOut.append(surrogateValuesIter[-1])
        outerIterationsTotal.append(outerIterationsTotalIter)
        innerIterationsTotal.append(innerIterationsTotalIter)
        FOMEvaluationsTotal.append(FOMEvaluationsTotalIter)
        surrogateEvaluationsTotal.append(surrogateEvaluationsTotalIter)
        trainingTimeTotal.append(trainingTimeTotalIter)
        runTimeTotal.append(runTimeTotalIter)
    print('Output: {}\n'.format(q))
    print('Method: {}\n'.format(method))
    print('FOM objective functional values: {}\n'.format(FOMValues))
    for i in range(rep):
        print('FOM objective functional output value, repetition {}: {}'.format(i, FOMValuesOut[i]))
    print('')
    print('Minimum output FOM objective functional values: {}'.format(np.min(FOMValuesOut)))
    print('Maximum output FOM objective functional values: {}'.format(np.max(FOMValuesOut)))
    print('Average output FOM objective functional values: {}\n'.format(np.sum(FOMValuesOut)/len(FOMValuesOut)))
    print('Surrogate functional values: {}\n'.format(surrogateValues))
    for i in range(rep):
        print('Last surrogate functional value, repetition {}: {}'.format(i, surrogateValuesOut[i]))
    print('')
    print('Number of outer iterations: {}\n'.format(outerIterationsTotal))
    print('Number of inner iterations: {}\n'.format(innerIterationsTotal))
    print('Number of FOM evaluations: {}\n'.format(FOMEvaluationsTotal))
    print('Number of surrogate evalutations: {}\n'.format(surrogateEvaluationsTotal))
    print('Training time (minutes): {}'.format(trainingTimeTotal))
    print('Minimum training time (minutes): {}'.format(np.min(trainingTimeTotal)))
    print('Maximum training time (minutes): {}'.format(np.max(trainingTimeTotal)))
    print('Average training time (minutes): {}\n'.format(np.sum(trainingTimeTotal)/len(trainingTimeTotal)))
    print('Total run time (minutes): {}'.format(runTimeTotal))
    print('Minimum total run time (minutes): {}'.format(np.min(runTimeTotal)))
    print('Maximum total run time (minutes): {}'.format(np.max(runTimeTotal)))
    print('Average total run time (minutes): {}\n'.format(np.sum(runTimeTotal)/len(runTimeTotal)))
    showOuterIterationPlots = showOuterIterationPlotsSave
    showInnerIterationPlots = showInnerIterationPlotsSave
    showPlots = showPlotsSave
    inspectDNNStructures = inspectDNNStructuresSave
