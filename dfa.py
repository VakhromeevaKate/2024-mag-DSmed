import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def integrate_series(signal):
    mean_val = np.mean(signal)
    Y = np.cumsum(signal - mean_val)
    return Y

def detrended_fluctuation(Y, scale, poly_order=1):
    """
    Вычисление RMS по окнам для заданного масштаба "scale".
    poly_order=1 для линейной аппроксимации.
    """
    N = len(Y)
    segments = N // scale
    F = []
    
    for v in range(2*segments):
        if v < segments:
            index_start = v*scale
            index_end = (v+1)*scale
        else:
            index_start = N - (v-segments+1)*scale
            index_end = N - (v-segments)*scale

        segment = Y[index_start:index_end]

        # Полиномиальная аппроксимация
        x = np.arange(len(segment))
        coeffs = np.polyfit(x, segment, poly_order)
        trend = np.polyval(coeffs, x)

        F.append(np.sqrt(np.mean((segment - trend)**2)))

    return np.array(F)

def dfa(signal, scales, q_list, poly_order=1):

    Y = integrate_series(signal)
    Fq = []

    for q in q_list:
        Fq_s = []
        for s in scales:
            F = detrended_fluctuation(Y, s, poly_order=poly_order)
            if q == 0:
                val = np.exp(np.mean(np.log(F[F>0]))) 
            else:
                val = (np.mean(F**q))**(1.0/q)
            Fq_s.append(val)
        Fq.append(Fq_s)

    Fq = np.array(Fq)  #

    log_scales = np.log(scales)
    Hq = []
    for i, q in enumerate(q_list):
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, np.log(Fq[i,:]))
        Hq.append(slope)

    Hq = np.array(Hq)

    # tau(q) = qH(q) - 1
    tau_q = q_list*Hq - 1
    # h(q) = tau(q)/q
    h_q = tau_q/q_list
    # D(q) = tau(q)/(q-1)
    D_q = tau_q/(q_list-1)

    return q_list, Hq, tau_q, h_q, D_q, scales, Fq


if __name__ == "__main__":
    np.random.seed(0)
    signal = np.random.randn(10000) 
    scales = np.unique(np.logspace(1.5, 3, num=20, base=10, dtype=int))
    q_list = np.linspace(-5,5,11)

    q_list, Hq, tau_q, h_q, D_q, scales, Fq = dfa(signal, scales, q_list, poly_order=1)

    plt.figure(figsize=(10,6))
    plt.subplot(2,2,1)
    plt.title("H(q)")
    plt.plot(q_list, Hq, 'o-')
    plt.xlabel("q")
    plt.ylabel("H(q)")

    plt.subplot(2,2,2)
    plt.title("tau(q)")
    plt.plot(q_list, tau_q, 'o-')
    plt.xlabel("q")
    plt.ylabel("tau(q)")

    plt.subplot(2,2,3)
    plt.title("h(q)")
    plt.plot(q_list, h_q, 'o-')
    plt.xlabel("q")
    plt.ylabel("h(q)")

    plt.subplot(2,2,4)
    plt.title("D(q)")
    plt.plot(q_list, D_q, 'o-')
    plt.xlabel("q")
    plt.ylabel("D(q)")

    plt.tight_layout()
    plt.show()

