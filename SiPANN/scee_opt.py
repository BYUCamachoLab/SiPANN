import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from tables import ComplexCol, Float64Col, IsDescription, open_file
from tqdm import tqdm

from SiPANN import scee

##################################################
###              HELPER FUNCTIONS              ###
###  used to help quickly define gap functions ###
##################################################

# def bernstein_quick(n, j, t):
#     """Computes the jth bernstein polynomial.
#     Updated to support NumPy Arrays (Vectorization).
#     """
#     # scipy.special.comb is fast and handles arrays natively
#     coeff = special.comb(n, j)
#     return coeff * (t ** j) * ((1 - t) ** (n - j))

def bernstein_quick(n, j, t):
    """Computes the jth bernstein polynomial.
    Updated to support NumPy Arrays and handle edge cases (j<0 or j>n).
    """
    # Safety Check: The polynomial is 0 if j is out of bounds [0, n]
    if j < 0 or j > n:
        # If t is an array, return an array of zeros. If scalar, return 0.
        if np.ndim(t) > 0:
            return np.zeros_like(t)
        return 0.0

    # scipy.special.comb is fast and handles arrays natively
    coeff = special.comb(n, j)
    return coeff * (t ** j) * ((1 - t) ** (n - j))

def bezier_quick(g, length):
    """Computes the bezier curve for the evenly spaced control points with gaps g.
    Optimized to handle vector inputs (t can be an array of 1000+ points).
    """
    n = len(g) - 1
    
    # We explicitly define the functions to handle array inputs correctly
    # axis=0 sums across the control points, preserving the spatial grid (axis 1)
    
    return {
        "g": g,
        "w": length,
        "f": lambda t: np.sum(
            np.array(
                [(g[j]) * bernstein_quick(n, j, t / length) for j in range(len(g))]
            ),
            axis=0,
        ),
        "df": lambda t: np.sum(
            np.array(
                [
                    n
                    * (g[j])
                    * (
                        bernstein_quick(n - 1, j - 1, t / length)
                        - bernstein_quick(n - 1, j, t / length)
                    )
                    for j in range(len(g))
                ]
            ),
            axis=0,
        )
        / length,
        "d2f": lambda t: np.sum(
            np.array(
                [
                    n
                    * (n - 1)
                    * (g[j] / 2)
                    * (
                        bernstein_quick(n - 2, j - 2, t / length)
                        - 2 * bernstein_quick(n - 2, j - 1, t / length)
                        + bernstein_quick(n - 2, j, t / length)
                    )
                    for j in range(len(g))
                ]
            ),
            axis=0,
        )
        / length ** 2,
    }


class Optimization(IsDescription):
    """Class to save the data in h5 files if dataCollect=True."""

    width = Float64Col()
    thickness = Float64Col()
    wavelength = Float64Col()
    k = ComplexCol(16)
    t = ComplexCol(16)
    length = Float64Col()
    g1 = Float64Col()
    g2 = Float64Col()
    g3 = Float64Col()
    g4 = Float64Col()
    g5 = Float64Col()
    g6 = Float64Col()
    g7 = Float64Col()
    g8 = Float64Col()


#########################################################
###                 ACTUAL OPTIMIZING                 ###
###  this function does all of the heavy lifting here ###
#########################################################
def make_coupler(
    goalK=0.4,
    arrayK=None,
    waveSweep=np.linspace(1500, 1600, 4),
    gapN=16,
    algo=35,
    edgeN=8,
    plot=False,
    collectData=False,
    width=500,
    thickness=220,
    radius=5000,
    maxiter=None,
    verbose=0,
):
    """Optimizes output from a directional coupler defined by a bezier curve."""

    import nlopt

    # initial values for the optimizer to use and bounds for values
    couplingWidth = 20000
    couplingMin = 5000
    couplingMax = 100000
    gapMin = 0
    gapMax = 1500
    iter = [0]
    mseVals = []
    coupler = scee.GapFuncSymmetric(
        width, thickness, lambda x: gapMax, lambda x: 0, 0, couplingWidth
    )
    curvatureFunc = lambda x: 0
    localOpt = False
    waveN = len(waveSweep)
    waveStart = waveSweep[0]
    waveStop = waveSweep[-1]

    # set up progress bars
    if verbose == 1:
        if maxiter is not None:
            loop = tqdm(total=maxiter * 2, position=0, leave=True)
        else:
            loop = tqdm(total=float("inf"), position=0, leave=True)

    # varying values of goal coupling if desired
    if arrayK is not None:
        goalK = np.array(
            [arrayK[0] if k < waveN / 2 else arrayK[1] for k in range(waveN)]
        )

    # define plot for debugging and final result
    if plot:
        plt.ion()
    fig, axes = plt.subplots(2, 1)

    axes[0].set_ylim(-gapMax, gapMax + 100)
    scatter1 = axes[0].scatter(
        np.linspace(0, couplingWidth, gapN), np.zeros(gapN), label="control points"
    )
    (line1,) = axes[0].plot(
        np.linspace(0, couplingWidth, 500), np.zeros(500), label="gap function"
    )
    (line2,) = axes[1].plot(
        waveSweep,
        np.zeros_like(waveSweep),
        "o-",
        label="cross-port goal = " + str(goalK),
    )
    axes[1].set_ylim(-5, 0)
    axes[0].set_xlabel("coupling length (nm)")
    axes[0].set_ylabel("coupling gap (nm)")
    axes[1].set_xlabel("wavelength (nm)")
    axes[1].set_ylabel("error (dB)")
    plt.tight_layout()
    axes[0].legend(loc=8)
    axes[1].legend(loc=8)

    # file to collect data in
    if collectData:
        h5file = open_file(
            "./data/" + ("%.2f" % goalK).split(".")[1] + "/" + "data.h5",
            mode="w",
            title="Simulation Data",
        )
        group = h5file.create_group("/", "coupler", "Optimization")
        table = h5file.create_table(group, "optdata", Optimization, "Data")
        point = table.row

    def f(g, grad):
        iter[0] += 1

        # define optimization points and distance between them
        x = np.linspace(0, g[0], 2 * gapN + 2 * edgeN)
        # mirror gap control points for symmetry
        g_total = np.append(
            np.append(np.append(np.full(edgeN, gapMax), g[1:]), g[-1:0:-1]),
            np.full(edgeN, gapMax),
        )

        # get interpolation of waveguide
        gap = bezier_quick(g_total, g[0])
        coupler.update(gap=gap["f"], dgap=gap["df"], zmax=gap["w"])

        # get the cross coupling
        currK = coupler.predict((1, 4), waveSweep)

        # draw current waveguide and power if debug mode on
        if plot and iter[0] % 10000 != 0:
            scatter1.set_offsets(np.array([x, g_total]).T)
            dom = np.linspace(0, g[0], 500)
            axes[0].set_xlim(0, g[0])

            line1.set_xdata(dom)
            line1.set_ydata(gap["f"](dom))
            line2.set_ydata(-np.abs(-10 * np.log10((np.abs(currK) ** 2) / goalK)))
            fig.canvas.draw()
            fig.canvas.flush_events()
        elif plot:
            scatter1.set_visible(False)
            line1.set_visible(False)
            x_plot = np.linspace(0, g[0], 500)
            axes[0].set_xlim(0, g[0])
            axes[0].get_legend().remove()

            y_plot = gap["f"](x_plot) / 2
            fill1 = axes[0].fill_between(x_plot, -y_plot - width, -y_plot, color="k")
            fill2 = axes[0].fill_between(x_plot, y_plot, y_plot + width, color="k")
            line2.set_ydata(-np.abs(-10 * np.log10((np.abs(currK) ** 2) / goalK)))
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.waitforbuttonpress()

            line1.set_visible(True)
            fill1.set_visible(False)
            fill2.set_visible(False)
            scatter1.set_visible(True)
            axes[0].legend(loc=8)

        # get the mean squared error between current coupling and goal
        mse = np.sum((np.log10((np.abs(currK) ** 2) / goalK)) ** 2) / waveN
        mseVals.append(mse)

        # add penalty if gap function has value < 100 nm
        mse += np.sum(gap["f"](np.linspace(0, g[0], 100)) < 100)

        # # print iteration progress
        if verbose == 2:
            print(f"MSE: {mse}")
            print(f"currK: {np.abs(currK)**2}")
            print(f"g: {g}")
            print("local optimization:", localOpt, "\n")
        elif verbose == 1:
            o = "LOCAL" if localOpt else "GLOBAL"
            loop.update(1)
            loop.set_description(
                f"{o}, MSE: {np.round(mse,4)}, Mean currK: {np.round((np.abs(currK)**2).mean(),4)}"
            )

        return mse

    def constraint(x, grad, radius=5000):
        deriv = lambda x: coupler.dgap(x) / 2
        domain = np.linspace(0.002, x[0] - 0.002, 500)
        curve = np.abs(curvatureFunc(domain) / ((1 + (deriv(domain) ** 2)) ** 1.5))

        if collectData and not np.any(curve > 1 / radius):
            waves = range(waveStart, waveStop + 1)
            currK = coupler.predict((1, 4), waves)
            currT = coupler.predict((1, 3), waves)
            for i in range(101):
                point["width"] = width
                point["thickness"] = thickness
                point["wavelength"] = waves[i]
                point["length"] = x[0]
                point["k"] = currK[i]
                point["t"] = currT[i]
                point["g1"] = x[1]
                point["g2"] = x[2]
                point["g3"] = x[3]
                point["g4"] = x[4]
                point["g5"] = x[5]
                point["g6"] = x[6]
                point["g7"] = x[7]
                point["g8"] = x[8]
                point.append()
                table.flush()

        return float(np.any(curve > 1 / radius))

    ##### initialize and perform global optimization  ########
    x0 = np.append(couplingWidth, np.linspace(gapMin, gapMax, gapN))
    opt = nlopt.opt(int(algo), gapN + 1)
    opt.set_lower_bounds(np.append(couplingMin, np.full((gapN), gapMin)))
    opt.set_upper_bounds(np.append(couplingMax, np.full((gapN), gapMax)))

    opt.set_min_objective(f)

    opt.set_xtol_rel(1e-5)
    opt.set_ftol_abs(1e-7)
    if maxiter is not None:
        opt.set_maxeval(maxiter)

    opt.add_equality_constraint(lambda x, grad: constraint(x, grad, radius), 1e-8)

    gap_final = opt.optimize(x0)

    ###### local optimize after global ########
    line1.set_color("k")
    scatter1.set_color("k")
    line2.set_color("k")
    localOpt = True

    opt = nlopt.opt(40, gapN + 1)
    opt.add_equality_constraint(lambda x, grad: constraint(x, grad), 1e-8)
    opt.set_lower_bounds(np.append(couplingMin, np.full((gapN), gapMin)))
    opt.set_upper_bounds(np.append(couplingMax, np.full((gapN), gapMax)))
    opt.set_min_objective(f)
    opt.set_xtol_rel(1e-5)
    opt.set_ftol_abs(1e-7)
    opt.set_ftol_rel(1e-6)
    if maxiter is not None:
        opt.set_maxeval(maxiter)

    gap_final = opt.optimize(gap_final)

    if collectData:
        h5file.close()

    gap_total = np.append(
        np.append(np.append(np.full(edgeN, gapMax), gap_final[1:]), gap_final[-1:0:-1]),
        np.full(edgeN, gapMax),
    )
    gap = bezier_quick(gap_total, gap_final[0])
    coupler.update(gap=gap["f"], dgap=gap["df"], zmax=gap["w"])

    if plot:
        dom = np.linspace(0, gap_final[0], 500)
        axes[0].set_xlim(0, gap_final[0])
        scatter1.set_visible(False)
    K_final = coupler.predict((1, 4), waveSweep)

    if plot:
        line1.set_visible(False)
        line2.set_ydata(-np.abs(np.log10(np.abs(K_final) ** 2 / goalK)))
        axes[0].fill_between(
            dom, gap["f"](dom) / 2, gap["f"](dom) / 2 + width, color="k"
        )
        axes[0].fill_between(
            dom, -gap["f"](dom) / 2 - width, -gap["f"](dom) / 2, color="k"
        )
        axes[0].get_legend().remove()

        minf = opt.last_optimum_value()
        print(gap_final)
        print(minf, f(gap_final, None))
        print(np.abs(K_final) ** 2)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.waitforbuttonpress()

    if plot:
        fig, axes = plt.subplots(1, 1)
        axes.semilogy(mseVals)
        axes.set_xlabel("optimizer iterations")
        axes.set_ylabel("MSE")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.waitforbuttonpress()

    return coupler, gap_total, gap_final[0]


#########################################################
###              SAVING & LOADING COUPLERS            ###
###        useful for caching and reusing later       ###
#########################################################
def save_coupler(width, thickness, control_pts, length, filename):
    """Used to save optimized couplers efficiently."""
    np.savez(
        filename,
        WIDTH=np.array([width]),
        THICKNESS=np.array([thickness]),
        LENGTH=np.array([length]),
        GAP=control_pts,
    )


def load_coupler(filename):
    """Used to load optimized couplers efficiently."""
    data = np.load(filename)
    g = data["GAP"]
    length = data["LENGTH"][0]
    width = data["WIDTH"][0]
    thickness = data["THICKNESS"][0]

    bez = bezier_quick(g, length)
    return (
        scee.GapFuncSymmetric(width, thickness, bez["f"], bez["df"], 0, bez["w"]),
        length,
    )


def premade_coupler(split):
    """Loads premade couplers."""
    if split not in [10, 20, 30, 40, 50, 100]:
        raise ValueError("That splitting ratio hasn't been made")

    filename = f"split_{split}_{100-split}.npz"
    data = np.load(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "COUPLER", filename)
    )
    g = data["GAP"]
    length = data["LENGTH"][0]
    width = data["WIDTH"][0]
    thickness = data["THICKNESS"][0]

    bez = bezier_quick(g, length)
    return (
        scee.GapFuncSymmetric(width, thickness, bez["f"], bez["df"], 0, bez["w"]),
        length,
    )