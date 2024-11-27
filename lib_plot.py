import matplotlib.pyplot as plt
import pyvista
from dolfinx.plot import vtk_mesh
from dolfinx.fem import Constant, Expression, FunctionSpace, Function
from ufl import grad, div, VectorElement, FiniteElement, nabla_grad, Identity, dot
from petsc4py import PETSc
import numpy as np
import pandas as pd
from IPython.display import display, update_display
import os.path as osp

'''From Visco-elastic plot, untested'''


def plot_margin(N, t, cumt, c, T, gd, margvel, fig=None, ax=None, figureSize=8):
    """Plot Myosin, tension and contraction rate profiles"""

    # Rescale x axis:
    cumt = cumt / cumt[-1] * 360 - 180  # in degrees

    if fig is None:
        fig, ax = plt.subplots(figsize=(figureSize, figureSize))
        newfig = True
    else:
        ax.cla()
        newfig = False

    ax.set_title('t={:.2f}'.format(t))
    ax.plot(cumt, c * 0, 'k')  # x-axis
    ax.plot(cumt, c, label='Myosin')
    ax.plot(cumt, T, label='Tension')
    ax.plot(cumt[0:N], gd, label='Ext. rate')
    ax.plot(cumt[0:N], margvel, label='Margin velocity')
    ax.set_ylim(-1, 1.5)
    ax.set_xlim(-180, 180)
    ax.legend()

    if newfig:
        display(fig, display_id='marginplot')
    else:
        update_display(fig, display_id='marginplot')

    return fig, ax


def plot_diff_FEM_PIV(data, uint, Umax, stress, totalstress, err, figsettings, timep, frint, cutoutarrows):
    """Plot superposition and difference of PIV and FEM, optionally scaled by maximal FEM velocity in each frame"""

    # unpack figsettings
    figureSize, plotscale, dpi = figsettings

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figureSize, figureSize), dpi=dpi)
    # FEM in colour
    colorU = np.sqrt((uint.dx.values / Umax) ** 2 + (uint.dy.values / Umax) ** 2)
    # Plot PIV and FEM in superposition
    ax1.quiver(data.x.values, data.y.values, data.dx.values / Umax, data.dy.values / Umax, color='grey',
               scale=plotscale)
    ax1.quiver(uint.x.values, uint.y.values, uint.dx.values / Umax, uint.dy.values / Umax, colorU, scale=plotscale,
               edgecolor='white')

    # Format
    ax1.set_aspect('equal')
    ax1.plot()
    ax1.set_title(f"t={timep:n}" + f"+{frint:n}" + "s, Stress: " + f"{stress[0]:.4f} ")  # ,
    # scaled plot, [Stress, Net stress, Err.]: "+"{:.4f} ".format(stress[0])+"{:.4f} ".format(totalstress[0])+"{
    # :.2f}".format(100*err[4])+"%" )

    # Calculate difference
    diff = data.copy()
    diff['dx'] = data.dx - uint.dx
    diff['dy'] = data.dy - uint.dy

    if cutoutarrows:  # Replace NaN values entries with zero
        data.loc[data['dx'] == np.nan, 'dx'] = 0
        data.loc[data['dy'] == np.nan, 'dy'] = 0
        diff2 = data.copy()
        diff2['dx'] = data.dx - uint.dx
        diff2['dy'] = data.dy - uint.dy
    else:
        diff2 = diff

    # Plot difference
    ax2.quiver(diff2.x.values, diff2.y.values, diff2.dx.values / Umax, diff2.dy.values / Umax, color='black',
               scale=plotscale)
    ax2.set_aspect('equal')
    ax2.plot()
    ax2.set_title('Difference PIV-FEM:')  # . Flux Error (post/pre): '+str(np.round((totflux/netflux-1)*100,1))+'%')

    # Display figures
    # plt.show()

    return diff, fig, (ax1, ax2)


# plot mesh
def plot_mesh_pyvista(W, mesh):
    # pyvista.start_xvfb()
    pyvista.set_jupyter_backend("static")
    _, _, _ = vtk_mesh(W.sub(0).collapse()[0])

    # Create a pyvista-grid for the mesh
    grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))

    # Create plotter
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, style="wireframe", color="k")
    plotter.view_xy()
    plotter.set_background('white')
    #plotter.screenshot('foo.png', window_size=[2000, 2000])

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        pass
        # pyvista.start_xvfb()
        #fig_as_array = plotter.screenshot("mesh.png")
    return


'''Pyvista plots'''


def plot_flow_pyvista(mesh, uh, V, scale, newfig=0, backend="static"):
    """Plot flow as arrows on grid"""
    # pyvista.start_xvfb()
    pyvista.set_jupyter_backend(backend)  # deactivate interactivity
    topology, cell_types, geometry = vtk_mesh(V)
    values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
    values[:, :len(uh)] = uh.x.array.real.reshape((geometry.shape[0], len(uh)))

    # Create a point cloud of glyphs
    function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    function_grid["u"] = values
    glyphs = function_grid.glyph(orient="u", factor=scale)

    # Create a pyvista-grid for the mesh
    grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))

    # Create plotter
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, style="wireframe", color="k")
    plotter.add_mesh(glyphs)
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        if newfig:
            display(plotter.show(), display_id='flowplot')
        else:
            update_display(plotter.show(), display_id='flowplot')
    else:
        pass
        #fig_as_array = plotter.screenshot("glyphs.png")

    return


# noinspection PyTypeChecker
def plot_u_pyvista(u, mesh, scale, THorder=1):
    """Plot vel"""

    # pyvista.start_xvfb()
    pyvista.set_jupyter_backend("trame")

    ex = Constant(mesh, PETSc.ScalarType((1, 0)))
    ey = Constant(mesh, PETSc.ScalarType((0, 1)))

    ux = dot(ex, u)
    uy = dot(ey, u)

    V = FunctionSpace(mesh, FiniteElement("Lagrange", mesh.ufl_cell(), THorder + 1))
    ux_expr = Expression(ux, V.element.interpolation_points())
    uy_expr = Expression(uy, V.element.interpolation_points())
    Ux = Function(V)
    Uy = Function(V)
    Ux.interpolate(ux_expr)
    Uy.interpolate(uy_expr)

    # Extract topology from mesh and create pyvista mesh
    topology, cell_types, x = vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)

    # Set deflection values and add it to plotter
    grid.point_data["Ux"] = Ux.x.array
    grid.point_data["Uy"] = Uy.x.array

    plotterx = pyvista.Plotter()
    grid.set_active_scalars("Ux")
    warped = grid.warp_by_scalar("Ux", factor=scale)
    plotterx.add_mesh(warped, style="wireframe", show_scalar_bar=True, scalars="Ux")
    plotterx.view_xz()
    plotterx.set_background('white')
    plottery = pyvista.Plotter()
    grid.set_active_scalars("Uy")
    warped = grid.warp_by_scalar("Uy", factor=scale)
    plottery.add_mesh(warped, style="wireframe", show_scalar_bar=True, scalars="Uy")
    plottery.view_xz()
    plottery.set_background('white')
    if not pyvista.OFF_SCREEN:
        print("ux")
        print("Min: " + str(np.min(Ux.x.array)))
        print("Max: " + str(np.max(Ux.x.array)))
        print("Avg: " + str(np.mean(Ux.x.array)))
        plotterx.show()
        print("uy")
        print("Min: " + str(np.min(Uy.x.array)))
        print("Max: " + str(np.max(Uy.x.array)))
        print("Avg: " + str(np.mean(Uy.x.array)))
        plottery.show()
    else:
        pass
        # pyvista.start_xvfb()
        #plotter.screenshot("deflection.png")
    return


def plot_mode_pyvista(p, mesh, scale, THorder=1, clim=None, backend='static'):
    """Plot pressure"""

    # pyvista.start_xvfb()
    pyvista.set_jupyter_backend(backend)  # use pythreejs for 3D

    # Extract topology from mesh and create pyvista mesh
    V = FunctionSpace(mesh, FiniteElement("Lagrange", mesh.ufl_cell(), THorder))
    topology, cell_types, x = vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)

    # Set deflection values and add it to plotter
    grid.point_data["p"] = p.x.array
    grid.set_active_scalars("p")
    # warped = grid.warp_by_scalar("p", factor=scale)

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=False, show_scalar_bar=False, scalars="p", colormap='turbo', style='surface',
                     clim=clim)
    plotter.view_xy()
    plotter.set_background('white')
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        # pyvista.start_xvfb()
        plotter.screenshot("deflection.png")
    return


def plot_p_pyvista(p, mesh, scale, THorder=1, clim=None, backend='static'):
    """Plot pressure"""

    # pyvista.start_xvfb()
    pyvista.set_jupyter_backend(backend)  # use pythreejs for 3D

    # Extract topology from mesh and create pyvista mesh
    V = FunctionSpace(mesh, FiniteElement("Lagrange", mesh.ufl_cell(), THorder))
    topology, cell_types, x = vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)

    # Set deflection values and add it to plotter
    grid.point_data["p"] = p.x.array
    grid.set_active_scalars("p")
    # warped = grid.warp_by_scalar("p", factor=scale)

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True, scalars="p", colormap='turbo', style='surface', clim=clim)
    plotter.view_xy()
    plotter.set_background('white')
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        # pyvista.start_xvfb()
        plotter.screenshot("deflection.png")
    return


def plot_div_pyvista(u, gamma, mesh, scale, option=0, THorder=1, clim=None, backend='static'):
    """Plot divergence"""

    # pyvista.start_xvfb()
    if clim is None:
        pass
        #clim = [-1, 1]
    pyvista.set_jupyter_backend(backend)

    assert option in [0, 1, 2]
    if option == 0:
        dv = div(u)
    elif option == 1:
        dv = gamma
    else:
        dv = div(u) - gamma

    V = FunctionSpace(mesh, FiniteElement("Lagrange", mesh.ufl_cell(), THorder))
    dv_expr = Expression(dv, V.element.interpolation_points())
    Dv = Function(V)
    Dv.interpolate(dv_expr)

    # Extract topology from mesh and create pyvista mesh
    topology, cell_types, x = vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)

    # Set deflection values and add it to plotter
    grid.point_data["dv"] = Dv.x.array
    grid.set_active_scalars("dv")
    # warped = grid.warp_by_scalar("dv", factor=scale)

    plotter = pyvista.Plotter()
    # plotter.add_mesh(warped, style="wireframe",show_scalar_bar=True, scalars="dv")
    plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True, scalars="dv", colormap='turbo', clim=clim,
                     style='surface')
    plotter.view_xy()
    plotter.set_background('white')
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        # pyvista.start_xvfb()
        plotter.screenshot("deflection.png")
    return


def plot_scalar_pyvista(u, mesh, scale=1, THorder=1, clim=None, backend='static'):
    """Plot arbitrary scalar u """
    pyvista.set_jupyter_backend(backend)

    # Interpolate scalar on mesh
    V = FunctionSpace(mesh, FiniteElement("Lagrange", mesh.ufl_cell(), THorder))
    dv_expr = Expression(u, V.element.interpolation_points())
    Dv = Function(V)
    Dv.interpolate(dv_expr)

    # Extract topology from mesh and create pyvista mesh
    topology, cell_types, x = vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)

    # Set deflection values and add it to plotter
    grid.point_data["dv"] = Dv.x.array * scale
    grid.set_active_scalars("dv")
    # warped = grid.warp_by_scalar("dv", factor=scale)

    plotter = pyvista.Plotter()
    # plotter.add_mesh(warped, style="wireframe",show_scalar_bar=True, scalars="dv")
    plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True, scalars="dv", colormap='turbo', clim=clim,
                     style='surface')
    plotter.view_xy()
    plotter.set_background('white')
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        # pyvista.start_xvfb()
        plotter.screenshot("deflection.png")
    return

def plot_stress_pyvista(u, p, mesh, scale, THorder=1):
    """Plot stress"""

    # pyvista.start_xvfb()
    pyvista.set_jupyter_backend("trame")

    ex = Constant(mesh, PETSc.ScalarType((1, 0)))
    ey = Constant(mesh, PETSc.ScalarType((0, 1)))

    e = nabla_grad(u) + grad(u) - div(u) * Identity(2)

    stressxx = -p + dot(ex, dot(ex, e))
    stressxy = dot(ey, dot(ex, e))
    stressyy = -p + dot(ey, dot(ey, e))

    V = FunctionSpace(mesh, FiniteElement("Lagrange", mesh.ufl_cell(), THorder))
    sxx_expr = Expression(stressxx, V.element.interpolation_points())
    sxy_expr = Expression(stressxy, V.element.interpolation_points())
    syy_expr = Expression(stressyy, V.element.interpolation_points())
    Sxx = Function(V)
    Sxy = Function(V)
    Syy = Function(V)
    Sxx.interpolate(sxx_expr)
    Sxy.interpolate(sxy_expr)
    Syy.interpolate(syy_expr)

    # Extract topology from mesh and create pyvista mesh
    topology, cell_types, x = vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)

    # Set deflection values and add it to plotter
    grid.point_data["Sxx"] = Sxx.x.array
    grid.point_data["Sxy"] = Sxy.x.array
    grid.point_data["Syy"] = Syy.x.array

    plotterxx = pyvista.Plotter()
    grid.set_active_scalars("Sxx")
    warped = grid.warp_by_scalar("Sxx", factor=scale)
    plotterxx.add_mesh(warped, style="wireframe", show_scalar_bar=True, scalars="Sxx")
    plotterxx.view_xz()
    plotterxx.set_background('white')
    plotterxy = pyvista.Plotter()
    grid.set_active_scalars("Sxy")
    warped = grid.warp_by_scalar("Sxy", factor=scale)
    plotterxy.add_mesh(warped, style="wireframe", show_scalar_bar=True, scalars="Sxy")
    plotterxy.view_xz()
    plotterxy.set_background('white')
    plotteryy = pyvista.Plotter()
    grid.set_active_scalars("Syy")
    warped = grid.warp_by_scalar("Syy", factor=scale)
    plotteryy.add_mesh(warped, style="wireframe", show_scalar_bar=True, scalars="Syy")
    plotteryy.view_xz()
    plotteryy.set_background('white')
    if not pyvista.OFF_SCREEN:
        plotterxx.show()
        plotterxy.show()
        plotteryy.show()
    else:
        pass
        # pyvista.start_xvfb()
        #plotter.screenshot("deflection.png")
    return


def plot_s_pyvista(s, mesh, scale, THorder=1):
    """Plot stress tensor"""

    # pyvista.start_xvfb()
    pyvista.set_jupyter_backend("trame")

    ex = Constant(mesh, PETSc.ScalarType((1, 0)))
    ey = Constant(mesh, PETSc.ScalarType((0, 1)))

    stressxx = dot(ex, dot(ex, s))
    stressxy = dot(ey, dot(ex, s))
    #stressyx = dot(ex, dot(ey, s))
    stressyy = dot(ey, dot(ey, s))

    V = FunctionSpace(mesh, FiniteElement("Lagrange", mesh.ufl_cell(), THorder))
    sxx_expr = Expression(stressxx, V.element.interpolation_points())
    sxy_expr = Expression(stressxy, V.element.interpolation_points())
    syy_expr = Expression(stressyy, V.element.interpolation_points())
    Sxx = Function(V)
    Sxy = Function(V)
    Syy = Function(V)
    Sxx.interpolate(sxx_expr)
    Sxy.interpolate(sxy_expr)
    Syy.interpolate(syy_expr)

    # Extract topology from mesh and create pyvista mesh
    topology, cell_types, x = vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)

    # Set deflection values and add it to plotter
    grid.point_data["Sxx"] = Sxx.x.array
    grid.point_data["Sxy"] = Sxy.x.array
    grid.point_data["Syy"] = Syy.x.array

    plotterxx = pyvista.Plotter()
    grid.set_active_scalars("Sxx")
    warped = grid.warp_by_scalar("Sxx", factor=scale)
    plotterxx.add_mesh(warped, style="wireframe", show_scalar_bar=True, scalars="Sxx")
    plotterxx.view_xz()
    plotterxx.set_background('white')
    plotterxy = pyvista.Plotter()
    grid.set_active_scalars("Sxy")
    warped = grid.warp_by_scalar("Sxy", factor=scale)
    plotterxy.add_mesh(warped, style="wireframe", show_scalar_bar=True, scalars="Sxy")
    plotterxy.view_xz()
    plotterxy.set_background('white')
    plotteryy = pyvista.Plotter()
    grid.set_active_scalars("Syy")
    warped = grid.warp_by_scalar("Syy", factor=scale)
    plotteryy.add_mesh(warped, style="wireframe", show_scalar_bar=True, scalars="Syy")
    plotteryy.view_xz()
    plotteryy.set_background('white')
    if not pyvista.OFF_SCREEN:
        print("sxx")
        print("Min: " + str(np.min(Sxx.x.array)))
        print("Max: " + str(np.max(Sxx.x.array)))
        print("Avg: " + str(np.mean(Sxx.x.array)))
        plotterxx.show()
        print("sxy")
        print("Min: " + str(np.min(Sxy.x.array)))
        print("Max: " + str(np.max(Sxy.x.array)))
        print("Avg: " + str(np.mean(Sxy.x.array)))
        plotterxy.show()
        print("syy")
        print("Min: " + str(np.min(Syy.x.array)))
        print("Max: " + str(np.max(Syy.x.array)))
        print("Avg: " + str(np.mean(Syy.x.array)))
        plotteryy.show()
    else:
        pass
        # pyvista.start_xvfb()
        #plotter.screenshot("deflection.png")
    return


'''Additonally from TM plot, untested'''


def plot_average(diffout, outpath, stressfn, figsettings, timeint):
    """Plot average residual between PIV and FEM, summed over entire dataset"""
    '''Weighs frames equally and thus assumes implicitly a uniform frame interval'''

    # unpack figsettings
    figureSize, plotscale, dpi = figsettings
    fig, ax = plt.subplots(figsize=(figureSize, figureSize), dpi=dpi)

    # calculate average
    xdiff = []
    ydiff = []
    for i, diff in enumerate(diffout):
        xdiff.append(diff['dx'].values)
        ydiff.append(diff['dy'].values)
    xdiff = np.asarray(xdiff)
    ydiff = np.asarray(ydiff)
    mean = diffout[0].copy()
    mean['dx'] = np.sum(xdiff, 0) * timeint
    mean['dy'] = np.sum(ydiff, 0) * timeint

    maxerr = np.max(np.sqrt(
        mean['dx'].values[~np.isnan(mean['dx'].values)] ** 2 + mean['dy'].values[~np.isnan(mean['dx'].values)] ** 2))

    # Plot difference
    ax.quiver(mean.x.values, mean.y.values, mean.dx.values, mean.dy.values, color='black', angles='xy',
              scale_units='xy', scale=1)  # plot arrows in microns
    ax.set_aspect('equal')
    ax.plot()
    ax.set_title('Cumulative diff PIV-FEM. Max diff: ' + "{:.2f}".format(maxerr) + " microns")

    # Save figure
    plt.savefig(osp.join(outpath, stressfn + '_avgerr.png'), dpi=dpi)

    return mean, maxerr


def plot_stokes_pyvista(u, p, mesh, scale, THorder=1):
    """Plot stokes residual"""

    e = grad(u) + nabla_grad(u) - div(u) * Identity(u.geometric_dimension())
    stokes = -grad(p) + div(e)

    V_stokes = FunctionSpace(mesh, VectorElement("Lagrange", mesh.ufl_cell(), THorder - 1))
    stokes_expr = Expression(stokes, V_stokes.element.interpolation_points())
    Stokes = Function(V_stokes)
    Stokes.interpolate(stokes_expr)

    # pyvista.start_xvfb()
    pyvista.set_jupyter_backend("pythreejs")  # use pythreejs for 3D
    topology, cell_types, geometry = vtk_mesh(V_stokes)
    values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
    values[:, :len(Stokes)] = Stokes.x.array.real.reshape((geometry.shape[0], len(Stokes)))

    # Create a point cloud of glyphs
    function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    function_grid["Stokes"] = values
    glyphs = function_grid.glyph(orient="Stokes", factor=scale)

    # Create a pyvista-grid for the mesh
    grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))

    # Create plotter
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, style="wireframe", color="k")
    plotter.add_mesh(glyphs)
    plotter.view_xy()
    plotter.set_background('white')
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        pass
        # pyvista.start_xvfb()
        # fig_as_array = plotter.screenshot("Stokes.png")
    return


def plot_stresses(stressfn, figSize):
    """Plot stresses from file"""
    df = pd.read_csv('data/FEMoutput/' + stressfn + '.csv')
    fig, ax = plt.subplots(figsize=(2 * figSize, figSize))
    ax.plot(df['t'].values, df['t'].values * 0, 'k')
    ax.plot(df['t'].values, df['stress_x'].values)
    ax.plot(df['t'].values, df['totalstress_x'].values)
    plt.show()
    return
