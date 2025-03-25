# DELFI-C3 - Parameter Estimation

## Import statements

# Load required standard modules
import numpy as np
import math
import pyproj
import itertools
from matplotlib import pyplot as plt

# Load required tudatpy modules
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy.astro.time_conversion import DateTime
from tudatpy.astro import element_conversion
from tudatpy.util import result2array
from tudatpy.kernel.astro import gravitation
from tudatpy import astro

## Define a set of radar positions for grid search

nposit = 3  # number of positions
nradars = 2  # number of radars per iteration

# Function for converting cartesian coordinates to geodetic ones
def cartesian_to_geodetic(x, y, z):  # not used
    enc_rad = 256600.0
    alt = enc_rad
    # Define the coordinate systems
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    # Perform the coordinate transformation
    lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=False)
    return lat, lon, alt

# Function for defining equidistant radar positions set along a Fibonacci spiral
def fibonacci_sphere(samples=10):
    # points = []
    pointsgeo = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        longi = np.rad2deg(math.atan2(y, x))  # * 360. / 2 * math.pi
        latit = np.rad2deg(math.asin(z / 1.))  # * 360 / 2 * math.pi
        # points.append((x, y, z))
        pointsgeo.append(longi)
        pointsgeo.append(latit)

    return pointsgeo


# Geodetic coordinates of the radar positions set
longi = fibonacci_sphere(nposit)[0::2]
latit = fibonacci_sphere(nposit)[1::2]

radars = []
for m in range(len(longi)):
    radars.append((longi[m], latit[m]))

print(len(longi))
#print(radars)

radars_combis = itertools.combinations(radars, nradars)

## Convert the iterator to a list to display all combinations
combis_list = list(radars_combis)

## Print the result
print("All possible combinations of 2 radars:")
print(len(combis_list))
#print(combis_list)

"""
 First, NAIF's `SPICE` kernels are loaded, to make the positions of various bodies such as the Enceladus, the Sun, 
 and Saturn known to `tudatpy`.
 Subsequently, the start and end epoch of the simulation are defined. Note that using `tudatpy`, the times are generally 
 specified in seconds since J2000. Hence, setting the start epoch to `0` corresponds to the 1st of January 2000. 
 The end epoch specifies a total duration of the simulation.
 For more information on J2000 and the conversion between different temporal reference frames, please refer to the 
 API documentation of the [`time_conversion module`](https://tudatpy.readthedocs.io/en/latest/time_conversion.html).
 """

# Load spice kernels
spice.load_standard_kernels()
kernels = ['/home/neumwl/TudatProjects/de438.bsp', '/home/neumwl/TudatProjects/sat427.bsp',
           '/home/neumwl/TudatProjects/pck00010.tpc']
spice.load_standard_kernels(kernels)

# Ab hier eine while-Schleife für die Definition der stations und die estimation procedure

normdiff = []

k = 0
while k < len(combis_list):

    ## Configuration

    ## Set up the environment
    """
    We will now create and define the settings for the environment of our simulation. In particular, this covers 
    the creation of (celestial) bodies, vehicle(s), and environment interfaces.
    """

    ### Create the main bodies
    """
    To create the systems of bodies for the simulation, one first has to define a list of strings of all bodies 
    that are to be included. Note that the default body settings (such as atmosphere, body shape, rotation model) 
    are taken from the `SPICE` kernel.
    These settings, however, can be adjusted. Please refer to the [Available Environment Models]
    (https://tudat-space.readthedocs.io/en/latest/_src_user_guide/state_propagation/environment_setup/create_models/available.html#available-environment-models) 
    in the user guide for more details.
    Finally, the system of bodies is created using the settings. This system of bodies is stored into the variable `bodies`.
    """

    # Set simulation start and end epochs
    start_gco = 0. * constants.JULIAN_YEAR  # beginning circular orbital phase 35.3844 * constants.JULIAN_YEAR
    # noinspection PyUnboundLocalVariable
    end_gco = start_gco + 1.35 * constants.JULIAN_DAY  # 13.5 * constants.JULIAN_DAY # 35.73 * constants.JULIAN_YEAR  # end circular orbital phase

    # Create default body settings for bodies_to_create, with "Enceladus"/"J2000" as the global frame origin and orientation
    global_frame_origin = "Enceladus"
    global_frame_orientation = "J2000"
    bodies_to_propagate = ["Orbiter"]
    central_bodies = ["Enceladus"]

    # Create default body settings for "Sun", "Enceladus", and "Saturn"
    bodies_to_create = ["Sun", "Enceladus", "Saturn"]
    body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin,
                                                                global_frame_orientation)


    def get_gravity_enceladus():

        mu_enceladus = 7.211292085479989E+9
        radius_enceladus = 252240.0
        cosine_coef = np.zeros((10, 10))
        sine_coef = np.zeros((10, 10))

        cosine_coef[0, 0] = 1.0

        cosine_coef[2, 0] = 5.4352E-03 / gravitation.legendre_normalization_factor(2, 0)  # wrong, correct is -5.4 ...
        cosine_coef[2, 1] = 9.2E-06 / gravitation.legendre_normalization_factor(2, 1)
        cosine_coef[2, 2] = 1.5498E-03 / gravitation.legendre_normalization_factor(2, 2)

        cosine_coef[3, 0] = -1.15E-04 / gravitation.legendre_normalization_factor(3, 0)  # wrong, correct is 1.15 ...

        sine_coef[2, 1] = 3.98E-05 / gravitation.legendre_normalization_factor(2, 1)
        sine_coef[2, 2] = 2.26E-05 / gravitation.legendre_normalization_factor(2, 2)

        return environment_setup.gravity_field.spherical_harmonic(mu_enceladus, radius_enceladus, cosine_coef,
                                                                  sine_coef, "IAU_Enceladus")


    # Define the spherical harmonics gravity model for Saturn
    saturn_gravitational_parameter = 3.7931208E+16
    saturn_reference_radius = 60330000.0

    # Normalize the spherical harmonic coefficients
    nor_sh_sat = astro.gravitation.normalize_spherical_harmonic_coefficients(
        [  # Iess et al. 2019, as in the minimal example by Andreas
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-16290.71E-6, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [935.83E-6, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-86.14E-6, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [10.E-6, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [  # Iess et al. 2019, as in the minimal example by Andreas
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])

    # Assign normalized cosine and sine coefficients
    saturn_normalized_cosine_coefficients = nor_sh_sat[0]
    saturn_normalized_sine_coefficients = nor_sh_sat[1]

    saturn_associated_reference_frame = "IAU_Saturn"

    # Create the gravity field settings and add them to the body "Saturn"
    body_settings.get("Saturn").gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
        saturn_gravitational_parameter,
        saturn_reference_radius,
        saturn_normalized_cosine_coefficients,
        saturn_normalized_sine_coefficients,
        saturn_associated_reference_frame)

    # Add setting for moment of inertia for Saturn
    body_settings.get("Saturn").gravity_field_settings.scaled_mean_moment_of_inertia = 0.210

    # Compute rotation rate for Enceladus
    mu_saturn = saturn_gravitational_parameter  # spice.get_body_properties("Saturn", "GM", 1)[0] * 1.0e9
    initial_state_enceladus = spice.get_body_cartesian_state_at_epoch("Enceladus", "Saturn", "J2000", "None", start_gco)
    keplerian_state_enceladus = element_conversion.cartesian_to_keplerian(initial_state_enceladus, mu_saturn)
    rotation_rate_enceladus = np.sqrt(mu_saturn / keplerian_state_enceladus[0] ** 3)

    # Set rotation model settings Enceladus
    initial_orientation_enceladus = spice.compute_rotation_matrix_between_frames("J2000", "IAU_Enceladus", start_gco)
    body_settings.get("Enceladus").rotation_model_settings = environment_setup.rotation_model.simple(
        "J2000", "IAU_Enceladus", initial_orientation_enceladus, start_gco, rotation_rate_enceladus)

    # Gravity field settings
    body_settings.get("Enceladus").gravity_field_settings = get_gravity_enceladus()

    # bodies.create_empty_body("Orbiter")
    # Create empty settings for RaTNOS Orbiter
    body_settings.add_empty_settings("Orbiter")

    # Create empty multi-arc ephemeris for RaTNOS orbiter
    empty_ephemeris_dict = dict()
    orbiter_ephemeris = environment_setup.ephemeris.tabulated(
        empty_ephemeris_dict,
        global_frame_origin,
        global_frame_orientation)
    orbiter_ephemeris.make_multi_arc_ephemeris = True
    body_settings.get("Orbiter").ephemeris_settings = orbiter_ephemeris

    # Create system of bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    ### Create the vehicle and its environment interface
    """
    We will now create the satellite - called Orbiter - for which an orbit will be simulated. Using an `empty_body` 
    as a blank canvas for the satellite, we define mass of 2150 kg, a reference area (used both for aerodynamic 
    and radiation pressure) of 100 m$^2$, and a aerodynamic drag coefficient of 1.2. Idem for the radiation 
    pressure coefficient. Finally, when setting up the radiation pressure interface, Enceladus is set as a body 
    that can occult the radiation emitted by the Sun.
    """

    # Create vehicle objects.
    bodies.get("Orbiter").mass = 2150.0

    # Create aerodynamic coefficient interface settings
    reference_area = 0.0  # 100.0 #(4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
    drag_coefficient = 1.2
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area, [drag_coefficient, 0.0, 0.0]
    )
    # Add the aerodynamic interface to the environment
    environment_setup.add_aerodynamic_coefficient_interface(bodies, "Orbiter", aero_coefficient_settings)

    # Create radiation pressure settings
    reference_area = 100.0  # (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
    radiation_pressure_coefficient = 1.2
    occulting_bodies = ["Enceladus"]  # occulting_bodies = {"Sun": ["Enceladus"]}
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area, radiation_pressure_coefficient, occulting_bodies
    )
    # Add the radiation pressure interface to the environment
    environment_setup.add_radiation_pressure_interface(bodies, "Orbiter", radiation_pressure_settings)

    ## Set up the propagation
    """
    Having the environment created, we will define the settings for the propagation of the spacecraft. 
    First, we have to define the body to be propagated - here, the spacecraft - and the central body - here, 
    Enceladus - with respect to which the state of the propagated body is defined.
    """

    ### Create the acceleration model
    """
    Subsequently, all accelerations (and there settings) that act on `Orbiter` have to be defined. In particular, we will consider:
    * Gravitational acceleration using a spherical harmonic approximation for Enceladus and Saturn.
    * Empirical acceleration for Enceladus.
    * Gravitational acceleration using a simple point mass model for the Sun.
    * Radiation pressure experienced by the spacecraft - shape-wise approximated as a spherical cannonball - due to the Sun.

    The defined acceleration settings are then applied to `Orbiter` by means of a dictionary, which is finally used 
    as input to the propagation setup to create the acceleration models.
    """

    # Define the accelerations acting on Orbiter
    accelerations_settings_orbiter = dict(
        Sun=[
            propagation_setup.acceleration.cannonball_radiation_pressure(),
            # propagation_setup.acceleration.radiation_pressure()
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Saturn=[
            propagation_setup.acceleration.spherical_harmonic_gravity(8, 8)
        ],
        Enceladus=[
            propagation_setup.acceleration.spherical_harmonic_gravity(3, 3),
            propagation_setup.acceleration.empirical()
        ])

    # Create global accelerations dictionary
    acceleration_settings = {"Orbiter": accelerations_settings_orbiter}

    # Create acceleration models
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    # Define propagation arcs during GCO (one day long) Enceladus
    arc_duration = 0.8 * (end_gco - start_gco)  # 0.5 * constants.JULIAN_DAY
    print('arc duration', arc_duration / 3600.0)

    arc_start_times = []
    arc_end_times = []
    arc_start = start_gco
    while arc_start + arc_duration <= end_gco:
        arc_start_times.append(arc_start)
        arc_end_times.append(arc_start + arc_duration)
        arc_start += arc_duration

    # Extract total number of (propagation) arcs during GCO
    nb_arcs = len(arc_start_times)
    print('Number of arcs during GCO', nb_arcs)

    ### Define the initial states for Orbiter wrt. Enceladus.
    """
    Realise that the initial state of the spacecraft always has to be provided as a cartesian state - i.e. in the form 
    of a list with the first three elements representing the initial position, and the three remaining elements 
    representing the initial velocity.
    """
    # The initial states need to be provided at the start of each propagation arc. We need to provide the initial states
    # for stable orbits in an inertial frame! Take K2 in body-fixed frame, transform to an inertial frame.

    # Get rotation matrix between IAU_Enceladus and global_frame_orientation
    rotation_matrix = spice.compute_rotation_matrix_between_frames("IAU_Enceladus", global_frame_orientation,
                                                                   arc_start_times[0])
    rotation_matrix_back = spice.compute_rotation_matrix_between_frames(global_frame_orientation, "IAU_Enceladus",
                                                                        arc_start_times[0])

    # Assign initial state in Cartesian coordinates in inertial frame
    initial_state = np.ndarray([6])
    # initial_state[0:3] = [475323.709, 102991.720, -48576.955]
    # initial_state[3:6] = [3.009, 75.062, 95.705]

    # initial_state = [4.37860747e+05, 9.48716723e+04, -4.47516654e+04, 6.67354927e-01, 7.94830037e+01,  9.86040217e+01] # k1'
    initial_state = [4.71789124e+05, 1.02222980e+05, -4.82193361e+04, 3.14085658e+00, 7.66580975e+01,
                     9.50418662e+01]  # k2'
    # initial_state = [4.60709175e+05,  9.98223410e+04, -4.70869558e+04,  2.46463058e+00, 7.56765950e+01,  9.76068794e+01] # k3'

    # print("initial state cartesian inertial")
    # print(initial_state)
    # print("initial state cartesian fixed rotated")
    # print(rotation_matrix_back.dot(initial_state[0:3]))

    initial_states = []
    for l in range(nb_arcs):
        initial_states.append(initial_state)

    ### Create the integrator settings
    """
    For the problem at hand, we will use an RKF78 integrator with a fixed step-size of 60 seconds. 
    This can be achieved by tweaking the implemented RKF78 integrator with variable step-size such that both 
    the minimum and maximum step-size is equal to 60 seconds and a tolerance of 1.0
    """
    # Create numerical integrator settings
    integrator_settings = propagation_setup.integrator. \
        runge_kutta_fixed_step_size(initial_time_step=10.0,
                                    coefficient_set=propagation_setup.integrator.CoefficientSets.rkdp_87)

    # Define dependent variables to be saved during propagation
    """dependent_variables_names = [
        propagation_setup.dependent_variable.latitude("Orbiter", "Enceladus"),
        propagation_setup.dependent_variable.longitude("Orbiter", "Enceladus"),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.spherical_harmonic_gravity_type, "Orbiter", "Enceladus"
        ),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.spherical_harmonic_gravity_type, "Orbiter", "Saturn"
        ),
        propagation_setup.dependent_variable.total_acceleration("Orbiter"),
        propagation_setup.dependent_variable.keplerian_state("Orbiter", "Enceladus"),
        propagation_setup.dependent_variable.altitude("Orbiter", "Enceladus")
    ]"""

    ### Create the propagator settings
    """
    By combining all of the above-defined settings we can define the settings for the propagator to simulate 
    the orbit of `Orbiter` around Enceladus. A termination condition needs to be defined so that the propagation stops 
    as soon as the specified end epoch is reached. Finally, the translational propagator's settings are created.
    """
    # Create termination settings
    termination_condition = propagation_setup.propagator.time_termination(end_gco)

    # Create propagation settings
    # Define arc-wise propagator settings
    propagator_settings_list = []
    for j in range(nb_arcs):
        propagator_settings_list.append(propagation_setup.propagator.translational(
            central_bodies, acceleration_models, bodies_to_propagate, initial_states[j], arc_start_times[j],
            integrator_settings, propagation_setup.propagator.time_termination(arc_end_times[j])))  # ,
    #        propagation_setup.propagator.cowell, dependent_variables_names)

    # Concatenate all arc-wise propagator settings into multi-arc propagator settings
    #propagator_settings = propagation_setup.propagator.multi_arc(propagator_settings_list)

    k += 1
    print('k =',k)

print(normdiff)
